// CPU test harness for the M8 fused kernel (cuda/06_attention_flash.cu).
//
// Launches the real kernel body under cpu_emu.h: one std::thread per CUDA
// thread, blocks executed sequentially, __syncthreads() backed by a real
// barrier. Diffs the result against an independent float64 two-pass reference.
//
// The reference is deliberately NOT the online algorithm. A reference that
// mirrored the kernel's structure would reproduce the kernel's bugs; the point
// of a golden model is that it was arrived at differently.
//
// Build + run (no GPU needed):
//   c++ -std=c++17 -O2 -DCPU_EMU -o build/flash_cpu \
//       cuda/test_flash_cpu.cpp && ./build/flash_cpu

#define CPU_EMU 1
#include "cpu_emu.h"
#include <random>
#include <cstring>

thread_local uint3_emu threadIdx{0,0,0};
uint3_emu blockIdx{0,0,0}, blockDim{1,1,1};
float smem[EMU_SMEM_FLOATS];
Barrier* g_barrier = nullptr;

#include "06_attention_flash.cu"     // the kernel under test, unmodified

// independent float64 reference: plain two-pass softmax, textbook definition
static void reference_cpu2(const float* Q, const float* K, const float* V,
                           double* O, int N, int D, int DV, int causal) {
    std::vector<double> s(N);
    for (int i = 0; i < N; i++) {
        int j_end = causal ? (i + 1) : N;
        double m = -1e300;
        for (int j = 0; j < j_end; j++) {
            double dot = 0.0;
            for (int k = 0; k < D; k++) dot += (double)Q[(size_t)i*D+k] * K[(size_t)j*D+k];
            s[j] = dot / sqrt((double)D);
            if (s[j] > m) m = s[j];
        }
        double den = 0.0;
        for (int j = 0; j < j_end; j++) { s[j] = exp(s[j] - m); den += s[j]; }
        for (int c = 0; c < DV; c++) {
            double v = 0.0;
            for (int j = 0; j < j_end; j++) v += s[j] * V[(size_t)j*DV+c];
            O[(size_t)i*DV+c] = v / den;
        }
    }
}

// emulate <<<gridDim, blockDimX, shmemBytes>>>
static void launch(int grid, int bdim, size_t shmem_floats,
                   const float* Q, const float* K, const float* V, float* O,
                   int N, int D, int DV, int causal) {
    blockDim = {(unsigned)bdim, 1, 1};
    if (shmem_floats > EMU_SMEM_FLOATS) { printf("shared mem too large\n"); exit(1); }
    for (int b = 0; b < grid; b++) {
        blockIdx = {(unsigned)b, 1, 1};
        Barrier bar(bdim);
        g_barrier = &bar;
        std::vector<std::thread> th;
        th.reserve(bdim);
        for (int t = 0; t < bdim; t++)
            th.emplace_back([&, t]{
                threadIdx = {(unsigned)t, 1, 1};
                attention_flash(Q, K, V, O, N, D, DV, causal);
            });
        for (auto& x : th) x.join();
    }
}

static int run(int N, int D, int DV, int causal, double tol) {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> Q((size_t)N*D), K((size_t)N*D), V((size_t)N*DV), O((size_t)N*DV, 0.f);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    // BC threads per block is what the kernel assumes for its shared layout;
    // use a smaller block when N is small so the test stays quick.
    int bdim = (BC < ((32 > N) ? 32 : N)) ? BC : ((32 > N) ? 32 : N);
    launch(N, bdim, (size_t)D + BC + 32, Q.data(), K.data(), V.data(), O.data(),
           N, D, DV, causal);

    std::vector<double> ref((size_t)N*DV);
    reference_cpu2(Q.data(), K.data(), V.data(), ref.data(), N, D, DV, causal);

    double maxerr = 0.0;
    for (size_t t = 0; t < O.size(); t++)
        maxerr = std::max(maxerr, fabs((double)O[t] - ref[t]));

    bool ok = maxerr < tol;
    printf("  N=%-5d D=%-4d causal=%d  threads=%-4d  max_err=%.3e  %s\n",
           N, D, causal, bdim, maxerr, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    printf("M8 fused flash kernel -- CPU emulation (no GPU)\n");
    printf("  running the real kernel body under a threaded barrier model\n\n");
    int fails = 0;
    // tolerance is float32 accumulation error vs a float64 reference; it grows
    // with the number of terms summed, hence the looser bound at large N.
    fails += run(4,    4,  4,  0, 1e-6);
    fails += run(4,    4,  4,  1, 1e-6);
    fails += run(64,   32, 32, 0, 1e-5);
    fails += run(64,   32, 32, 1, 1e-5);
    fails += run(128,  64, 64, 0, 1e-5);
    fails += run(300,  64, 64, 0, 1e-5);   // not a multiple of BC: tail tile
    fails += run(300,  64, 64, 1, 1e-5);   // causal + ragged tail together
    fails += run(512, 128, 128, 0, 2e-5);
    printf("\n%s\n", fails == 0 ? "  ALL PASS -- kernel is correct before it ever sees a GPU"
                                : "  FAILURES");
    return fails != 0;
}
