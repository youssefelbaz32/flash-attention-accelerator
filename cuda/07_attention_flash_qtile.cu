// Milestone 8, step 2: the same online-softmax attention, with QUERY TILING.
//
// WHY. Nsight Compute and a measured roofline on an RTX 4070 Laptop GPU put
// 06_attention_flash.cu at 0.44 FLOP per L2 byte: each of the N blocks owns ONE
// query row and streams all of K and V through L2 for it, 19.4 GB at N=4096,
// D=128. Staging K through shared memory (the coalescing fix in 06) took the
// L1 limit away and left this one. The only way to move right on the roofline
// is to make each byte of K and V serve more than one query.
//
// MAPPING:
//   one WARP   <-> one query row      (owns m, l and DV/32 accumulators)
//   one BLOCK  <-> WR query rows      (WR warps share every K/V tile)
//   one tile   <-> BC = 32 keys       (lane l scores key tile+l)
//
// K and V are loaded into shared memory once per tile and read by all WR warps,
// so L2 traffic drops by WR. Every reduction is inside one warp, done with
// shuffles, so there is no block-wide scratch at all -- the class of race that
// 06 had between blockReduceMax and blockReduceSum cannot occur here. The only
// barriers guard the shared K/V tile.
//
// Same recurrence as rtl/flash_top.sv and 06:
//     m_new = max(m_run, max_j s_j);  corr = exp(m_run - m_new)
//     l_run = l_run*corr + sum_j exp(s_j - m_new)
//     acc_c = acc_c*corr + sum_j exp(s_j - m_new) * V[j][c]
//     O_c   = acc_c / l_run
//
// BUILD:  nvcc -O3 -arch=native -o flash_qtile cuda/07_attention_flash_qtile.cu
//   ./flash_qtile              # toy dims vs data/O_golden.npy
//   ./flash_qtile 4096 128 1   # one case
//   ./flash_qtile bench        # the same sweep as ./flash bench

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
      cudaError_t err = (call);                                    \
      if (err != cudaSuccess) {                                    \
          fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                  cudaGetErrorString(err), __FILE__, __LINE__);    \
          exit(1);                                                 \
      }                                                            \
  } while (0)

#ifndef WR
#define WR 16         // query rows (warps) per block: the L2 traffic divisor.
                      // Measured on sm_89: 4 -> 25.8 ms, 8 -> 10.4, 16 -> 7.6 at
                      // N=4096 D=128. 32 needs more than 48 KB of shared memory.
#endif
#define BC 32         // keys per tile = lanes per warp
#define MAX_D 128     // largest head dim supported (accumulators are DV/32 per lane)
#define NACC (MAX_D / 32)

// =============================================================================
// the kernel
//   grid  = ceil(N / WR) blocks, block = WR*32 threads
//   shared = sQ[WR][D] + sK[BC][D+1] + sV[BC][DV]
// -----------------------------------------------------------------------------
__global__ void attention_flash_qtile(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O,
                                      int N, int D, int DV, int causal) {
    extern __shared__ float smem[];
    float* sQ = smem;                    // WR x D     the block's query rows
    float* sK = sQ + WR * D;             // BC x (D+1) padded: lane l reads row l
    float* sV = sK + BC * (D + 1);       // BC x DV    lanes read consecutive columns

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = blockIdx.x * WR;
    const int i    = row0 + warp;        // THIS WARP OWNS QUERY ROW i
    const bool live = (i < N);           // tail warps still help load, never store

    for (int e = threadIdx.x; e < WR * D; e += blockDim.x) {
        const int r = e / D, k = e % D;
        sQ[e] = (row0 + r < N) ? Q[(size_t)(row0 + r) * D + k] : 0.0f;
    }

    const float scale = rsqrtf((float)D);
    float m_run = -INFINITY, l_run = 0.0f;
    float acc[NACC];
    #pragma unroll
    for (int a = 0; a < NACC; a++) acc[a] = 0.0f;

    // This warp's keys stop at i+1 when causal; the BLOCK must keep loading
    // tiles until its last row is done, so the tile loop runs to the block max.
    const int j_end_row   = causal ? min(i + 1, N) : N;
    const int j_end_block = causal ? min(row0 + WR, N) : N;

    for (int tile = 0; tile < j_end_block; tile += BC) {
        const int rows = min(BC, N - tile);
        __syncthreads();                 // previous tile fully consumed (and sQ ready)
        // Coalesced: consecutive threads take consecutive floats of one row.
        for (int e = threadIdx.x; e < BC * D; e += blockDim.x) {
            const int r = e / D, k = e % D;
            sK[r * (D + 1) + k] = (r < rows) ? K[(size_t)(tile + r) * D + k] : 0.0f;
        }
        for (int e = threadIdx.x; e < BC * DV; e += blockDim.x) {
            const int r = e / DV, c = e % DV;
            sV[e] = (r < rows) ? V[(size_t)(tile + r) * DV + c] : 0.0f;
        }
        __syncthreads();

        if (!live || tile >= j_end_row) continue;    // no barrier inside: safe

        // -- score for key tile+lane ------------------------------------------
        const int j = tile + lane;
        const bool valid = (j < j_end_row);
        float dot = 0.0f;
        const float* q = sQ + warp * D;              // broadcast read
        const float* k = sK + lane * (D + 1);        // bank (lane + kk) % 32
        for (int kk = 0; kk < D; kk++) dot += q[kk] * k[kk];
        const float s = valid ? dot * scale : -INFINITY;

        // -- the rebase, all inside one warp ----------------------------------
        float m_tile = s;
        for (int off = 16; off > 0; off >>= 1)
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));
        const float m_new = fmaxf(m_run, m_tile);
        const float corr  = __expf(m_run - m_new);   // 0 on the first tile
        const float p     = valid ? __expf(s - m_new) : 0.0f;
        float l_tile = p;
        for (int off = 16; off > 0; off >>= 1)
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, off);
        l_run = l_run * corr + l_tile;

        // -- acc_c = acc_c*corr + sum_t p_t * V[t][c]; lane owns c = lane+32a --
        #pragma unroll
        for (int a = 0; a < NACC; a++) acc[a] *= corr;
        const int t_end = min(BC, j_end_row - tile);
        for (int t = 0; t < t_end; t++) {
            const float pt = __shfl_sync(0xffffffff, p, t);
            #pragma unroll
            for (int a = 0; a < NACC; a++) {
                const int c = lane + 32 * a;
                if (c < DV) acc[a] += pt * sV[t * DV + c];
            }
        }
        m_run = m_new;
    }

    if (live) {
        const float inv_l = 1.0f / l_run;
        #pragma unroll
        for (int a = 0; a < NACC; a++) {
            const int c = lane + 32 * a;
            if (c < DV) O[(size_t)i * DV + c] = acc[a] * inv_l;
        }
    }
}

// =============================================================================
// host: .npy loader, float64 reference, one measured run, main
// -----------------------------------------------------------------------------
float* load_npy(const char* path, int rows, int cols) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open file: %s\n", path); return nullptr; }
    float* data = (float*) malloc((size_t)rows * cols * sizeof(float));
    fseek(f, 8, SEEK_SET);
    uint16_t hdr_len;
    if (fread(&hdr_len, sizeof(uint16_t), 1, f) != 1) { free(data); fclose(f); return nullptr; }
    fseek(f, hdr_len, SEEK_CUR);
    if ((size_t)rows * cols != fread(data, sizeof(float), (size_t)rows * cols, f)) {
        free(data); fclose(f); return nullptr;
    }
    fclose(f);
    return data;
}

// Two-pass float64 reference, deliberately NOT the online algorithm.
void reference_cpu(const float* Q, const float* K, const float* V, double* O,
                   int N, int D, int DV, int causal) {
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
        double denom = 0.0;
        for (int j = 0; j < j_end; j++) { s[j] = exp(s[j] - m); denom += s[j]; }
        for (int c = 0; c < DV; c++) {
            double v = 0.0;
            for (int j = 0; j < j_end; j++) v += s[j] * V[(size_t)j*DV+c];
            O[(size_t)i*DV+c] = v / denom;
        }
    }
}

struct Result { double ms; double gflops; double max_err; };

Result run_case(int N, int D, int DV, int causal, bool check, int iters,
                const float* hQ_in, const float* hK_in, const float* hV_in) {
    if (D > MAX_D || DV > MAX_D) { fprintf(stderr, "D, DV must be <= %d\n", MAX_D); exit(1); }
    size_t szQ = (size_t)N*D, szV = (size_t)N*DV;
    std::vector<float> hQ, hK, hV;
    const float *Qp = hQ_in, *Kp = hK_in, *Vp = hV_in;
    if (!Qp) {   // same generator and seed as 06, so the two kernels see identical inputs
        std::mt19937 rng(0);
        std::normal_distribution<float> nd(0.f, 1.f);
        hQ.resize(szQ); hK.resize(szQ); hV.resize(szV);
        for (auto& x : hQ) x = nd(rng);
        for (auto& x : hK) x = nd(rng);
        for (auto& x : hV) x = nd(rng);
        Qp = hQ.data(); Kp = hK.data(); Vp = hV.data();
    }

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, szQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, szQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, szV*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, szV*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Qp, szQ*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, Kp, szQ*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, Vp, szV*sizeof(float), cudaMemcpyHostToDevice));

    size_t shmem = (size_t)(WR * D + BC * (D + 1) + BC * DV) * sizeof(float);
    int grid = (N + WR - 1) / WR;

    attention_flash_qtile<<<grid, WR * 32, shmem>>>(dQ, dK, dV, dO, N, D, DV, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++)
        attention_flash_qtile<<<grid, WR * 32, shmem>>>(dQ, dK, dV, dO, N, D, DV, causal);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
    double ms = ms_total / iters;

    std::vector<float> hO(szV);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, szV*sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = -1.0;
    if (check) {
        std::vector<double> ref(szV);
        reference_cpu(Qp, Kp, Vp, ref.data(), N, D, DV, causal);
        max_err = 0.0;
        for (size_t t = 0; t < szV; t++) max_err = fmax(max_err, fabs((double)hO[t] - ref[t]));
    }
    double pairs = causal ? ((double)N*(N+1)/2.0) : ((double)N*N);
    double flops = 2.0*pairs*D + 2.0*pairs*DV;
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return { ms, flops/(ms*1e6), max_err };
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (sm_%d%d, %d SMs)  WR=%d rows/block\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, WR);

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        printf("\n%6s %5s %7s %10s %10s %12s\n", "N", "D", "causal", "ms", "GFLOP/s", "max_err");
        int Ns[] = {128, 512, 2048, 4096};
        int Ds[] = {32, 64, 128};
        for (int ci = 0; ci <= 1; ci++)
          for (int ni = 0; ni < 4; ni++)
            for (int di = 0; di < 3; di++) {
                int N = Ns[ni], D = Ds[di];
                bool check = (N <= 512);
                Result r = run_case(N, D, D, ci, check, 20, nullptr, nullptr, nullptr);
                printf("%6d %5d %7d %10.4f %10.1f   ", N, D, ci, r.ms, r.gflops);
                if (check) printf("%12.3e\n", r.max_err); else printf("%12s\n", "(skipped)");
            }
        return 0;
    }

    int N = (argc > 1) ? atoi(argv[1]) : 4;
    int D = (argc > 2) ? atoi(argv[2]) : 4;
    int causal = (argc > 3) ? atoi(argv[3]) : 0;
    if (N == 4 && D == 4 && !causal) {
        float* hQ = load_npy("data/Q.npy", 4, 4);
        float* hK = load_npy("data/K.npy", 4, 4);
        float* hV = load_npy("data/V.npy", 4, 4);
        if (hQ && hK && hV) {
            Result r = run_case(4, 4, 4, 0, true, 100, hQ, hK, hV);
            printf("\ntoy dims vs float64 reference : max abs err = %.6e  (%.4f ms)\n", r.max_err, r.ms);
            free(hQ); free(hK); free(hV);
            return 0;
        }
    }
    Result r = run_case(N, D, D, causal, N <= 512, 20, nullptr, nullptr, nullptr);
    printf("\nN=%d D=%d causal=%d : %.4f ms  %.1f GFLOP/s", N, D, causal, r.ms, r.gflops);
    if (r.max_err >= 0) printf("  max_err=%.3e", r.max_err);
    printf("\n");
    return 0;
}
