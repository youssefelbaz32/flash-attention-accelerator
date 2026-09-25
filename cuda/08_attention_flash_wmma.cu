// Milestone 8, step 3: the same online-softmax attention on TENSOR CORES.
//
// WHY. After coalescing and query tiling (07), N=4096 D=128 runs at 1.13 TFLOP/s,
// 9.6% of the FP32 CUDA-core ceiling, and the two matmuls (QK^T and PV) are now
// most of the work. The measured fp16 tensor-core ceiling on the same card is
// 39.2 TFLOP/s. This kernel moves both matmuls onto tensor cores through WMMA
// (16x16x16, fp16 in, fp32 accumulate) and keeps the softmax in fp32.
//
// MAPPING:
//   one WARP  <-> 16 query rows (one WMMA row tile), fully independent
//   one BLOCK <-> WARPS such warps; there is no __syncthreads() anywhere,
//                 every shared buffer is private to one warp
//   one tile  <-> BC = 32 keys
//
// Per tile, per warp:
//   S  = Q_w K^T           2 x (D/16) mma      -> sS, fp32 16x32
//   online softmax on sS   2 lanes per row, pair shuffles for max and sum
//   O  = O * corr + P V    O lives in shared, fp32 16xD; the accumulator is
//                          loaded, multiplied into, and stored back, because
//                          WMMA's fragment layout is opaque and the rescale
//                          by corr is per ROW
//
// Q, K and V are read straight from global as WMMA operands. K stored row-major
// N x D IS K^T in column-major, so no transpose is ever materialized.
//
// Inputs are fp16, so the error floor is fp16's, not fp32's: the reference is
// computed in float64 from the SAME fp16-rounded inputs, and P is rounded to
// fp16 before the PV product, exactly as FlashAttention does.
//
// BUILD:  nvcc -O3 -arch=native -o flash_wmma cuda/08_attention_flash_wmma.cu
//   ./flash_wmma 4096 128 0   one case
//   ./flash_wmma bench        the same sweep as ./flash bench (D >= 32)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
      cudaError_t err = (call);                                    \
      if (err != cudaSuccess) {                                    \
          fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                  cudaGetErrorString(err), __FILE__, __LINE__);    \
          exit(1);                                                 \
      }                                                            \
  } while (0)

#ifndef WARPS
#define WARPS 2       // warps per block; each owns 16 rows independently.
                      // Measured on sm_89: 2 and 4 tie at large N, 2 wins small
                      // and causal (finer blocks); 8 exceeds 48 KB at D=128.
#endif
#define BR 16         // query rows per warp = WMMA M
#define BC 32         // keys per tile = two WMMA N tiles
#define ROWS_PER_BLOCK (WARPS * BR)

// =============================================================================
// the kernel, templated on head dim so every loop over D/16 fully unrolls
// -----------------------------------------------------------------------------
template <int D>
__global__ void attention_flash_wmma(const half* __restrict__ Q,
                                     const half* __restrict__ K,
                                     const half* __restrict__ V,
                                     float* __restrict__ O,
                                     int N, int causal) {
    extern __shared__ float smem[];
    __shared__ __align__(32) half sP_all[WARPS][BR * BC];

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* sS = smem + warp * (BR * BC + BR * D);   // 16 x BC scores, fp32
    float* sO = sS + BR * BC;                       // 16 x D  output accumulator
    half*  sP = sP_all[warp];                       // 16 x BC probabilities, fp16

    const int r0 = (blockIdx.x * WARPS + warp) * BR;
    if (r0 >= N) return;             // no block barriers exist, so this is safe

    // Q tile stays in registers as WMMA A-fragments for the whole key loop.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
    #pragma unroll
    for (int kk = 0; kk < D / 16; kk++)
        wmma::load_matrix_sync(qf[kk], Q + (size_t)r0 * D + kk * 16, D);
    for (int e = lane; e < BR * D; e += 32) sO[e] = 0.0f;

    // Lanes 2r and 2r+1 own row r: each takes half the tile's columns for the
    // softmax and half of D for the rescale. Both keep identical m and l.
    const int r = lane >> 1, hlf = lane & 1;
    const int row = r0 + r;
    const float scale = rsqrtf((float)D);
    float m_run = -INFINITY, l_run = 0.0f;

    // Causal: this warp's last row is r0+15. Tile starts are multiples of 32
    // and r0 of 16, so every processed tile starts at or before r0, and every
    // row sees at least one unmasked key -- m never stays -inf.
    const int j_end = causal ? min(N, r0 + BR) : N;

    for (int tile = 0; tile < j_end; tile += BC) {
        // ---- S = Q K^T on tensor cores --------------------------------------
        #pragma unroll
        for (int n = 0; n < BC / 16; n++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
            wmma::fill_fragment(sf, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / 16; kk++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, K + (size_t)(tile + n * 16) * D + kk * 16, D);
                wmma::mma_sync(sf, qf[kk], kf, sf);
            }
            wmma::store_matrix_sync(sS + n * 16, sf, BC, wmma::mem_row_major);
        }
        __syncwarp();

        // ---- online softmax, fp32 --------------------------------------------
        const float* srow = sS + r * BC + hlf * 16;
        const int jb = tile + hlf * 16;
        float v[16], m_tile = -INFINITY;
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            const int j = jb + c;
            const bool ok = (j < N) && (!causal || j <= row);
            v[c] = ok ? srow[c] * scale : -INFINITY;
            m_tile = fmaxf(m_tile, v[c]);
        }
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1));
        const float m_new = fmaxf(m_run, m_tile);
        const float corr  = __expf(m_run - m_new);  // 0 on the first tile
        float l_tile = 0.0f;
        half* prow = sP + r * BC + hlf * 16;
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            const float p = __expf(v[c] - m_new);    // masked -> exp(-inf) = 0
            l_tile += p;
            prow[c] = __float2half(p);
        }
        l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1);
        l_run = l_run * corr + l_tile;
        m_run = m_new;
        float* orow = sO + r * D;
        for (int c = hlf; c < D; c += 2) orow[c] *= corr;
        __syncwarp();

        // ---- O += P V on tensor cores ----------------------------------------
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf[BC / 16];
        #pragma unroll
        for (int kk = 0; kk < BC / 16; kk++) wmma::load_matrix_sync(pf[kk], sP + kk * 16, BC);
        #pragma unroll
        for (int dn = 0; dn < D / 16; dn++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
            wmma::load_matrix_sync(of, sO + dn * 16, D, wmma::mem_row_major);
            #pragma unroll
            for (int kk = 0; kk < BC / 16; kk++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, V + (size_t)(tile + kk * 16) * D + dn * 16, D);
                wmma::mma_sync(of, pf[kk], vf, of);
            }
            wmma::store_matrix_sync(sO + dn * 16, of, D, wmma::mem_row_major);
        }
        __syncwarp();
    }

    if (row < N) {
        const float inv_l = 1.0f / l_run;
        for (int c = hlf; c < D; c += 2) O[(size_t)row * D + c] = sO[r * D + c] * inv_l;
    }
}

// =============================================================================
// host
// -----------------------------------------------------------------------------
void reference_cpu(const float* Q, const float* K, const float* V, double* O,
                   int N, int D, int causal) {
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
        for (int c = 0; c < D; c++) {
            double v = 0.0;
            for (int j = 0; j < j_end; j++) v += s[j] * V[(size_t)j*D+c];
            O[(size_t)i*D+c] = v / denom;
        }
    }
}

template <int D>
void launch(const half* dQ, const half* dK, const half* dV, float* dO, int N, int causal) {
    const int grid = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const size_t shmem = (size_t)WARPS * (BR * BC + BR * D) * sizeof(float);
    attention_flash_wmma<D><<<grid, WARPS * 32, shmem>>>(dQ, dK, dV, dO, N, causal);
}

struct Result { double ms; double gflops; double max_err; };

Result run_case(int N, int D, int causal, bool check, int iters) {
    // Pad N to a whole number of block row tiles and key tiles with zeros, so
    // every WMMA load is in bounds. Padded keys are masked in the softmax and
    // padded rows are never stored.
    const int Npad = ((N + 63) / 64) * 64;
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hQ((size_t)N*D), hK((size_t)N*D), hV((size_t)N*D);
    for (auto& x : hQ) x = nd(rng);
    for (auto& x : hK) x = nd(rng);
    for (auto& x : hV) x = nd(rng);
    std::vector<half> pQ((size_t)Npad*D, __float2half(0.f)), pK = pQ, pV = pQ;
    for (size_t t = 0; t < hQ.size(); t++) {
        pQ[t] = __float2half(hQ[t]); hQ[t] = __half2float(pQ[t]);   // reference sees
        pK[t] = __float2half(hK[t]); hK[t] = __half2float(pK[t]);   // the same fp16
        pV[t] = __float2half(hV[t]); hV[t] = __half2float(pV[t]);   // values
    }

    half *dQ, *dK, *dV; float* dO;
    size_t hb = (size_t)Npad * D * sizeof(half);
    CUDA_CHECK(cudaMalloc(&dQ, hb)); CUDA_CHECK(cudaMalloc(&dK, hb)); CUDA_CHECK(cudaMalloc(&dV, hb));
    CUDA_CHECK(cudaMalloc(&dO, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, pQ.data(), hb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, pK.data(), hb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, pV.data(), hb, cudaMemcpyHostToDevice));

    auto go = [&]() {
        switch (D) {
            case 32:  launch<32>(dQ, dK, dV, dO, N, causal); break;
            case 64:  launch<64>(dQ, dK, dV, dO, N, causal); break;
            case 128: launch<128>(dQ, dK, dV, dO, N, causal); break;
            default: fprintf(stderr, "D must be 32, 64 or 128\n"); exit(1);
        }
    };
    go();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++) go();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
    double ms = ms_total / iters;

    double max_err = -1.0;
    if (check) {
        std::vector<float> hO((size_t)N * D);
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<double> ref(hO.size());
        reference_cpu(hQ.data(), hK.data(), hV.data(), ref.data(), N, D, causal);
        max_err = 0.0;
        for (size_t t = 0; t < hO.size(); t++) max_err = fmax(max_err, fabs((double)hO[t] - ref[t]));
    }
    double pairs = causal ? ((double)N*(N+1)/2.0) : ((double)N*N);
    double flops = 4.0 * pairs * D;
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return { ms, flops / (ms * 1e6), max_err };
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (sm_%d%d, %d SMs)  fp16 WMMA, %d warps/block\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, WARPS);

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        printf("\n%6s %5s %7s %10s %10s %12s\n", "N", "D", "causal", "ms", "GFLOP/s", "max_err");
        int Ns[] = {128, 512, 2048, 4096};
        int Ds[] = {32, 64, 128};
        for (int ci = 0; ci <= 1; ci++)
          for (int ni = 0; ni < 4; ni++)
            for (int di = 0; di < 3; di++) {
                int N = Ns[ni], D = Ds[di];
                bool check = (N <= 512);
                Result r = run_case(N, D, ci, check, 20);
                printf("%6d %5d %7d %10.4f %10.1f   ", N, D, ci, r.ms, r.gflops);
                if (check) printf("%12.3e\n", r.max_err); else printf("%12s\n", "(skipped)");
            }
        return 0;
    }
    int N = (argc > 1) ? atoi(argv[1]) : 300;
    int D = (argc > 2) ? atoi(argv[2]) : 64;
    int causal = (argc > 3) ? atoi(argv[3]) : 1;
    Result r = run_case(N, D, causal, N <= 1024, 20);
    printf("\nN=%d D=%d causal=%d : %.4f ms  %.1f GFLOP/s", N, D, causal, r.ms, r.gflops);
    if (r.max_err >= 0) printf("  max_err=%.3e", r.max_err);
    printf("\n");
    return 0;
}
