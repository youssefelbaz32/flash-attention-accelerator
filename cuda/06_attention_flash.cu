// Milestone 8 (GPU half): FUSED single-head attention with ONLINE softmax (FlashAttention-1
// forward), float32. The same algorithm as rtl/flash_top.sv, on a GPU.
//
// WHY FUSED. M3 (naive) and M4 (tiled) both materialize the full N x N score
// row before normalizing, so they need O(N) registers or O(N^2) memory per
// query and they touch the scores TWICE (once to find the max, once to
// exponentiate). At N=4 that is free. At N=4096 the score row alone is 16 KB
// per query -- it does not fit anywhere near the ALUs, and attention becomes a
// memory problem rather than an arithmetic one.
//
// THE FIX, identical to M8's RTL: never materialize the row. Stream keys in
// tiles, carry a running (m, l, acc), and rebase the accumulator whenever a
// bigger max shows up:
//
//     m_new = max(m_run, max_j s_j)
//     corr  = exp(m_run - m_new)
//     l_run = l_run * corr + sum_j exp(s_j - m_new)
//     acc_c = acc_c * corr + sum_j exp(s_j - m_new) * V[j][c]
//     O_c   = acc_c / l_run            (once, at the very end)
//
// Per-query state is m, l and DV accumulators -- INDEPENDENT OF N. That is the
// whole trick, and it is the same sentence as the RTL commit.
//
// MAPPING TO THE GPU:
//   one thread BLOCK  <-> one query row       (owns the running state)
//   blockDim.x = BC   <-> one key tile        (thread t handles key tile+t)
//   the Q row lives in shared memory, loaded once and reused by every tile
//   K and V stream from global; L2 serves them to the blocks working in parallel
//
// The two block-wide reductions (max, then sum) are the GPU cost that the RTL
// does not pay: the FPGA keeps its running state in one register file, whereas
// here BC threads must agree on m_tile and l_tile every tile. That is what the
// __syncthreads() pairs below are buying.
//
// BUILD (from the project root):
//   nvcc -O3 -arch=native -o flash cuda/06_attention_flash.cu && ./flash
//   (-arch=native needs CUDA 11.5+; otherwise ./preflight.sh prints your sm_XX)
//   ./flash 4                 # toy dims, diffed against data/O_golden.npy
//   ./flash 4096 128 1        # N=4096, D=128, causal
//   ./flash bench             # the full sweep for the results table
//
// PROFILE:
//   ncu --set full -o flash_prof ./flash 4096 128 0

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#ifdef CPU_EMU
  #include "cpu_emu.h"     // run the SAME kernel source on the CPU, no GPU needed
#else
  #include <cuda_runtime.h>
#endif

#define CUDA_CHECK(call) do {                                      \
      cudaError_t err = (call);                                    \
      if (err != cudaSuccess) {                                    \
          fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                  cudaGetErrorString(err), __FILE__, __LINE__);    \
          exit(1);                                                 \
      }                                                            \
  } while (0)

// Threads per block = keys per tile. 128 is a reasonable default: 4 warps, so
// the block reductions stay cheap, and enough parallelism to hide global loads.
#define BC 128
// Max value dim held in registers per thread. DV <= BC keeps it at one
// accumulator per thread, which is the fast path.
#define MAX_DV 128
// K is staged through shared memory KC feature columns at a time. Reading K
// straight from global, thread tid walks its own key row, so the 32 lanes of a
// warp sit D floats apart and every load touches 32 different sectors. Nsight
// measured 7.1 useful bytes per 32-byte sector and L1 at 98% of peak. Staging
// lets consecutive threads load consecutive floats of one row instead.
#define KC 32

// =============================================================================
// CHUNK 1: block-wide reductions
//   Warp shuffles first (no memory, no barrier inside a warp), then one shared
//   slot per warp, then the first warp reduces those. Standard two-level shape.
// -----------------------------------------------------------------------------
#ifndef CPU_EMU
__inline__ __device__ float warpReduceMax(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__inline__ __device__ float warpReduceSum(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

#endif  // !CPU_EMU

#ifdef CPU_EMU
// On the CPU the warp does not exist, so the two BLOCK reducers are provided by
// cpu_emu.h using a shared array plus the same barriers. Everything BELOW this
// point -- the entire kernel body, the algorithm, the indexing -- is the
// identical source that nvcc compiles. The emulation boundary is exactly these
// two functions, and it is drawn here on purpose: a harness that rewrote the
// kernel would prove nothing about the kernel.
#endif

// Both reducers BROADCAST the result back to every thread, because every thread
// needs m_new and corr to rebase its own accumulator. A reduce-to-thread-0 form
// would need a third barrier to publish it.
//
// The broadcast goes through scratch[32], NOT scratch[0]. The kernel calls Max
// then Sum back to back with no barrier between them, so with a shared slot a
// fast warp 0 could write its partial sum into scratch[0] before a slow warp had
// read the max out of it -- that warp then rebases with the wrong m_new. Found
// on an RTX 4070 Laptop GPU: nondeterministic errors up to 0.28 at N=512. The
// CPU emulator runs threads in order and cannot see it. With a separate slot,
// scratch[32] is written only between a reduce's two barriers and read only
// after the second; the next write needs every thread past the next reduce's
// first barrier, i.e. after its read. No extra barrier needed.
#define BCAST 32
#ifndef CPU_EMU
__inline__ __device__ float blockReduceMax(float v, float* scratch) {
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    v = warpReduceMax(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : -INFINITY;
    if (warp == 0) v = warpReduceMax(v);
    if (threadIdx.x == 0) scratch[BCAST] = v;
    __syncthreads();
    return scratch[BCAST];
}
__inline__ __device__ float blockReduceSum(float v, float* scratch) {
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    v = warpReduceSum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) v = warpReduceSum(v);
    if (threadIdx.x == 0) scratch[BCAST] = v;
    __syncthreads();
    return scratch[BCAST];
}

#endif  // !CPU_EMU

// =============================================================================
// CHUNK 2: the fused kernel
//   grid  = N blocks (one per query row)
//   block = BC threads (one per key in the tile)
//   shared = Q row (D floats) + score tile (BC floats) + reduction scratch
//            + K chunk (KC x (BC+1) floats, transposed, padded against conflicts)
// -----------------------------------------------------------------------------
__global__ void attention_flash(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int N, int D, int DV, int causal) {
    extern __shared__ float smem[];
    float* sQ       = smem;              // D floats  -- the query row, reused
    float* sS       = sQ + D;            // BC floats -- this tile's scores
    float* scratch  = sS + BC;           // 33 floats -- 32 warp partials + broadcast
    float* sK       = scratch + 33;      // KC x (BC+1) floats -- K chunk, sK[c][r]

    const int i   = blockIdx.x;          // THIS BLOCK OWNS QUERY ROW i
    const int tid = threadIdx.x;
    if (i >= N) return;                  // whole block returns: no barrier split

    // Load the query row once. Every tile below reuses it from shared -- this is
    // the M4 "load once, reuse many" idea, now applied to the ONE operand that
    // is genuinely reused across the entire key loop.
    for (int k = tid; k < D; k += blockDim.x) sQ[k] = Q[(size_t)i * D + k];
    __syncthreads();

    const float scale = rsqrtf((float)D);

    // ---- the running state. THIS is what replaces the N-element score row ----
    float m_run = -INFINITY;             // running max
    float l_run = 0.0f;                  // running sum of exp
    float acc[(MAX_DV + BC - 1) / BC];   // running sum of exp*V, one per owned dim
    #pragma unroll
    for (int a = 0; a < (MAX_DV + BC - 1) / BC; a++) acc[a] = 0.0f;

    // Causal masking: query i may only attend to keys j <= i, so the key loop
    // can simply STOP at i+1 instead of computing and masking. Skipping work is
    // strictly better than masking it -- roughly half the FLOPs at large N.
    const int j_end = causal ? (i + 1) : N;

    for (int tile = 0; tile < j_end; tile += BC) {
        const int j = tile + tid;
        const int valid = (j < j_end);

        // -- scores for this tile -------------------------------------------
        // Stage K[tile .. tile+BC) x [kc .. kc+KC) into shared, transposed. On
        // the load, consecutive threads take consecutive columns of one row, so
        // a warp reads one contiguous 128-byte run of K. On the compute, thread
        // r reads sK[c][r], consecutive threads hit consecutive banks, and the
        // BC+1 row pitch keeps the transposed store conflict-free too. The
        // accumulation order over k is unchanged, so the result is bit-identical.
        const int rows = min(BC, j_end - tile);
        float dot = 0.0f;
        for (int kc = 0; kc < D; kc += KC) {
            const int cols = min(KC, D - kc);
            for (int e = tid; e < KC * BC; e += blockDim.x) {
                const int r = e / KC, c = e % KC;
                sK[c * (BC + 1) + r] = (r < rows && c < cols)
                    ? K[(size_t)(tile + r) * D + kc + c] : 0.0f;
            }
            __syncthreads();
            if (valid)
                for (int c = 0; c < cols; c++)
                    dot += sQ[kc + c] * sK[c * (BC + 1) + tid];
            __syncthreads();                 // sK is overwritten next chunk
        }
        float s = valid ? dot * scale : -INFINITY;
        sS[tid] = valid ? s : 0.0f;

        // -- the rebase ------------------------------------------------------
        // Every thread gets m_tile back, because every thread owns accumulators
        // that need the same correction factor.
        float m_tile = blockReduceMax(s, scratch);
        float m_new  = fmaxf(m_run, m_tile);
        float corr   = __expf(m_run - m_new);     // 0 on the first tile (m_run = -inf)

        float e = valid ? __expf(s - m_new) : 0.0f;
        sS[tid] = e;                               // publish for the PV step
        float l_tile = blockReduceSum(e, scratch); // (this call syncs twice)

        l_run = l_run * corr + l_tile;

        // -- acc_c = acc_c * corr + sum_j e_j * V[j][c] ----------------------
        // Thread `tid` owns output dims tid, tid+BC, ... so the accumulator
        // never leaves registers. The inner loop walks the tile's keys.
        const int tile_len = min(BC, j_end - tile);
        for (int c = tid, a = 0; c < DV; c += blockDim.x, a++) {
            float sum = 0.0f;
            for (int t = 0; t < tile_len; t++)
                sum += sS[t] * V[(size_t)(tile + t) * DV + c];
            acc[a] = acc[a] * corr + sum;
        }
        m_run = m_new;
        __syncthreads();      // sS is rewritten next iteration -- protect it
    }

    // ---- ONE division per output element, at the very end -------------------
    const float inv_l = 1.0f / l_run;
    for (int c = tid, a = 0; c < DV; c += blockDim.x, a++)
        O[(size_t)i * DV + c] = acc[a] * inv_l;
}

// =============================================================================
// CHUNK 3: host-side reference + .npy loader
// -----------------------------------------------------------------------------
#ifndef CPU_EMU
float* load_npy(const char* path, int rows, int cols) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open file: %s\n", path); return nullptr; }
    float* data = (float*) malloc((size_t)rows * cols * sizeof(float));
    if (!data) { fprintf(stderr, "malloc failed\n"); fclose(f); return nullptr; }
    fseek(f, 8, SEEK_SET);
    uint16_t hdr_len;
    if (fread(&hdr_len, sizeof(uint16_t), 1, f) != 1) {
        fprintf(stderr, "hdr read failed: %s\n", path); free(data); fclose(f); return nullptr;
    }
    fseek(f, hdr_len, SEEK_CUR);                 // hdr_len is BYTES, do not x2
    if ((size_t)rows * cols != fread(data, sizeof(float), (size_t)rows * cols, f)) {
        fprintf(stderr, "data read failed: %s\n", path); free(data); fclose(f); return nullptr;
    }
    fclose(f);
    return data;
}

#endif  // !CPU_EMU  (loader needs stdio only; kept with the CUDA host code)
#ifndef CPU_EMU_NO_REF
// Double-precision CPU reference. Deliberately NOT the online algorithm -- a
// reference that shares the kernel's structure would also share its bugs.
// Plain two-pass softmax in float64, the textbook definition.
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

#endif

#ifndef CPU_EMU
// =============================================================================
// CHUNK 4: one measured run
// -----------------------------------------------------------------------------
struct Result { double ms; double gflops; double gbps; double max_err; };

Result run_case(int N, int D, int DV, int causal, bool check, int iters,
                const float* hQ_in, const float* hK_in, const float* hV_in) {
    size_t szQ = (size_t)N*D, szV = (size_t)N*DV;
    std::vector<float> hQ, hK, hV;
    const float *Qp = hQ_in, *Kp = hK_in, *Vp = hV_in;

    if (!Qp) {   // generate reproducible inputs when no .npy is supplied
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

    size_t shmem = (D + BC + 33 + KC * (BC + 1)) * sizeof(float);

    // warm-up: the first launch pays JIT / context / cache-cold costs that have
    // nothing to do with the kernel. Timing it is the most common way to
    // publish a wrong number.
    attention_flash<<<N, BC, shmem>>>(dQ, dK, dV, dO, N, D, DV, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++)
        attention_flash<<<N, BC, shmem>>>(dQ, dK, dV, dO, N, D, DV, causal);
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
        for (size_t t = 0; t < szV; t++)
            max_err = fmax(max_err, fabs((double)hO[t] - ref[t]));
    }

    // FLOPs: QK^T is 2*N*N*D, PV is 2*N*N*DV. Causal halves both.
    double pairs  = causal ? ((double)N*(N+1)/2.0) : ((double)N*N);
    double flops  = 2.0*pairs*D + 2.0*pairs*DV;
    // Bytes: the fused kernel reads K and V once per QUERY BLOCK from L2's
    // perspective, but from DRAM's it reads Q,K,V once each if L2 holds a tile.
    // Report the compulsory (best-case) traffic so the number is honest.
    double bytes  = (double)(szQ + szQ + szV + szV) * sizeof(float);

    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return { ms, flops/(ms*1e6), bytes/(ms*1e6), max_err };
}

// =============================================================================
// CHUNK 5: main -- correctness at toy dims, then the benchmark sweep
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (sm_%d%d, %d SMs, %.1f GB)\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        printf("\n%6s %5s %7s %10s %10s %10s %12s\n",
               "N", "D", "causal", "ms", "GFLOP/s", "GB/s", "max_err");
        int Ns[] = {128, 512, 2048, 4096};
        int Ds[] = {32, 64, 128};
        for (int ci = 0; ci <= 1; ci++)
          for (int ni = 0; ni < 4; ni++)
            for (int di = 0; di < 3; di++) {
                int N = Ns[ni], D = Ds[di];
                // the float64 CPU reference is O(N^2 D); only affordable small
                bool check = (N <= 512);
                Result r = run_case(N, D, D, ci, check, 20, nullptr, nullptr, nullptr);
                printf("%6d %5d %7d %10.4f %10.1f %10.1f   ", N, D, ci, r.ms, r.gflops, r.gbps);
                if (check) printf("%12.3e\n", r.max_err); else printf("%12s\n", "(skipped)");
            }
        return 0;
    }

    int N = (argc > 1) ? atoi(argv[1]) : 4;
    int D = (argc > 2) ? atoi(argv[2]) : 4;
    int causal = (argc > 3) ? atoi(argv[3]) : 0;

    // At the project's toy dims, diff against the SAME golden file M3 and M4
    // used, so all three kernels are comparable to each other and to Python.
    if (N == 4 && D == 4 && !causal) {
        float* hQ = load_npy("data/Q.npy", 4, 4);
        float* hK = load_npy("data/K.npy", 4, 4);
        float* hV = load_npy("data/V.npy", 4, 4);
        float* hG = load_npy("data/O_golden.npy", 4, 4);
        if (hQ && hK && hV && hG) {
            Result r = run_case(4, 4, 4, 0, true, 100, hQ, hK, hV);
            printf("\ntoy dims vs data/O_golden.npy : max abs err = %.6e  (%.4f ms)\n",
                   r.max_err, r.ms);
            printf("expect ~1.19e-07 -- one float32 ULP near unity, same as M3/M4.\n");
            free(hQ); free(hK); free(hV); free(hG);
            return 0;
        }
        printf("(data/*.npy not found -- falling back to generated inputs)\n");
    }

    Result r = run_case(N, D, D, causal, N <= 512, 20, nullptr, nullptr, nullptr);
    printf("\nN=%d D=%d causal=%d : %.4f ms  %.1f GFLOP/s  %.1f GB/s",
           N, D, causal, r.ms, r.gflops, r.gbps);
    if (r.max_err >= 0) printf("  max_err=%.3e", r.max_err);
    printf("\n");
    return 0;
}
#endif  // !CPU_EMU -- the CPU harness supplies its own main()
