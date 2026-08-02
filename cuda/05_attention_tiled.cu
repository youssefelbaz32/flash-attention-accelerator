// Milestone 4: CUDA TILED single-head attention (shared memory).
//
// Same float32 math as M3, same expected result (max abs err ~1.19e-7 vs golden).
// The ONLY thing that changes is WHERE the data lives while we compute:
//   M3 naive : every thread re-reads Q,K,V straight from GLOBAL memory (slow,
//              ~500 cyc). K and V get fetched N times more than necessary.
//   M4 tiled : the block loads Q,K,V from global into SHARED memory ONCE
//              (~20-30 cyc), syncs, then every thread reuses the shared copy.
//              "Load once, reuse many."
//
// At N=4 this is NOT faster (whole problem fits in registers) — the point is to
// write the shared-memory + __syncthreads pattern CORRECTLY so it scales.
//
// THE RULE (M4's new hazard): every thread in the block must reach every
// __syncthreads(). Never early-return before a barrier. Guard the WORK, not the
// sync:   load -> __syncthreads() -> if (i < N) { compute; write }.
//
// Dev Mac has no nvcc -> write here, compile + run on the GPU PC.
// Write CHUNK BY CHUNK. Do not fill everything at once.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

// Dims must match the export (python/03_export_data.py): N = d = d_v = 4.
#define N   4
#define D   4
#define DV  4

// --- carried over from M3 (already written + tested), reuse as-is -------------
#define CUDA_CHECK(call) do {                                      \
      cudaError_t err = (call);                                    \
      if (err != cudaSuccess) {                                    \
          fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                  cudaGetErrorString(err), __FILE__, __LINE__);    \
          exit(1);                                                 \
      }                                                            \
  } while (0)

float* load_npy(const char* path, int rows, int cols) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open file: %s\n", path); return nullptr; }
    float* data = (float*) malloc(rows * cols * sizeof(float));
    if (!data) { fprintf(stderr, "malloc failed\n"); fclose(f); return nullptr; }
    fseek(f, 8, SEEK_SET);
    uint16_t hdr_len;
    if (fread(&hdr_len, sizeof(uint16_t), 1, f) != 1) {
        fprintf(stderr, "hdr read failed: %s\n", path); free(data); fclose(f); return nullptr;
    }
    fseek(f, hdr_len, SEEK_CUR);
    if ((size_t)(rows * cols) != fread(data, sizeof(float), rows * cols, f)) {
        fprintf(stderr, "data read failed: %s\n", path); free(data); fclose(f); return nullptr;
    }
    fclose(f);
    return data;
}

// =============================================================================
// CHUNK A: the TILED kernel.
//   __global__ void attention_tiled(const float* Q, const float* K,
//                                    const float* V, float* O)
//
//   1. Declare __shared__ tiles for Q, K, V (block-visible scratchpad).
//   2. COOPERATIVE LOAD: threads split the work of copying global -> shared.
//   3. __syncthreads();   <-- ALL threads reach this (do NOT return before it).
//   4. if (i < N) { compute this row from the SHARED tiles; write O; }
//      (same math as M3: dot -> row-max -> exp/sum -> normalize -> PV)
// -----------------------------------------------------------------------------
// TODO(you): the __global__ tiled kernel
__global__ void attention_tiled(const float* Q, const float* K, const float* V, float* O) {
    __shared__ float sQ[N*D]; 
    __shared__ float sK[N*D];
    __shared__ float sV[N*DV];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int k = 0; k < D; k++) sQ[i*D + k] = Q[i*D + k];
        for (int k = 0; k < D; k++) sK[i*D + k] = K[i*D + k];
        for (int c = 0; c < DV; c++) sV[i*DV + c] = V[i*DV + c];
    }
    __syncthreads();

    if (i < N) {
        float s[N]; 

        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < D; k++) {
                dot += sQ[i*D + k] * sK [j*D + k];
            }
            s[j] = dot / sqrtf( (float) D);
        }

        float m = s[0];
        for (int j = 0; j < N; j++) if (s[j] > m) m = s[j];
        float denom = 0.0f;
        // exponentiating so we can softmax
        for (int j = 0; j < N; j++) {
            s[j] = expf(s[j] - m);
            denom += s[j];
        }


        //normalizing (softmax) -- now rows are probabilities

        for (int j = 0; j < N; j++) {
            s[j] /= denom;
        }

        for (int c = 0; c < DV; c++) {
            float val = 0.0f;
            for (int j = 0; j < N; j++) {
                val += s[j] * sV[j*DV + c];
            }
            O[i * DV + c] = val;
        }


    }


}

// =============================================================================
// CHUNK B: host main() — identical shape to M3.
//   load -> cudaMalloc -> H2D -> launch<<<1, threadsPerBlock>>> ->
//   cudaGetLastError + cudaDeviceSynchronize -> D2H -> max-abs-err -> free
// -----------------------------------------------------------------------------
// TODO(you): int main()

int main() {
    float *hQ = load_npy("data/Q.npy", N, D);
    float *hK = load_npy("data/K.npy", N, D);
    float *hV = load_npy("data/V.npy", N, DV);
    float *hO_golden = load_npy("data/O_golden.npy", N, DV);

    float *hO = (float *) malloc(N * DV * sizeof(float));

    if (!hQ || !hK || !hV || !hO_golden || !hO) {
        fprintf(stderr, "Error in loading npy files");
        return 1; 
    }


    float *dQ;
    CUDA_CHECK(cudaMalloc(&dQ, N * D * sizeof(float)));

    float *dK;
    CUDA_CHECK(cudaMalloc(&dK, N * D * sizeof(float)));
    
    float *dV;
    CUDA_CHECK(cudaMalloc(&dV, N * DV * sizeof(float)));

    float *dO;
    CUDA_CHECK(cudaMalloc(&dO, N * DV * sizeof(float)));


    CUDA_CHECK(cudaMemcpy(dQ, hQ, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, N * DV * sizeof(float), cudaMemcpyHostToDevice));

    
    int threadsPerBlock = 256; 
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock; // acts as a ceil function

    attention_tiled<<<numBlocks, threadsPerBlock>>>(dQ, dK, dV, dO);

    CUDA_CHECK(cudaGetLastError()); //gets launch config error
    CUDA_CHECK(cudaDeviceSynchronize()); //report run time errors AFTER device is done

    CUDA_CHECK(cudaMemcpy(hO, dO, N * DV * sizeof(float), cudaMemcpyDeviceToHost));


    float max_error = 0.0; 
    for(int i = 0; i < N * DV; i++) {
        float diff = fabsf(hO[i] - hO_golden[i]);
        if (diff > max_error) max_error = diff; 
    }

    printf("Max error is %e\n", max_error);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);

    free(hQ);
    free(hK);
    free(hV);
    free(hO_golden);
    free(hO);

    return 0;
}