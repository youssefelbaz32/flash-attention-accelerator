// Milestone 3: CUDA naive single-head attention.
//
// Goal: parallelize the FLOAT attention math on the GPU. One thread per output
// row (query token). Load Q,K,V from the .npy bridge, run the kernel, copy O
// back, diff against O_golden.npy. Deliberately float32 (not fixed-point): we
// isolate "did I parallelize correctly?" from "did I quantize correctly?".
//
// Dev Mac is M1 Pro (no nvcc) -> write here, compile + run on the GPU PC.
//
// Write CHUNK BY CHUNK. Do not fill everything at once.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

// Dims must match the export (python/03_export_data.py): N = d = d_v = 4.
#define N   4
#define D   4
#define DV  4
#define CUDA_CHECK(call) do {                           \
      cudaError_t err = (call);                             \
      if (err != cudaSuccess) {                                      \
          fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                  cudaGetErrorString(err), __FILE__, __LINE__);   \
          exit(1);                                                   \
      }                                                          \
  } while (0)

// =============================================================================
// CHUNK 1: .npy loader — read a (rows x cols) float32 array into a host buffer.
//   .npy layout: \x93NUMPY | ver(2) | hdr_len(2, LE u16) | ascii dict | raw data
//   Strategy: open file, skip 8 bytes, read hdr_len, skip the header, fread the
//   floats. (We already know the shape, so we don't parse the dict.)
// -----------------------------------------------------------------------------
// TODO(you): float* load_npy(const char* path, int rows, int cols)

float* load_npy(const char* path, int rows, int cols) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        return nullptr;
    }

    float* data = (float*) malloc(rows * cols * sizeof(float));
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for data\n");
        fclose(f);
        return nullptr;
    }

    fseek(f, 8, SEEK_SET); // skip 8 bytes of magic and version from start
    uint16_t hdr_len;
    size_t cap1 = fread(&hdr_len, sizeof(uint16_t), 1, f); // read 2 bytes of header length
    //size_t fread(void *buffer, size_t size, size_t count, FILE *stream); count of size
    if (cap1 != 1) {
        fprintf(stderr, "Failed to read header length from file: %s\n", path);
        free(data);
        fclose(f);
        return nullptr;
    }
    fseek(f, hdr_len, SEEK_CUR);
    size_t captured = fread(data, sizeof(float), rows * cols, f);
    if ((size_t) (rows * cols) != captured) {
        fprintf(stderr, "Failed to read the expected number of floats from file: %s\n", path);
        free(data);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    return data;
}



// =============================================================================
// CHUNK 2: the kernel — one thread computes one output row of O.
//   __global__ void attention_kernel(const float* Q, const float* K,
//                                     const float* V, float* O)
//   thread i: i = blockIdx.x*blockDim.x + threadIdx.x;  guard i >= N
//   steps: scores s[j] = dot(Q[i],K[j])/sqrt(D)  ->  row max  ->  exp+sum
//          ->  O[i][c] = sum_j (exp(s[j]-max)/den) * V[j][c]
// -----------------------------------------------------------------------------
// TODO(you): the __global__ kernel

__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* O) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s[N];
    
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int k = 0; k < D; k++) {
            dot += Q[i*D + k] * K[j*D + k];
        }
        s[j] = dot/sqrtf((float) D);
    }
    float m = s[0];
    for (int j = 1; j < N; j++) {
        if (s[j] > m) m = s[j];
    }

    for (int j = 0; j < N; j++) {
        s[j] = expf(s[j] - m);
    }
    float denom = 0.0f;
    for (int j = 0; j < N; j++) {
        denom += s[j];
    }

    for (int j = 0; j < N; j++) s[j] /= denom; 

    for (int c = 0; c < DV; c++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += s[j] * V[j*DV + c];
        }
        O[i*DV + c] = sum;
    }


}




// =============================================================================
// CHUNK 3: host main() — load, cudaMalloc, H2D copy, launch, D2H copy, diff.
//   load Q,K,V,O_golden -> cudaMalloc dQ,dK,dV,dO -> cudaMemcpy H2D
//   -> launch <<<blocks,threads>>> -> cudaMemcpy O back -> max abs err vs golden
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


    attention_kernel<<<numBlocks, threadsPerBlock>>>(dQ, dK, dV, dO); 
    CUDA_CHECK(cudaGetLastError()); //gets launch config error
    CUDA_CHECK(cudaDeviceSynchronize()); //report run time errors AFTER device is done

    CUDA_CHECK(cudaMemcpy(hO, dO, N*DV*sizeof(float), cudaMemcpyDeviceToHost));

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
