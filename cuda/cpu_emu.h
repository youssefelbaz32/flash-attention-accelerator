// CPU emulation of the CUDA execution model, enough to run the M8 fused kernel
// on a machine with no GPU.
//
// WHY. The dev Mac has no nvcc, and "it compiled on the GPU box" is a poor place
// to discover an indexing bug. This harness runs the UNMODIFIED kernel body --
// same source file nvcc compiles -- under a real threaded barrier model, so the
// algorithm, the tiling arithmetic and the shared-memory indexing are all
// exercised here. It also makes the kernel testable in CI.
//
// WHAT IT DOES AND DOES NOT PROVE.
//   proves     : loop bounds, tile arithmetic, the online-softmax recurrence,
//                shared-memory layout, barrier placement (a missing
//                __syncthreads DEADLOCKS here, loudly, instead of producing
//                plausible garbage on hardware)
//   proves NOT : warp-level behavior, coalescing, occupancy, real races,
//                anything about performance
//
// The emulation boundary is exactly two functions -- blockReduceMax and
// blockReduceSum, which use warp shuffles on the GPU and a shared array here.
// Everything else is the real kernel.

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>

#define __global__
#define __device__
#define __host__
#define __restrict__
#ifndef __inline__
#define __inline__ inline
#endif

// CUDA's built-ins. threadIdx is per-thread; blockIdx/blockDim are per-launch,
// and blocks are executed one at a time, so plain globals are safe.
struct uint3_emu { unsigned x, y, z; };
extern thread_local uint3_emu threadIdx;
extern uint3_emu blockIdx, blockDim;
inline thread_local uint3_emu threadIdx_storage{0,0,0};

#define warpSize 32

// __shared__ becomes a per-block buffer handed to every thread of that block.
// The kernel declares `extern __shared__ float smem[]`, so provide that symbol.
#define __shared__
// The kernel declares `extern __shared__ float smem[]`, so this must be an
// ARRAY symbol, not a pointer -- the harness defines the storage.
extern float smem[];
#define EMU_SMEM_FLOATS (64 * 1024)

// CUDA provides these as builtins; the CPU does not.
__inline__ int   min(int a, int b) { return a < b ? a : b; }
__inline__ int   max(int a, int b) { return a > b ? a : b; }
__inline__ float __expf(float x) { return expf(x); }
__inline__ float rsqrtf(float x) { return 1.0f / sqrtf(x); }

// ---- a reusable N-thread barrier -------------------------------------------
// A missing __syncthreads() in the kernel hangs here rather than silently
// corrupting data -- which is exactly the failure mode you want during bring-up.
class Barrier {
  public:
    explicit Barrier(int n) : n_(n), count_(n), gen_(0) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        int g = gen_;
        if (--count_ == 0) { gen_++; count_ = n_; cv_.notify_all(); }
        else cv_.wait(lk, [&]{ return g != gen_; });
    }
  private:
    int n_, count_, gen_;
    std::mutex m_;
    std::condition_variable cv_;
};
extern Barrier* g_barrier;

#define __syncthreads() g_barrier->wait()

// ---- the two emulated reductions -------------------------------------------
// Same CONTRACT as the GPU versions: reduce across the block and BROADCAST the
// result to every thread. `scratch` is the same shared slice the kernel passes.
__inline__ float blockReduceMax(float v, float* scratch) {
    static std::vector<float> vals;
    if (threadIdx.x == 0) { vals.assign(blockDim.x, -INFINITY); }
    __syncthreads();
    vals[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (unsigned t = 0; t < blockDim.x; t++) m = fmaxf(m, vals[t]);
        scratch[0] = m;
    }
    __syncthreads();
    float r = scratch[0];
    __syncthreads();
    return r;
}

__inline__ float blockReduceSum(float v, float* scratch) {
    static std::vector<float> vals;
    if (threadIdx.x == 0) { vals.assign(blockDim.x, 0.0f); }
    __syncthreads();
    vals[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned t = 0; t < blockDim.x; t++) s += vals[t];
        scratch[0] = s;
    }
    __syncthreads();
    float r = scratch[0];
    __syncthreads();
    return r;
}
