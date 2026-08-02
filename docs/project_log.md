# Project Log — Attention Accelerator (Python → CUDA → RTL)

Running log of what we built, what passed/failed, bugs, and lessons.

---

## 2026-06-13 — M1: Python float golden model ✅

**What we built:** `python/01_golden_model.py` — float64 single-head scaled
dot-product attention for N=d=d_v=4, built chunk by chunk:
1. imports + `np.random.seed(0)`
2. dimensions N, d, d_v
3. Q, K, V via `np.random.randn`
4. `S = Q @ K.T / np.sqrt(d)`
5. stable softmax (row-max subtract → exp → normalize) → P
6. `O = P @ V` + `assert np.allclose(P.sum(axis=1), 1.0)` + prints

**What passed:** assert passed; row sums = 1; softmax preserves rank; O leans
toward most-weighted V row. PASS.

**Bugs:** none this milestone.

**Assumptions:** random toy inputs (no trained weights); diagonal of S carries
no special structure in a random model.

**What I understand now:**
- matmul contracts the shared inner dim; `.T` placement decides which axis dies
- `Q@K.T` (token×token) vs `Q.T@K` (feature×feature)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
- `1/sqrt(d)` is numerical conditioning for softmax
- row-max subtraction = the reason `row_max_engine.sv` will exist
- `keepdims`/`axis=1` broadcasting traps in softmax

**Next step:** M2 — Python fixed-point model; compare to golden; error stats.

**Portfolio note:** "Started from a self-checking float reference so every lower
level has an unambiguous oracle — mismatches downstream are real bugs, never
input drift (fixed seed)."

---

## 2026-06-28 — M2: Python fixed-point model ✅

**What we built:** `python/02_fixed_point_model.py` — single-head attention on a
Q8.8 integer grid, diffed against the M1 golden model. Same seed → same Q/K/V.
Chunk by chunk:
1. format + helpers: `FRAC=8`, `SCALE=256`, `quantize`/`dequantize`
2. saturation: 16-bit range, `saturate()` clamps before storing
3. fixed-point forward path: `Qq,Kq → S_raw (int matmul) → S_fixed → softmax →
   P_fixed`, then `Vq,Pq → O_raw (int matmul) → O`
4. golden float `O_golden` recomputed locally (chose this over importing 01 —
   the `01_` filename isn't a valid module name) + error metrics

**What passed:** runs standalone; golden P rows sum to 1 after the fix.
- **Max abs error 0.00631**, **mean abs error 0.00205** vs golden. This is the
  M2 error budget.

**Bugs:**
- *Missing softmax normalization in golden* — wrote `P = exp(S-max)` with no
  `/sum`; golden P rows summed to 1.7–2.9 and `O_golden` was ~2× too big.
  Caught by inspecting `P.sum(axis=1)`. Fixed with `P /= np.sum(..., keepdims=True)`.
- *Earlier draft used raw float `V` (not `Vq`) in PV* — PV wasn't on the grid;
  fixed by quantizing both P and V before the integer matmul.

**FPGA issue surfaced (on purpose):** quantized softmax rows no longer sum to 1 —
`Pq` row sums = [256, 257, 255, 256] instead of all 256. ±1 rounding errors don't
cancel; compounds at large N. Renormalization / more frac bits is the real fix.

**What I understand now:**
- QA.B × QA.B → Q2A.2B; divide the accumulator by `SCALE^2` after each fixed×fixed matmul
- saturate *before* storing or values wrap silently
- `P @ V` needs no `.T` (keys axis already aligned); `Q @ K.T` does (contract d)
- small probabilities are the first casualties of quantization
- softmax `exp` is still float here — only QK^T and PV are truly on the grid

**Next step:** M3 — CUDA naive kernel; diff vs M1 golden, sanity-check against
this M2 error budget.

**Portfolio note:** "Built a bit-accurate fixed-point twin of the float model so
I could quote a concrete error budget (0.6% worst-case at Q8.8) before writing a
line of HDL — and caught the classic 'quantized softmax stops summing to 1' issue
in Python where it's cheap to see."

---

## 2026-07-06 — M3: CUDA naive kernel ✅ (completed 2026-07-18)

**Plan:** parallelize the *float* attention math on the GPU. Deliberately back to
float (not fixed-point) so we isolate "did I parallelize correctly?" from "did I
quantize correctly?" — never debug two new axes at once. Dev Mac is M1 Pro (no
nvcc), so: write `.cu` here, transfer to the GPU PC to compile + run + diff.

**Data bridge — `python/03_export_data.py`:** export Q, K, V, O_golden to `.npy`
so CUDA host code loads byte-identical inputs (C++ `rand` can't reproduce numpy).

**Design decision — cast to float32 *before* saving** (not export f64 + downcast
in CUDA):
- *Single source of truth:* both sides read identical float32 bytes, so any diff
  is the kernel's fault, never a silent f64→f32 truncation. Same discipline as the
  fixed seed — kill every variable except the one under test.
- *Matches target hardware:* GPUs are float32-native (f64 often ~1/32 throughput
  on consumer NVIDIA); FPGA later is narrower still. Don't carry precision the
  datapath will never have.
- *Half the bytes:* f8→f4 halves file size / bandwidth; right habit for the
  bandwidth-bound regime at large N.
- *Known consequence (the M3 error floor):* golden is computed in f64 then cast
  to f32, so expect residual CUDA-vs-golden diffs ~1e-6–1e-7 from float32 rounding
  even with a perfect kernel. Don't chase it to zero.

**`.npy` format learned:** magic `\x93NUMPY` (6B) + version (2B) + header-len (2B,
LE uint16) + ASCII dict header (`{'descr':'<f4','fortran_order':False,'shape':...}`,
space-padded to 64B alignment) + raw row-major little-endian data. CUDA side skips
the header, freads the floats.

**Kernel written (`cuda/04_attention_naive.cu`):** one thread per output row.
- CHUNK 1 `load_npy`: `fseek(8)` → read 2B hdr_len → `fseek(hdr_len, SEEK_CUR)`
  (hdr_len is already a byte count — do NOT multiply by 2) → `fread` the floats,
  **checked** against `rows*cols`.
- CHUNK 2 kernel: dot → row-max → exp/sum → normalize → PV, flat indexing
  `Q[i*D+k]`, `if (i >= N) return;` guard.
- CHUNK 3 host: `load → cudaMalloc → H2D → launch → cudaGetLastError +
  cudaDeviceSynchronize → D2H → max-abs-err diff → free`.

**Bugs hit (all mine, caught before GPU):**
1. `fseek(2 * hdr_len, ...)` — wrong units, seeked past EOF. **fseek past EOF is
   silently legal** (returns 0); only the later `fread` fails, and only via its
   return count (ferror stays 0). This is why checking `fread`'s return is
   load-bearing, not defensive boilerplate — same silent-failure family as
   `cudaMemcpy`/kernel-launch errors.
2. Tried to derive read-count from the destination buffer (`sizeof(float)*data`) —
   doesn't compile; the only witness to how much was read is `fread`'s return.
3. `malloc` without `(float*)` cast — nvcc compiles as C++, no implicit
   `void*→float*`.
4. `CUDA_CHECK` macro missing line-continuation `\` — multi-line `#define` is one
   logical line; verify with `clang -E`, don't eyeball whitespace.
5. Summed cudaError codes instead of per-call `CUDA_CHECK` — loses
   `cudaGetErrorString` + which call failed.
6. `fprintf(fmt, ...)` instead of `printf` — fprintf's 1st arg is a `FILE*`.

**Design notes learned:**
- `do { } while(0)` macro wrapper → macro is a single statement, survives inside
  braceless `if/else` (a plain `{}` block + trailing `;` breaks the `else`).
- `cudaMalloc(&dQ, ...)`: `&dQ` is `float**` — cudaMalloc writes the *device*
  address back into a *host* variable. `dQ` is a host-side label for a GPU address;
  the `float*` type carries element-size for pointer math + kernel type-checking,
  even though the host may never dereference it.
- Launch `<<<numBlocks, threadsPerBlock>>>`: total threads = product; ceil-div
  `(N+tpb-1)/tpb` blocks; guard drops the over-provisioned threads. Launch returns
  `void` + is async → check via `cudaGetLastError()` (config) then
  `cudaDeviceSynchronize()` (runtime + barrier before D2H).

**RESULT (2026-07-18):** ✅ **M3 DONE.** Validated on CPU by stubbing the CUDA
runtime and emulating the launch (loop threadIdx over all 256 threads, run the
real kernel body over `data/`). **Max abs err = 1.192093e-07** — exactly the
float32 machine epsilon (`2^-23`), the theoretical floor vs an f64→f32 golden.
Proves loader + host plumbing + kernel math correct; does NOT prove
CUDA-specific behavior (real parallelism/device mem/races). GPU build (to run on
the GPU PC): `nvcc -O2 -o attention_naive cuda/04_attention_naive.cu && ./attention_naive`
from project root — expect ~1.19e-7 to match CPU bit-for-bit.

**Next:** M4 — CUDA tiled kernel (shared memory). Will run M3 + M4 together on the
GPU PC.

---

## 2026-08-01 — M4: CUDA tiled kernel (shared memory) ✅

**What we built:** `cuda/05_attention_tiled.cu`. Same float32 math as M3, same
result — the ONLY change is WHERE inputs live during compute: staged Q,K,V from
global into `__shared__` tiles once, then reused. "Load once, reuse many."

**Why it matters (exact global-access count, no caching, N=D=Dv=4):**
- Naive per thread: Q reads N·D, K reads N·D, V reads N·Dv. ×N threads →
  Q,K,V each N²·D = 64 → **192 reads** + 16 writes = 208 accesses.
- Distinct input elements = 3·16 = 48. So naive does 192/48 = **4× = N** redundant
  reads (K,V re-fetched once per query row). Tiling drives the N² terms toward N:
  load 48 to shared once, reuse on-chip. At N=4 invisible on the clock; at large N
  it's the whole memory-bound problem.

**GPU memory hierarchy learned:** registers (~1cyc, per-thread) < shared (~20-30cyc,
per-BLOCK) < L2 (~200) < global (~400-600, per-GPU). `__shared__` = on-chip
scratchpad, ONE copy per block, all threads see it, lives only while the block runs,
size must be compile-time known. Shared is a manual reuse cache — you decide what to
stage instead of praying to L2.

**THE new hazard — `__syncthreads()` (block-wide barrier):** every thread in the
block must reach EVERY barrier. A barrier inside a divergent path (e.g. after
`if (i>=N) return;`) → threads that returned never arrive → **deadlock (actually
UB per spec — may hang, may return garbage, may look fine then break)**. Rule:
**guard the WORK, not the barrier.** Kernel shape: cooperative load `if(i<N){...}`
→ `__syncthreads()` (unguarded, ALL threads) → `if(i<N){ compute; write O; }`.

**Kernel design decisions:**
- Cooperative load: thread `i` loads ROW `i` of Q,K,V (one row per thread; N
  threads cover N rows). Row (not column) load is also coalesced (contiguous).
- O is NOT staged in shared — written once per element by its owner thread, zero
  reuse → straight to global. (Shared is for reused *inputs*; write-once outputs go
  direct. NB: real FlashAttention DOES keep O on-chip, but for *accumulation* reuse
  across tiles — different reason.)
- One thread per output row: builds row's softmax `s[]` once in registers, then
  emits the whole row (`for c` over Dv) reusing `s[]`.

**Bugs hit (all caught before GPU):**
1. `__global__ tiled_kernel(...)` missing `void` — `__global__` MUST return void.
2. Scores loop `j < D` should be `j < N` — latent, masked by N==D; wrong intent.
3. PV `sV[c*DV + j]` transposed — should be `sV[j*DV + c]`; real answer-corrupter.
4. `cudaMalloc(&dQ, ...)` copy-paste where `&dV` meant — leaks dQ, leaves dV wild.
5. Launch `<<numBlocks, tpb>>` (bit-shift!) — must be triple `<<<...>>>`.
6. Error-check order inverted: D2H copied BEFORE getLastError/sync — must be
   launch → cudaGetLastError → cudaDeviceSynchronize → D2H.
7. Duplicate unreachable `return 0;`.

**RESULT:** ✅ **M4 DONE.** Validated on CPU with a two-pass harness over the
UNMODIFIED kernel (`__shared__`→`static`, `__syncthreads()`→no-op; sweep 1 fills
tiles, sweep 2 computes — mimics the barrier). **Max abs err = 1.192093e-07**,
bit-for-bit identical to M3 (same math, different data path). Proves shared-tile
indexing + host plumbing correct; does NOT prove real-concurrency behavior
(true `__syncthreads`, races) — CPU runs threads sequentially. GPU build:
`nvcc -O2 -o attention_tiled cuda/05_attention_tiled.cu && ./attention_tiled` from
project root — expect ~1.19e-7; if it differs from naive, it's a concurrency bug
(diff against M3 as known-good).

**Next:** M5 — SystemVerilog RTL (per build order: RTL qkt → row_max → softmax →
pv → full naive). User will run M3 + M4 together on the GPU PC.
