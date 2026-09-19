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

---

## 2026-08-02 — Portfolio review: RAISED completion bar (for kernel / AI-chip roles)

External review of the portfolio pages. Verdict: strong flagship project for
GPU-kernel and AI-chip / HW-SW co-design roles **assuming completion means
MEASUREMENTS, not just working code**. TBD perf tables = "learning project";
reproducible profiling + synthesis + hardware bring-up = "serious new-grad
kernel/accelerator portfolio piece." Ratings (on completion w/ upgrades): CUDA
kernel 9/10, AI-accel RTL 9/10, HW-SW co-design 9.5/10, compiler/MLIR 7/10.

**Completion bar is now higher than the original 8-milestone roadmap. Added work:**

Kernel side (extends M3/M4/M8):
- Benchmark REAL dims (seq 128/512/2048/4096; head 32/64/128; fp32/fp16/bf16;
  causal + non-causal; batch/head sweeps). Keep N=4 only for verification.
- Compare vs PyTorch SDPA and a FlashAttention-style baseline. Don't claim to beat
  FA unless true; report % of peak bandwidth / within Q% of library.
- Nsight Compute table: DRAM throughput, L2 hit, shared-mem throughput, occupancy,
  regs/thread, warp exec efficiency, mem-load efficiency, kernel duration, FLOP/s.
- Add a fused online-softmax CUDA kernel + a **Triton** version (maybe CUTLASS/WMMA).
- Wrap as a **PyTorch custom op** and test vs `F.scaled_dot_product_attention`.

Hardware side (extends M5/M6/M7):
- FULL datapath end-to-end: Q/K/V load, QK^T, numerically-stable/online softmax
  (explicit exp decision: LUT / piecewise / base-2 / CORDIC), PV, output buffer,
  backpressure, reset/error behavior.
- **Parameterize** beyond N=D=4 (N, D, DATA_W, FRAC_W, LANES) — synthesizably
  configurable, not runtime-arbitrary.
- Real synthesis results: target device, LUT/FF/DSP/BRAM, WNS, fmax, latency
  (cycles + µs), throughput, per-module breakdown, results at 2-3 lane counts.
  Best artifact = **Pareto curve** (MAC lanes vs DSP/latency/fmax) = architecture
  judgment.
- On-board demo: host sends Q/K/V, gets O, auto-diffs vs the fixed-point model
  (UART ok; AXI-DMA/PCIe more representative).
- Strong verification: directed + constrained-random, saturation/negative/
  backpressure, handshake-stability assertions, scoreboard vs golden, functional
  coverage, a few **formal** valid/ready + FSM properties (ties to user's FV
  internship).

**Presentation:** top-of-page result summary with the headline numbers (done,
placeholders for now); Code / Benchmarks / Architecture / Demo links; one-command
repro script; tested GPU/FPGA/tool versions; CI running Python+CUDA correctness;
short demo video / waveform anim; "what I implemented personally" section +
attribution for any borrowed loader/softmax/reference.

**Technical claims CORRECTED on the site (were overreaching):**
- "attention is memory-bound" → config-dependent (shape/dtype/hw; decode vs prefill).
- 1.192093e-07 = "theoretical floor / nothing to chase" → ~one float32 ULP *near
  unity*; close agreement for THIS test case, not a universal error bound.
- "valid/ready IS AXI4-Stream" → follows AXI4-Stream handshake *semantics*; wrapping
  as AXIS later is straightforward (valid/ready alone ≠ full AXIS).
- "1/√d = right shift because dims are powers of two" → exact only when √d is a power
  of two, i.e. d a power of FOUR (4/16/64); d=8/32/128 need a fixed-point multiply.
- tiling "loads each input once" → true only for the single-block toy; at real sizes
  tiles re-load across thread blocks (shared mem can't hold full tensors).

**SECURITY:** physical address + phone were public in the site footer — REMOVED from
all pages (email + LinkedIn + GitHub kept).

**Estimate impact:** original ~40h-to-done assumed toy-dim completion. The
measurement/benchmark/synthesis/Triton/PyTorch-op/formal upgrades roughly DOUBLE
the remaining effort for a "portfolio-grade" finish (order ~70-100h), though the
core correctness path is unchanged. Track hours vs milestone to sharpen this.

**Next:** M5 RTL core continues (qkt in progress). Keep portfolio polish as separate
side-sessions off the critical path.

---

## 2026-09-19 — M5: SystemVerilog RTL, full naive datapath ✅

**What we built:** the complete attention engine in synthesizable SystemVerilog —
`dot4` → `qkt` → `row_max` → `softmax` → `pv` → `attention_top`, plus a new
bit-exact Python spec and a golden-vector flow tying the two together.

### The key decision: M2 could not be the golden model for RTL

M2 quantizes Q/K/V and both matmuls but still calls `np.exp` and does the
softmax divide in **float64**. Hardware can't. So M2 is a *precision study*, not
an executable spec — RTL matching it bit-for-bit is impossible by construction.

Wrote `python/04_rtl_fixed_model.py`: every operation is an integer operation the
RTL performs, in the same order, with the same truncation and saturation. It
emits `rtl/vectors/*.hex`, and the testbenches `$readmemh` them. Comparison is
**exact, zero tolerance** — a tolerance would hide exactly the bugs this is for.

It also emits `exp_lut.hex`, which the RTL loads. **One definition of the exp
table, two consumers** — software and hardware cannot drift apart.

### Measured decision: where to spend rounding hardware

Floor division biases all N softmax terms the same direction (visible as `P` row
sums of `[255,254,254,253]` — never above 256). Tested round-to-nearest over a
**200-seed sweep**, because seed 0 alone gave the *opposite* answer:

| divide | pv shift | max err | mean err | p99 |
|---|---|---|---|---|
| trunc | trunc | 0.02646 | 0.005144 | 0.01837 |
| round | round | **0.02432** | **0.004071** | **0.01593** |

21% mean-error cut for **two adders**. But adding the same rounding to `qkt`
bought 0.3% (0.004071 → 0.004057) — because `S` feeds an exp LUT indexed by
`(-x) >> 3`, so a ±1 LSB wobble in `S` is shifted out of the index 7 times in 8.
`dot4` is also the module replicated `LANES` times. **Conclusion: round in pv and
softmax, truncate in dot4.** Exposed as `ROUND_EN` on dot4.

Cost of moving softmax into hardware: max abs err vs the M1 float golden went
**0.0063 (M2, float softmax) → 0.0133 (M5, LUT + integer divide)**.

### Design notes

- **`dot4` serves both matmuls.** `qkt` instantiates it `SCALE_EN=1, ROUND_EN=0`;
  `pv` instantiates it `D=N, SCALE_EN=0, ROUND_EN=1` (where the net shift
  degenerates to exactly `>>> FRAC`). "Attention is two matmuls with a softmax
  between them" is cheap to say; one MAC module serving both makes it true.
- **V's transpose is free.** `pv` gathers V's columns with a generate loop —
  compile-time indices, zero logic. The cheapest thing in the RTL and the most
  expensive thing in CUDA (uncoalesced access).
- **No top-level FSM.** Every module speaks valid/ready, so the chain is just
  producer→consumer wiring and backpressure propagates backwards for free.
- **Two skip-ahead registers** at the top (`s_reg`, `v_reg`) — values needed
  later than the stage that produced them.
- `row_max` exists because softmax(S) == softmax(S−c); choosing c = row max makes
  every exp argument ≤ 0, so the LUT only stores the negative half-line (half the
  ROM), and exp can never overflow.

### Bugs hit

1. **Divider off-by-one → results exactly 2×.** The FSM left `DIV_ITER` on
   `div_cnt == 0`, but `curr_state` is still `DIV_ITER` that cycle, so the
   datapath body ran a 26th time and shifted the quotient left once too many.
   Fix: load `NUMW-1`, not `NUMW`. **A clean power-of-two error is always a
   shift-count bug** — that fingerprint found it in one pass.
2. `logic d4_all_in_ready = &d4_in_ready;` — a *variable initializer*, evaluated
   once at time 0, not a continuous assign. Would have frozen forever. Needs
   `assign`.
3. `j_base` declared `[$clog2(D)-1:0]` — `j` indexes **keys (N)**, not features.
   Masked by N==D==4. Same family as the M4 `j < D` bug.
4. `last_j` must compare against the constant `N-LANES`, not test
   `j_base+LANES == N` — `j_base` is 2 bits at N=4, so that sum wraps to 0 at
   exactly the moment it needs to read 4, and the engine hangs in `WAIT`.
5. `logic signed [D*DW-1:0] q_mem [N]` — `signed` on a *container* of four Q8.8
   words is meaningless; it declares one wide signed number.
6. Enum ternary `next_state = last_pair ? DONE : ISSUE` needs an explicit cast in
   strict tools; written as if/else.
7. iverilog 13: `automatic` inside a for-loop body is unsupported, and an unsized
   array argument to a task with differing bus widths **segfaults the compiler**.

### RESULT ✅ M5 DONE — every stage bit-exact vs the Python model

```
attention_top  (N=4 D=4 DV=4 LANES=1, Q8.8)
  S  (qkt)       16/16 exact  PASS
  m  (row_max)    4/ 4 exact  PASS
  P  (softmax)   16/16 exact  PASS
  O  (pv)        16/16 exact  PASS
```
Tested under a hostile-but-legal producer (inputs scrambled the cycle after the
accept beat) and 5 cycles of output backpressure.

### THE headline measurement: Amdahl's law, measured

| LANES | mults | qkt | row_max | softmax | pv | **total** | speedup | softmax share |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 97 | 17 | 449 | 102 | **659** | 1.00× | 68% |
| 2 | 2 | 49 | 17 | 449 | 54 | **563** | 1.17× | 79% |
| 4 | 4 | 25 | 17 | 449 | 30 | **515** | 1.28× | 87% |

**4× the multipliers bought 1.28×.** The matmuls did get 3.4× faster — but the
sequential restoring divider is 449 cycles no matter what, and `LANES` cannot
touch it. Adding lanes only makes the divider a *larger* share of the problem.

This is the measurement that sets the M8 agenda: the bottleneck in a naive
attention accelerator is not the matmul everyone optimizes, it's the softmax
normalization. One reciprocal per row (1 divide + N multiplies instead of N
divides) should cut ~4× off the dominant stage — worth far more than more MACs.

**Next:** M8 — online/streaming softmax (FlashAttention-lite): kill the O(N²)
intermediates and the per-element divide.

---

## 2026-09-19 — M8: FlashAttention-lite, RTL half (online softmax) ✅

**What we built:** `python/05_online_softmax_model.py` (the spec) and
`rtl/flash_top.sv` (the hardware), plus `rtl/exp_rom.sv` factored out so M5's
softmax and M8's datapath share ONE definition of exp().

**The trick.** Carry a running max and retroactively fix the accumulator. Per
block of BLK keys:
```
m_new = max(m_run, max_j s_j)
corr  = exp(m_run - m_new)                 <= 1.0, the rebase factor
l_run = l_run * corr + sum_j exp(s_j - m_new)
acc_c = acc_c * corr + sum_j exp(s_j - m_new) * V[j][c]
```
and only at the end, `O[i][c] = acc_c / l_run`.

`corr` is the entire idea: when a later block holds a bigger score, everything
already accumulated is wrong by **exactly** `exp(m_old - m_new)` — a *constant* —
so one multiply rebases the whole history. That is why softmax, which looks
irreducibly global, is streamable.

### Honest accounting — I got this wrong once and the sweep caught it

Seed 0 showed online softmax as **2× more accurate** than naive (0.0065 vs
0.0133) and I nearly wrote that down. Over 200 seeds it is not true:

| BLK | online mean err | naive mean err |
|---|---|---|
| 1 | 0.004082 | 0.004071 |
| 2 | 0.003924 | 0.004071 |
| 4 | 0.003686 | 0.004071 |

A wash at BLK=1, ~9% better at BLK=4. **Flash is a memory/divide win, not an
accuracy win.** And the *direction* is the real finding: smaller blocks are
worse, because `corr` is applied once per block and its rounding **compounds
N/BLK times down the row** — the only place in the design where an error feeds
back into itself. Streaming harder = streaming less accurately.

(Second time a single toy seed pointed the opposite way from a 200-seed sweep —
first was the rounding-mode study in M5. Rule going forward: **no fixed-point
decision from one seed, ever.**)

### RESULT ✅ M8 DONE — bit-exact vs its own spec at every BLK

| BLK | mults | cycles | vs M5 naive |
|---|---|---|---|
| 1 | 1 | **360** | max delta 3 LSB |
| 2 | 2 | **280** | max delta 2 LSB |
| 4 | 4 | **240** | max delta 2 LSB |

The delta vs M5 is *supposed* to be small and nonzero — two different algorithms,
not two implementations of one. Zero would mean the streaming path isn't
streaming.

### THE headline: the algorithm beat the hardware

| | mults | cycles | intermediate storage |
|---|---|---|---|
| M5 naive, LANES=1 | 1 | 659 | 2N² words (S and P) |
| M5 naive, LANES=4 | 4 | 515 | 2N² words |
| **M8 flash, BLK=1** | **1** | **360** | **DV+2 words** |
| M8 flash, BLK=4 | 4 | 240 | DV+2 words |

**M8 with ONE multiplier (360 cy) beats M5 with FOUR (515 cy).** 4× the area
bought 1.28×; changing the algorithm bought 1.83× at identical area.

And the storage term is the one that actually decides whether this fits on a
part. The N² intermediates vanish entirely: at N=128, D=DV=64, M5 needs
2·128² = 32768 words ≈ **64 KB of BRAM** for S and P; M8 needs DV+2 = **66
words ≈ 132 B**, *independent of N*. That is the difference between synthesizing
and not.

### Implementation notes

- **`corr` and `e` are unsigned, `acc` is signed.** SystemVerilog makes an
  ENTIRE expression unsigned if ANY operand is — so both are widened into
  explicit signed values (`$signed({1'b0, x})`) before multiplying the
  accumulator. Skipping this turns a negative `acc` into a huge positive one.
  Worth a formal assertion later.
- **One exp ROM, two users**, muxed by state: the rebase factor in `SCORE_WAIT`,
  the score exponentials in `EXPF`. They never need it on the same cycle, so
  streaming costs no extra ROM over M5.
- **One multiplier reused DV times** for the rebase rather than DV multipliers
  sitting idle — the rebase is DV cycles, not 1, and that's the right trade at
  these dimensions.
- The `div_cnt <= RNUMW-1` off-by-one from M5's softmax was already understood,
  so the reciprocal divider worked first try. Writing the bug down paid for
  itself within one milestone.

**Next:** the GPU half of M8 — fused online-softmax CUDA kernel + Triton
version, running the same recurrence on the GPU to close the loop with M3/M4.
M6 (host comms) and M7 (FPGA bring-up) both need board access.

---

## 2026-09-19 — M8 GPU half: fused CUDA kernel + Triton + CPU verification ✅ (code)

**What we built:**
- `cuda/06_attention_flash.cu` — fused online-softmax kernel. One thread block
  per query row; the running `(m, l, acc)` lives in registers; K and V stream
  from global in tiles of `BC=128`. Causal masking *stops the key loop early*
  rather than computing-and-masking (~2× fewer FLOPs at large N). Benchmark
  harness sweeps N∈{128,512,2048,4096} × D∈{32,64,128} × causal, reports
  ms/GFLOP·s⁻¹/GB·s⁻¹ and diffs against a float64 CPU reference where affordable.
- `python/06_triton_attention.py` — the same recurrence in Triton, wrapped as a
  drop-in for `F.scaled_dot_product_attention` (same `(Z,H,M,D)` layout, so a
  benchmark can't accidentally time a transpose), plus a head-to-head bench.
- `cuda/cpu_emu.h` + `cuda/test_flash_cpu.cpp` — **runs the unmodified kernel
  body on the CPU.** One `std::thread` per CUDA thread, blocks sequential,
  `__syncthreads()` backed by a real barrier.
- `run_all.sh` + `.github/workflows/ci.yml` — one-command repro, gated in CI.

**The emulation boundary is exactly two functions** (`blockReduceMax`,
`blockReduceSum`), which use warp shuffles on the GPU and a shared array on the
CPU. Everything else — the algorithm, loop bounds, tile arithmetic, shared
layout — is the identical source nvcc compiles. A harness that rewrote the
kernel would prove nothing about the kernel.

*Proves:* loop bounds, tile arithmetic, the online recurrence, barrier placement
(a missing `__syncthreads()` **deadlocks loudly** here instead of producing
plausible garbage on hardware).
*Does not prove:* warp behavior, coalescing, occupancy, real races, performance.

```
N=4     D=4    causal=0  max_err=1.484e-07  PASS
N=4     D=4    causal=1  max_err=1.421e-07  PASS
N=64    D=32   causal=1  max_err=4.052e-07  PASS
N=128   D=64   causal=0  max_err=3.958e-07  PASS
N=300   D=64   causal=0  max_err=2.766e-07  PASS   <- ragged tail (300 % 128 != 0)
N=300   D=64   causal=1  max_err=6.577e-07  PASS   <- ragged tail + causal together
N=512   D=128  causal=0  max_err=3.102e-07  PASS
```

`N=300` is deliberately not a multiple of the tile width. The masked tail tile
is where boundary bugs live, and a clean power-of-two sweep never catches them.

**Kernel design notes:**
- The Q row loads to shared ONCE and is reused by every key tile — M4's "load
  once, reuse many", now applied to the one operand genuinely reused across the
  whole key loop.
- Both block reductions **broadcast** the result to every thread, because every
  thread owns accumulators needing the same `corr`. A reduce-to-thread-0 form
  would need a third barrier to publish it.
- `if (i >= N) return;` is at BLOCK granularity (`i = blockIdx.x`), so the whole
  block leaves together and no barrier is ever split — the M4 rule, respected by
  construction rather than by care.
- Warm-up launch before timing. Timing the first launch (JIT + context + cold
  cache) is the single most common way to publish a wrong speedup.
- Reported bandwidth is *compulsory* traffic (Q,K,V,O once each), not an
  inflated best case.

**Claim policy for the Triton benchmark:** SDPA dispatches to real
FlashAttention-2/cuDNN. Beating it is not the claim. The number to report is
**% of SDPA**; this kernel uses no tensor cores (no WMMA/MMA), so parity is not
expected.

**Milestone numbering corrected.** The online-softmax work was briefly logged as
"M6". Per the roadmap, M6 is host↔FPGA comms and M7 is FPGA bring-up;
**FlashAttention-lite is M8**, and the RTL and CUDA/Triton versions are two
implementations of that one milestone. Renumbered across all files.

**Status:** M1–M5 and M8 complete (M8 GPU timings pending hardware). M6 and M7
both need board access.

**Next (needs hardware):**
- GPU box: `nvcc -O3 -arch=sm_80 -o build/flash cuda/06_attention_flash.cu`,
  then `./build/flash` (toy, expect ~1.19e-07), `./build/flash bench`, and
  `ncu --set full` for the Nsight table. `pip install triton` for the SDPA
  head-to-head.
- Vivado: synthesize `attention_top` and `flash_top` at LANES/BLK ∈ {1,2,4} for
  the LUT/FF/DSP/BRAM + fmax/WNS Pareto curve — the artifact that turns the
  cycle counts above into an area/latency argument.
