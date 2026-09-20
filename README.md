# Attention Accelerator

Single-head scaled dot-product attention, built five times over: NumPy, Q8.8
fixed-point, CUDA, SystemVerilog, and Triton. Every level is checked against the
level above it, and most of them match bit for bit rather than within a
tolerance.

```
S = Q·Kᵀ / √d      →      P = softmax(S)      →      O = P·V
```

```bash
./run_all.sh
```

That reproduces every number below. No GPU or FPGA needed.

[![correctness](https://github.com/youssefelbaz32/flash-attention-accelerator/actions/workflows/ci.yml/badge.svg)](../../actions)

## The result I did not expect

I parameterized the MAC count in the RTL and swept it, expecting the usual
story. One multiplier, then two, then four. The matmuls got 3.4x faster. The
whole pipeline got 1.28x faster.

The per-stage profile explains it:

| LANES | qkt | row_max | softmax | pv | total | softmax share |
|---|---|---|---|---|---|---|
| 1 | 97 | 17 | 449 | 102 | 659 | 68% |
| 2 | 49 | 17 | 449 | 54 | 563 | 79% |
| 4 | 25 | 17 | 449 | 30 | 515 | 87% |

The sequential divider inside softmax costs 449 cycles no matter what I do to
the multipliers. Adding MACs just makes the divider a bigger fraction of the
problem. Amdahl's law, with a waveform attached.

So I changed the algorithm instead. Online softmax carries a running max and
rebases whatever it has already accumulated by `exp(m_old - m_new)` whenever a
bigger max shows up. That factor is a constant, so a single multiply corrects
the entire history, which is why softmax can stream at all.

| design | multipliers | cycles | intermediate storage |
|---|---|---|---|
| naive, LANES=1 | 1 | 659 | 2N² words (S and P) |
| naive, LANES=4 | 4 | 515 | 2N² words |
| online, BLK=1 | 1 | 360 | DV+2 words |
| online, BLK=4 | 4 | 240 | DV+2 words |

One multiplier running online softmax beats four running the naive version, at a
quarter of the arithmetic area. The storage column matters more on an FPGA
though: at N=128 and D=64 the naive path wants about 64 KB of BRAM for S and P,
while the online path needs 66 words, and that number does not grow with N.

## Milestones

| # | Level | Status | Checked against |
|---|---|---|---|
| M1 | Python float64 golden | done | self-check, softmax rows sum to 1 |
| M2 | Python Q8.8 fixed-point | done | M1, error budget |
| M3 | CUDA naive | done | M1, max abs err 1.19e-07 |
| M4 | CUDA tiled (shared memory) | done | M1 and M3, bit-identical to M3 |
| M5 | SystemVerilog, naive datapath | done | integer spec, exact at all 4 stages |
| M6 | Host to FPGA comms (UART, packet FSM, AXIS) | needs board | |
| M7 | FPGA bring-up on Vivado | needs board | |
| M8 | FlashAttention-lite, RTL | done | its own integer spec, exact at every BLK |
| M8 | FlashAttention-lite, CUDA and Triton | code done, timings pending GPU | float64 CPU reference |

M8 is one milestone with two implementations. M6 and M7 are the bring-up path
and both need hardware I do not have in front of me yet.

## Why there are two Python models

M2 quantizes Q, K, V and both matmuls, but it still calls `np.exp` and does the
softmax division in float64. Hardware cannot do that, so no RTL can ever match
it. M2 is a precision study, not a spec.

`python/04_rtl_fixed_model.py` is the spec. Every operation in it is an integer
operation the RTL performs, in the same order, with the same truncation and the
same saturation. It writes `rtl/vectors/*.hex` and the testbenches read those
back with `$readmemh` and compare with zero tolerance. A tolerance would have
hidden the divider bug described below.

It also emits `exp_lut.hex`, which the RTL loads directly. One definition of
`exp()`, two consumers, so software and hardware cannot drift apart.

## Bugs worth remembering

These cost me the most time, and all of them are the kind that pass a casual
test.

**The divider was exactly 2x too large.** The softmax FSM left `DIV_ITER` when
`div_cnt == 0`, but `curr_state` is still `DIV_ITER` on that cycle, so the
datapath ran one extra shift-subtract and shifted the quotient left once too
many. Loading `NUMW - 1` instead of `NUMW` fixed it. The useful part is the
fingerprint: a clean power-of-two error is almost always a shift-count bug, and
that narrowed 400 lines of RTL down to one register load.

**`logic d4_all_in_ready = &d4_in_ready;` does nothing.** At module scope in
SystemVerilog that is a variable initializer, not a continuous assignment. It
samples once at time zero and never updates. It needs to be an `assign`.

**A counter that wraps exactly when you need it not to.** I first wrote the
end-of-row test as `j_base + LANES == N`. With N=4, `j_base` is two bits wide,
so that sum wraps to 0 at precisely the moment it should read 4. The test never
fires and the engine hangs in `WAIT` forever. Comparing against the constant
`N - LANES` avoids the wrap entirely.

**Indexing the wrong dimension.** `j_base` was declared `[$clog2(D)-1:0]`, but
`j` indexes keys, which is N, not features. At N = D = 4 it works by accident.
This is the same bug I hit in the CUDA tiled kernel, where a scores loop said
`j < D` instead of `j < N`. Toy dimensions hide an entire class of error.

**Signed times unsigned poisons the whole expression.** In the online-softmax
datapath, `corr` and `e` are naturally unsigned and `acc` is signed. SystemVerilog
makes an entire expression unsigned if any operand is, so a negative accumulator
silently becomes a huge positive one. Both have to be widened into explicit
signed values first.

**Capturing one cycle after the handshake.** `dot4` originally latched its
inputs the cycle after `valid && ready`. A polite testbench that holds the data
steady never catches this, because a conforming producer is allowed to drop the
data immediately after the transfer. I wrote `tb_dot4_strict.sv` to be as
hostile as the protocol permits and it failed on the first run.

**`fseek` past EOF is silently legal.** In the CUDA loader I seeked
`2 * hdr_len` bytes instead of `hdr_len`. `fseek` returned 0, `ferror` stayed
clean, and only `fread`'s return count knew anything was wrong. That is why
checking `fread`'s return value is load-bearing rather than defensive.

**An assertion that was itself wrong.** `dot4` carried
`initial if (D != (1 <<< M)) $error("D is not a power of 2")`. That check only
matters for the `1/√d` path, because the scale has to become an integer shift.
When `pv` reuses the same module as a plain dot product over the key axis with
`SCALE_EN=0`, any `D` is legal. The assertion fired on correct code. Guarding it
with `SCALE_EN &&` fixed it. Worth remembering that a false alarm in a check is
still a bug, and it is the worse kind, because the reflex is to trust the check
and change the design.

**A ternary between two enum literals will not compile.** This is fine in some
tools and an elaboration error in others:

```systemverilog
next_state = last_pair ? DONE : ISSUE;   // rejected, needs an explicit cast
```

The result of `?:` is a plain vector, and assigning it to an enum-typed variable
is a type violation. I rewrote it as if/else, which reads better anyway.

**The simulator crashed rather than complained.** Passing an unsized array
argument to a SystemVerilog task, then calling it with buses of different widths,
segfaults Icarus 13:

```
tb_attention_top.sv:97: assert: elab_expr.cc:4593:
  failed assertion ntype->type_compatible(net->net_type())
Abort trap: 6
```

I was checking a 4-element bus and a 16-element bus with one helper. Splitting it
into a scalar compare fixed it. When a tool aborts with a file-and-line from its
own source, the bug is in your code's shape, not its logic, and no amount of
staring at the RTL will find it.

**Array versus pointer in the CUDA emulation.** The kernel declares
`extern __shared__ float smem[]`. With `__shared__` defined away on the CPU that
becomes `extern float smem[]`, which does not match a `float*` definition, so the
harness would not link. It has to be backed by a real array. Small, but it is the
kind of thing that makes people give up on emulating a kernel and go back to
waiting for the GPU.

**Two numerics decisions came out backwards on seed 0.** I nearly shipped both.
Truncating division looked better than round-to-nearest, and online softmax
looked twice as accurate as naive. A 200-seed sweep reversed both. Round-to-
nearest cuts mean error 21%, and online softmax is a memory and divide win
rather than an accuracy win. Sixteen output values is not a distribution.

## Accuracy at each level

| Level | softmax implementation | max abs err vs float64 | mean |
|---|---|---|---|
| M2 Python Q8.8 | `np.exp` plus float divide | 0.0063 | 0.0021 |
| M5 RTL naive | 256-entry exp LUT plus restoring divider | 0.0133 | 0.0050 |
| M8 RTL online | same LUT, one reciprocal per row | 0.0065 | 0.0032 |
| M3, M4, M8 CUDA | float32 `expf` | 1.19e-07 | |

The RTL numbers above are seed 0. Across 200 seeds the naive and online paths
are much closer, mean 0.00407 against 0.00369 at BLK=4 and a dead heat at
BLK=1. Rounding in the rebase factor compounds N/BLK times down a row, so
streaming in smaller blocks is slightly less accurate, not more.

## One recurrence, three implementations

```
m_new = max(m_run, max_j s_j)
corr  = exp(m_run - m_new)
l_run = l_run * corr + Σ_j exp(s_j - m_new)
acc_c = acc_c * corr + Σ_j exp(s_j - m_new) · V[j][c]
O_c   = acc_c / l_run
```

| | owns the running state | how the reduction happens |
|---|---|---|
| `rtl/flash_top.sv` | one register file | comparator chain, explicit FSM |
| `cuda/06_attention_flash.cu` | one thread block | warp shuffles and `__syncthreads()` |
| `python/06_triton_attention.py` | one program | `tl.max` over a tile axis |

## Testing the CUDA kernel without a GPU

`cuda/test_flash_cpu.cpp` runs the unmodified kernel body on the CPU: one
`std::thread` per CUDA thread, blocks executed sequentially, and
`__syncthreads()` backed by a real barrier. The emulation boundary is exactly
two functions, `blockReduceMax` and `blockReduceSum`, which use warp shuffles on
the GPU and a shared array here. Everything else is the source nvcc compiles.

```
N=4     D=4    causal=0  max_err=1.484e-07  PASS
N=300   D=64   causal=1  max_err=6.577e-07  PASS
N=512   D=128  causal=0  max_err=3.102e-07  PASS
```

N=300 is deliberately not a multiple of the tile width. The masked tail tile is
where boundary bugs live and a clean power-of-two sweep never visits it.

This catches loop bounds, tile arithmetic, the online recurrence and barrier
placement. A missing `__syncthreads()` deadlocks here loudly instead of
producing plausible garbage on hardware. It proves nothing about coalescing,
occupancy, real races or performance.

## Layout

```
python/
  01_golden_model.py          M1, float64 oracle
  02_fixed_point_model.py     M2, Q8.8 twin and drift metrics
  03_export_data.py           .npy bridge to CUDA, float32
  04_rtl_fixed_model.py       M5 spec, bit-exact integer model, emits golden vectors
  05_online_softmax_model.py  M8 spec, the streaming recurrence
  06_triton_attention.py      M8, Triton kernel, PyTorch op, SDPA benchmark
cuda/
  04_attention_naive.cu       M3, one thread per output row
  05_attention_tiled.cu       M4, shared-memory tiles
  06_attention_flash.cu       M8, fused online softmax, causal, benchmark harness
  cpu_emu.h, test_flash_cpu.cpp   run the kernel with no GPU
rtl/
  dot4.sv          time-multiplexed MAC, SCALE_EN folds 1/√d, ROUND_EN picks rounding
  qkt.sv           S = Q·Kᵀ/√d, LANES parallel MACs
  row_max.sv       the numerical stability trick, one comparator
  exp_rom.sv       exp LUT, shared by both datapaths
  softmax.sv       LUT plus restoring divider
  pv.sv            O = P·V, reuses dot4, free V transpose
  attention_top.sv M5, the naive chain, no top-level FSM
  flash_top.sv     M8, online softmax, state independent of N
  vectors/         golden .hex emitted by the Python spec
docs/project_log.md   full build log, every bug and every measurement
```

## Parameters

Nothing is hard-coded to the toy dimensions. `DW` (word width), `FRAC`
(fraction bits), `N`, `D`, `DV`, `LANES` and `BLK` (the area against latency
knob), `RECIP_SH` (reciprocal precision), and the LUT geometry `EXP_N` and
`EXP_RANGE` are all synthesis-time parameters. CI sweeps `LANES` and `BLK`.

## Claims I am not making

- `1.19e-07` is one float32 ULP near unity for this test case. It is close
  agreement, not a universal error bound.
- valid/ready follows AXI4-Stream handshake semantics. It is not a full AXIS
  port.
- `1/√d` is a shift only when d is a power of four. For d of 8, 32 or 128 you
  need the `1/√2` constant multiply that `dot4` implements as `HALF_MUL`.
- "Attention is memory-bound" depends on shape, dtype, hardware and whether you
  are prefilling or decoding. It is not a law.
- Tiling loads each input once only in the single-block toy case. At real sizes
  tiles reload across thread blocks.
- M8 has no measured GPU numbers yet. The kernel is verified, not timed. The
  Triton file benchmarks against `F.scaled_dot_product_attention`, which
  dispatches to real FlashAttention-2 and cuDNN kernels. Beating that is not the
  goal and this kernel uses no tensor cores; percent of SDPA is the number I
  will report.
