# Attention Accelerator — one algorithm, five levels, all diffed against each other

Single-head scaled dot-product attention, rebuilt from scratch in **NumPy →
fixed-point → CUDA → SystemVerilog → Triton**, where every level is checked
against the level above it — most of them **bit-exactly**, not within a tolerance.

```
S = Q·Kᵀ / √d      →      P = softmax(S)      →      O = P·V
```

```bash
./run_all.sh        # reproduces every number below, no GPU or FPGA needed
```

[![correctness](https://github.com/youssefelbaz32/flash-attention-accelerator/actions/workflows/ci.yml/badge.svg)](../../actions)

---

## Headline results

**The algorithm beat the hardware.** On the FPGA datapath, 4× the multipliers
bought 1.28×; switching from naive to online (FlashAttention-style) softmax
bought 1.83× at *identical* area — and deleted the O(N²) buffers entirely.

| RTL design | multipliers | cycles | speedup | intermediate storage |
|---|---|---|---|---|
| M5 naive, `LANES=1` | 1 | 659 | 1.00× | 2N² words (S and P) |
| M5 naive, `LANES=4` | 4 | 515 | 1.28× | 2N² words |
| **M6 online, `BLK=1`** | **1** | **360** | **1.83×** | **DV+2 words** |
| M6 online, `BLK=4` | 4 | 240 | 2.75× | DV+2 words |

Why more multipliers barely helped — per-stage profile of the naive design:

| LANES | qkt | row_max | **softmax** | pv | total | softmax share |
|---|---|---|---|---|---|---|
| 1 | 97 | 17 | **449** | 102 | 659 | 68% |
| 2 | 49 | 17 | **449** | 54 | 563 | 79% |
| 4 | 25 | 17 | **449** | 30 | 515 | **87%** |

The sequential divider in softmax is 449 cycles no matter what. Adding lanes
only makes it a *larger* fraction of the problem. That measurement is what set
the M6 agenda — and it's the opposite of where most attention tutorials point.

The storage column is the one that decides whether a design fits on a part. At
`N=128, D=DV=64`: the naive path needs `2·128² = 32768` words ≈ **64 KB of
BRAM** for S and P; the online path needs `DV+2 = 66` words ≈ **132 bytes**, and
that number is **independent of N**.

## Accuracy, level by level

| Level | softmax implementation | max abs err vs float64 golden | mean |
|---|---|---|---|
| M2 Python Q8.8 | `np.exp` + float divide | 0.0063 | 0.0021 |
| M5 RTL naive | 256-entry exp LUT + restoring divider | 0.0133 | 0.0050 |
| M6 RTL online | same LUT, one reciprocal per row | 0.0065\* | 0.0032\* |
| M3/M4/M8 CUDA | float32 `expf` | 1.19e-07 | — |

\* seed 0. Over **200 seeds** the online/naive gap is much smaller — mean 0.00369
vs 0.00407 at `BLK=4`, and a dead heat at `BLK=1`. **Online softmax is a memory
and divide win, not an accuracy win**, and the correction factor's rounding
compounds `N/BLK` times down each row, so streaming harder means streaming
slightly less accurately. Two separate fixed-point decisions in this project
came out *backwards* when judged on a single seed.

## Build order — every level diffed against the one above

| # | Level | Status | Checked against | How |
|---|---|---|---|---|
| M1 | Python float64 golden | ✅ | self-check | softmax rows = 1 |
| M2 | Python Q8.8 fixed-point | ✅ | M1 | error budget |
| M3 | CUDA naive | ✅ | M1 | max abs err 1.19e-07 |
| M4 | CUDA tiled (shared mem) | ✅ | M1 + M3 | bit-identical to M3 |
| M5 | SystemVerilog, naive datapath | ✅ | bit-exact integer model | **exact, all 4 stages** |
| M6 | SystemVerilog, online softmax | ✅ | its own integer spec | **exact, every BLK** |
| M7 | FPGA bring-up | ⬜ | needs board access | — |
| M8 | CUDA fused + Triton | ✅ code, ⬜ measured | float64 CPU reference | needs GPU for timings |

## The part that makes "bit-exact" possible

M2 quantizes the matmuls but still calls `np.exp` and divides in float64.
Hardware can't — so **M2 cannot be the golden model for RTL.** It's a precision
study, not an executable spec.

`python/04_rtl_fixed_model.py` is the spec: every operation is an integer
operation the RTL performs, in the same order, with the same truncation and
saturation. It emits `rtl/vectors/*.hex`; the testbenches `$readmemh` them and
compare with **zero tolerance**. A tolerance would hide exactly the bugs this is
for.

It also emits `exp_lut.hex`, which the RTL loads — **one definition of `exp()`,
two consumers.** Software and hardware cannot drift apart by construction.

## One recurrence, three implementations

```
m_new = max(m_run, max_j s_j)
corr  = exp(m_run - m_new)                      ≤ 1.0 — the rebase factor
l_run = l_run * corr + Σ_j exp(s_j - m_new)
acc_c = acc_c * corr + Σ_j exp(s_j - m_new) · V[j][c]
O_c   = acc_c / l_run                           once, at the very end
```

`corr` is the whole idea: when a later block holds a bigger max, everything
already accumulated is wrong by **exactly** `exp(m_old − m_new)` — a *constant* —
so one multiply rebases the entire history. That's why softmax, which looks
irreducibly global, is streamable.

| | who owns the running state | how the reduction happens |
|---|---|---|
| `rtl/flash_top.sv` | one register file | comparator chain, explicit FSM |
| `cuda/06_attention_flash.cu` | one thread block | warp shuffles + `__syncthreads()` |
| `python/06_triton_attention.py` | one program | `tl.max` over a tile axis |

## Verified without the hardware

`cuda/test_flash_cpu.cpp` runs the **unmodified** CUDA kernel body on the CPU:
one `std::thread` per CUDA thread, blocks sequential, `__syncthreads()` backed
by a real barrier. It proves loop bounds, tile arithmetic, the online-softmax
recurrence and barrier placement — a missing `__syncthreads()` **deadlocks
loudly** here instead of producing plausible garbage on hardware. It does not
prove coalescing, occupancy or real races.

```
N=4     D=4    causal=0  max_err=1.484e-07  PASS
N=300   D=64   causal=1  max_err=6.577e-07  PASS   ← ragged tail + causal
N=512   D=128  causal=0  max_err=3.102e-07  PASS
```

`N=300` is deliberately not a multiple of the tile width — the masked tail tile
is where boundary bugs live, and a clean power-of-two sweep never catches them.

## Repository layout

```
python/
  01_golden_model.py          M1 — float64 oracle
  02_fixed_point_model.py     M2 — Q8.8 twin + drift metrics
  03_export_data.py           .npy bridge to CUDA (float32, single source of truth)
  04_rtl_fixed_model.py       M5 spec — bit-exact integer model, emits golden vectors
  05_online_softmax_model.py  M6 spec — the streaming recurrence
  06_triton_attention.py      M8 — Triton kernel + PyTorch op + SDPA benchmark
cuda/
  04_attention_naive.cu       M3 — one thread per output row
  05_attention_tiled.cu       M4 — shared-memory tiles
  06_attention_flash.cu       M8 — fused online softmax, causal, benchmark harness
  cpu_emu.h, test_flash_cpu.cpp   run the kernel with no GPU
rtl/
  dot4.sv          time-multiplexed MAC; SCALE_EN folds 1/√d, ROUND_EN picks rounding
  qkt.sv           S = Q·Kᵀ/√d, LANES parallel MACs
  row_max.sv       the numerical-stability trick, in one comparator
  exp_rom.sv       exp LUT, shared by M5 and M6
  softmax.sv       LUT + restoring divider
  pv.sv            O = P·V, reuses dot4, free V transpose
  attention_top.sv M5 — naive chain, no top-level FSM
  flash_top.sv     M6 — online softmax, O(1)-in-N state
  vectors/         golden .hex emitted by the Python spec
docs/project_log.md   full build log: every bug, every measurement, every reversal
```

## Parameters

Nothing is hard-coded to the toy dimensions. `DW` (word width), `FRAC`
(fraction bits), `N` (sequence), `D` (head dim), `DV` (value dim), `LANES` /
`BLK` (the area↔latency knob), `RECIP_SH` (reciprocal precision), and the exp
LUT geometry (`EXP_N`, `EXP_RANGE`) are all synthesis-time parameters. `LANES`
and `BLK` are swept in CI.

## What's honest about the claims here

- **`1.19e-07` is one float32 ULP near unity** for this test case — close
  agreement, not a universal error bound.
- **valid/ready follows AXI4-Stream handshake *semantics***; it is not a full
  AXIS port.
- **`1/√d` is a shift only when `d` is a power of four.** `d = 8/32/128` need the
  `1/√2` constant multiply that `dot4` implements as `HALF_MUL`.
- **"Attention is memory-bound" is config-dependent** (shape, dtype, hardware,
  decode vs prefill), not a universal law.
- **Tiling "loads each input once"** is true for the single-block toy only; at
  real sizes tiles re-load across thread blocks.
- **M8 has no measured GPU numbers yet** — the kernel is verified, not timed.
  The Triton file benchmarks against `F.scaled_dot_product_attention`, which
  dispatches to real FlashAttention-2/cuDNN. Beating it is not the claim;
  "% of SDPA" is the number that will be reported.
