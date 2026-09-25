# Attention Accelerator

Single-head scaled dot-product attention, built five times over: NumPy, Q8.8
fixed-point, CUDA, SystemVerilog, and Triton. Every level is checked against the
level above it, and most of them match bit for bit rather than within a
tolerance.

```
S = Q·Kᵀ / √d      →      P = softmax(S)      →      O = P·V
```

```bash
./preflight.sh      # checks the toolchain, prints your GPU's sm_XX
./run_all.sh        # reproduces every number below. No GPU or FPGA needed.
```

On a Windows machine, see [docs/SETUP_windows_gpu.md](docs/SETUP_windows_gpu.md).
Short version: WSL2 for the software flow, native Windows for Vivado.

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
| M8 | FlashAttention-lite, CUDA and Triton | done, timed on an RTX 4070 Laptop GPU | float64 CPU reference, SDPA |

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

**A shared-memory race the CPU emulator could not see.** The M8 CUDA kernel
passed every emulated test. On the first real GPU run it failed about half the
N=512 cases, with errors up to 0.28 that changed from run to run. Both block
reducers broadcast their result through `scratch[0]`. The kernel calls
`blockReduceMax` and then `blockReduceSum` with no barrier between them, so a
fast warp 0 could write its partial sum into `scratch[0]` before a slow warp had
read the max out of it. That warp then rebased its accumulator with the wrong
`m_new`. The emulated reducers happened to have a barrier after the read and the
GPU ones did not. Threads also run one at a time on the CPU, so the emulator
could never have seen it. `compute-sanitizer --tool racecheck` reports 21
hazards before the fix and 0 after. The fix broadcasts through a separate slot,
`scratch[32]`, which is only written between a reduce's two barriers and only
read after the second one, so it costs no extra barrier. Nondeterministic error
is the fingerprint of a race, and the emulation boundary was exactly where it
lived.

**The fp32 check was testing TF32.** The Triton fp32 correctness cases failed at
about 2e-3 against a 2e-4 tolerance. On fp32 inputs `tl.dot` defaults to TF32,
which keeps about 10 mantissa bits, so the kernel was fine and the test was
measuring the wrong precision. Passing `input_precision="ieee"` for fp32 brings
the error down to about 1e-6.

**A tile that fits one GPU and not another.** fp32 at D=128 died with
`OutOfResources: shared memory, Required: 115712, Hardware limit: 101376`. With
64x64 tiles and two pipeline stages, fp32 needs 113 KB, and an sm_89 block gets
99 KB. Cards with larger shared memory, like the A100, hide this. fp32 now uses
one stage. fp16, the benchmarked path, is unchanged.

**The baseline was not the kernel I thought it was.** I wrote that SDPA
dispatches to FlashAttention-2 and cuDNN. On the Windows PyTorch build neither
is compiled in, and SDPA quietly falls back to the memory-efficient kernel. The
only hint is a one-line `UserWarning`. Triton reading 100-123% of SDPA looked
like a result until I forced each backend in turn and found FLASH_ATTENTION and
CUDNN_ATTENTION both unavailable. Always check which kernel the baseline actually
runs.

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
| M3, M4 CUDA | float32 `expf` | 1.19e-07 | |
| M8 CUDA | float32 `__expf`, online rebase | 1.77e-07 | |

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

That last caveat bit on the first real GPU run. See the shared-memory race
under [Bugs worth remembering](#bugs-worth-remembering).

## M8 on a GPU

NVIDIA GeForce RTX 4070 Laptop GPU (sm_89, 36 SMs, 8 GB), driver 581.57, CUDA
13.4, torch 2.4.0+cu124, triton-windows 3.1.0, native Windows.

**What SDPA means here.** This Windows PyTorch build has no FlashAttention-2 or
cuDNN attention compiled in. SDPA runs the memory-efficient (xFormers/CUTLASS)
kernel, so percent of SDPA is percent of that. On a Linux build with
FlashAttention-2 the Triton numbers would be lower.

**Triton**, fp16, batch 1, 16 heads, `triton.testing.do_bench` median. All 16
correctness cases pass, in fp16 and fp32.

| N | D | causal | Triton ms | SDPA ms | % of SDPA | TFLOP/s |
|---|---|---|---|---|---|---|
| 512 | 64 | no | 0.058 | 0.055 | 95% | 18.6 |
| 512 | 128 | no | 0.134 | 0.110 | 82% | 16.0 |
| 2048 | 64 | no | 0.493 | 0.595 | 121% | 34.8 |
| 2048 | 128 | no | 1.533 | 1.423 | 93% | 22.4 |
| 4096 | 64 | no | 2.019 | 2.491 | 123% | 34.0 |
| 4096 | 64 | yes | 1.080 | 1.320 | 122% | 31.8 |
| 4096 | 128 | no | 5.855 | 5.862 | 100% | 23.5 |
| 4096 | 128 | yes | 3.190 | 3.095 | 97% | 21.5 |

Unlike the CUDA kernel, the Triton kernel does use tensor cores: `tl.dot` on
fp16 tiles lowers to MMA. That is most of the gap between the two tables.

**Against cuDNN.** torch 2.14.0+cu130 with triton-windows 3.8 adds cuDNN
attention for fp16. FlashAttention-2 is still missing from every official
Windows build. Same method as above, and all 16 correctness cases still pass.

| N | D | causal | Triton ms | cuDNN ms | % of cuDNN | % of mem-efficient |
|---|---|---|---|---|---|---|
| 512 | 128 | no | 0.117 | 0.089 | 76% | 98% |
| 2048 | 64 | no | 0.516 | 0.523 | 102% | 119% |
| 2048 | 64 | yes | 0.277 | 0.363 | 131% | 131% |
| 2048 | 128 | no | 1.344 | 1.021 | 76% | 107% |
| 4096 | 64 | no | 2.150 | 2.057 | 96% | 126% |
| 4096 | 128 | no | 5.404 | 3.989 | 74% | 110% |
| 4096 | 128 | yes | 2.851 | 2.138 | 75% | 108% |

Triton is at parity with cuDNN for D=64 and about 75% of it at D=128.

**CUDA**, fp32, one head, one thread block per query row, mean of 20 launches.
SDPA is timed at the same shape and dtype. All 24 sweep cases pass. Every
N<=512 case is checked against float64, worst error 1.4e-06.

| N | D | causal | CUDA ms | SDPA ms | % of SDPA | GFLOP/s |
|---|---|---|---|---|---|---|
| 512 | 64 | no | 0.246 | 0.089 | 36% | 214 |
| 2048 | 64 | no | 3.69 | 0.357 | 9.7% | 239 |
| 4096 | 64 | no | 14.5 | 1.00 | 6.9% | 296 |
| 4096 | 128 | no | 28.3 | 1.69 | 6.0% | 303 |
| 4096 | 128 | yes | 14.3 | 1.15 | 8.0% | 300 |

At N=128 the CUDA kernel reads 120-150% of SDPA. That is launch overhead
dominating both sides, not a real win. It plateaus near 300 GFLOP/s because
each block owns one query row and re-streams all of K and V from L2. There is
no query tiling and no tensor cores, so the next step is to give each block a
tile of query rows.

### Roofline

The ceilings are measured on this card, not taken from a spec sheet. The SM
clock held at 2400 MHz, drawing 58 to 72 W.

| ceiling | how | measured |
|---|---|---|
| FP32 compute | 8192^3 SGEMM, TF32 off | 11.8 TFLOP/s |
| FP16 tensor-core compute | 8192^3 fp16 GEMM | 39.2 TFLOP/s |
| DRAM bandwidth | 1 GB device copy | 223 GB/s |
| L2 bandwidth | 8 MB device copy, fits the 32 MB L2 | 847 GB/s |

Ridge points: 53 FLOP/byte against DRAM and 14 FLOP/byte against L2 for FP32.

**CUDA kernel, N=4096 D=128: bound by L2 bandwidth.** Each of the N blocks
streams all of K and V, so the traffic is `N * N * D * 8` bytes. That is 17.2
GB for 8.6 GFLOP, an intensity of 0.5 FLOP/byte, far left of either ridge. K
and V together are 4 MB and fit in L2, so that traffic is L2 traffic. DRAM only
sees the compulsory 8 MB, about 1000 FLOP/byte, which is nowhere near a limit.

| | value | share of ceiling |
|---|---|---|
| achieved compute | 303 GFLOP/s | 2.6% of FP32 |
| modeled L2 traffic | 607 GB/s | 72% of L2 |
| L2 roof at 0.5 FLOP/byte | 424 GFLOP/s | kernel reaches 72% of it |

So the kernel is not slow at arithmetic. It is re-reading the same K and V from
L2 once per query row.

**Triton kernel, N=4096 D=128 fp16:** 25.4 TFLOP/s, 65% of the measured
tensor-core ceiling. cuDNN reaches 34.4 TFLOP/s, 88%. Triton is compute-bound
already, and the remaining gap is instruction scheduling, not the algorithm.

### What Nsight Compute said

Profiled at N=4096 with `ncu --set full`. The kernel runs slower under the
profiler (1.88 GHz, 35 ms), so compare shares, not absolute times.

| metric | D=128 | D=64 | what it means |
|---|---|---|---|
| L2 to L1 bytes | 19.4 GB | 8.58 GB | the model said 17.2 and 8.59: re-streaming confirmed |
| DRAM bytes | 85 MB | 29 MB | 0.6% of peak, not a DRAM problem |
| L1/TEX throughput | 98% | 97% | **the real limit** |
| L2 throughput | 42% | 35% | not saturated yet |
| useful bytes per 32-byte sector | 7.1 | | uncoalesced loads |
| warps active | 97% | 98% | occupancy is fine |
| top stalls, cycles per instruction | long scoreboard 57, MIO throttle 57 | long scoreboard 60, barrier 56 | waiting on memory |

The model got the traffic right and the binding limit wrong. Before L2 bandwidth
could matter, L1 was saturated by uncoalesced loads. Thread `tid` computed its
dot product by walking its own key row, `K[j*D + k]`, so the 32 lanes of a warp
sat D floats apart. Every load touched 32 sectors and used 4 bytes of each.

### Following the profile

Two changes, each aimed at the limit the profile named, each verified the same
way: the 24-case sweep against float64, and `compute-sanitizer` racecheck,
memcheck and synccheck.

1. **Coalesce K** (`06_attention_flash.cu`). Stage K through shared memory 32
   columns at a time, transposed with a padded pitch so both the load and the
   read are conflict-free. The accumulation order is unchanged, and every
   checked case reports the same error as before.
2. **Tile the queries** (`07_attention_flash_qtile.cu`). One warp owns one query
   row, and a block of WR warps shares every K and V tile through shared
   memory, so L2 traffic drops by WR. Every reduction is a warp shuffle, so
   there is no block-wide scratch and the race above cannot recur.

| N | D | causal | original | coalesced | query-tiled, WR=16 | speedup | % of SDPA fp32 |
|---|---|---|---|---|---|---|---|
| 512 | 128 | no | 0.479 ms | 0.340 | 0.181 | 2.7x | 69% |
| 2048 | 64 | no | 3.69 | 3.02 | 1.48 | 2.5x | 24% |
| 2048 | 128 | no | 7.08 | 5.15 | 2.33 | 3.0x | 21% |
| 4096 | 64 | no | 14.5 | 11.1 | 4.71 | 3.1x | 21% |
| 4096 | 128 | no | 28.3 | 19.4 | 7.59 | 3.7x | 22% |
| 4096 | 128 | yes | 14.3 | 10.2 | 3.95 | 3.6x | 29% |

N=4096 D=128 goes from 303 GFLOP/s to 1.13 TFLOP/s, 9.6% of the FP32 ceiling.
Rows per block was swept, not guessed: WR=4 takes 25.8 ms, WR=8 10.4 ms, and
WR=16 7.6 ms. WR=32 needs more than the default 48 KB of shared memory. At
N=128 the query-tiled kernel is slower, because 8 blocks cannot fill 36 SMs.

The next limit is arithmetic in FP32 CUDA cores. SDPA and Triton run the
multiplies on tensor cores, which is the remaining 4-5x.

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
  07_attention_flash_qtile.cu M8, query-tiled: one warp per row, WR rows share K/V
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
- The M8 GPU numbers come from one laptop GPU on native Windows, where SDPA
  falls back to the memory-efficient kernel. Triton at or above 100% of SDPA
  here does not mean it beats FlashAttention-2. See [M8 on a GPU](#m8-on-a-gpu).
