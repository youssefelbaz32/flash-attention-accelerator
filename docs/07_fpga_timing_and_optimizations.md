# FPGA timing paths and optimization backlog

Running record for the ZUBoard 1CG builds (`xczu1cg-sbva484-1-e`, Vivado 2024.1).
Every build adds a row to the timing table; every idea goes in the backlog with
its status, so nothing gets tried twice or forgotten. Numbers come from
`fpga/build/<tag>/timing.txt` and `utilization.txt`.

## Timing paths, worst setup path per build

All paths are inside `acc/inst/u_core` (`flash_top`). Nothing in the PS, DMA or
interconnect has come close to critical.

| date | build | clock | WNS (ns) | fmax | levels | source -> destination |
|---|---|---|---|---|---|---|
| 2026-09-25 | N4 D4 BLK2 | 148 MHz | -0.966 | 130 MHz | 23 | `gen_lane[1].u_dot/acc_reg` (DSP) -> `gen_lane[0].u_dot/l_rebased0` |
| 2026-09-25 | N4 D4 BLK2 | 100 MHz | +1.968 | 124 MHz | 24 | `gen_lane[1].u_dot/acc_reg` -> `l_rebased0` |
| 2026-09-25 | N16 D16 BLK2 | 100 MHz | +1.026 | 111 MHz | 25 | `gen_lane[0].u_dot/acc_reg` -> `l_run_reg[11]` |
| 2026-09-25 | N16 D16 BLK4 | 100 MHz | +0.180 | 102 MHz | 24 | `gen_lane[1].u_dot/acc_reg` -> `l_rebased0` |
| 2026-09-25 | N16 D16 BLK8 | 100 MHz | -0.470 | 96 MHz | 36 | `gen_lane[1].u_dot/acc_reg` -> `l_run_reg[16]` |
| 2026-09-25 | N16 D16 BLK16 | 100 MHz | -5.377 | 65 MHz | 50 | `gen_lane[0].u_dot/acc_reg` -> `l_run_reg[18]` |

fmax = 1000 / (period - WNS). About 55% of every path is routing.

### What the critical path is

It is one combinational cone in `rtl/flash_top.sv`, from the score leaving a
`dot4` lane to the running sum:

1. `d4_s[t]`, straight out of the lane's DSP accumulator, unregistered.
2. `m_new_c`: the max over `m_run` and all BLK lane scores, written as a
   **linear** compare chain (`for t ... if (d4_s[t] > m_new_c)`), so its depth
   grows with BLK, not log2(BLK). This is why levels go 24 -> 36 -> 50 from
   BLK 4 to 8 to 16.
3. `exp_x = m_run - m_new_c`, then clamp, negate and shift in `exp_rom`.
4. The 256-entry exp ROM, which is **asynchronous** (LUT ROM), so it adds its
   own levels rather than a register.
5. `corr` into the `l_run * corr` rebase multiply, which Vivado has absorbed
   into the `l_rebased`/`l_run` logic.

## Optimization backlog

Status: **idea**, **trying**, **done** (with the build that proved it), **dropped** (with why).

### Clock (fmax)

| # | idea | expected effect | status |
|---|---|---|---|
| T1 | Register `d4_s` before the max (one extra SCORE_WAIT cycle per block) | takes the DSP clock-to-out and first routing hop off the path | idea |
| T2 | Replace the linear max with a balanced tree | depth from BLK compares to log2(BLK); should flatten fmax across BLK | idea |
| T3 | Pipeline the max: register `m_new_c`, compute `exp(m_run - m_new)` next cycle | splits the cone in two; probably the single biggest fmax win | idea |
| T4 | Make the exp ROM synchronous (registered output, can sit in BRAM) | removes the ROM from the combinational path; costs a cycle in EXPF | idea |
| T5 | Use the DSP's internal pipeline registers (MREG/PREG) in `dot4` and the rebase multiplies | frees fabric levels, DSPs are nearly free here (22 of 216 at BLK16) | idea |
| T6 | Default PL clock 150 -> 100 MHz | makes BLK <= 4 timing-clean | done (2026-09-25, all N16 BLK<=4 builds) |
| T7 | Re-target 150 MHz once T2 + T3 are in | the goal the scripts were first written for | idea |

### Cycles (throughput)

| # | idea | expected effect | status |
|---|---|---|---|
| C1 | Parallelise FOLD: DV MACs instead of one, so a block folds in BLK cycles, not BLK*DV | FOLD is BLK*DV cycles per block, i.e. N*DV per row whatever BLK is. BLK currently only speeds up the dot products, so it cannot pay off until this is done | idea |
| C2 | Parallelise REBASE the same way (DV multiplies in one cycle) | saves DV-1 cycles per block | idea |
| C3 | Overlap the next block's dot products with this block's EXPF/FOLD | hides the dot4 latency | idea |
| C4 | Pack two 16-bit elements per 32-bit stream beat | halves load and store beats; matters when the DMA is the limit (`CYC_LOAD_STALL`) | idea |

### Area

| # | idea | expected effect | status |
|---|---|---|---|
| A1 | Q/K/V memories are flops (24.5k FF at N16 BLK2, vs 7.8k at N4). Move them to BRAM / LUTRAM | big FF cut as N grows; needs the load path and lane reads to tolerate a read cycle | idea |

## How to add a row

After a build, take the first path under `Max Delay Paths` in `timing.txt`
(Slack, Source, Destination, Logic Levels) and the LUT/FF/DSP lines from
`utilization.txt`, and add them here. When a backlog item is tried, record the
build tag and what WNS did, even if it made things worse.
