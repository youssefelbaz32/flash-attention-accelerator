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
| 2026-09-26 | N16 D16 BLK2, T2+T3 | 150 MHz | -1.463 | 122 MHz | 21 | `c_cnt_reg` -> `o_mem_reg` (output scaling multiply) |
| 2026-09-26 | N16 D16 BLK4, T2+T3 | 150 MHz | -1.375 | 123 MHz | 22 | `c_cnt_reg` -> `o_mem_reg` |
| 2026-09-26 | N16 D16 BLK16, T2+T3 | 150 MHz | -1.025 | 129 MHz | 21 | `acc_reg` -> `o_mem_reg` |
| 2026-09-26 | N16 D16 BLK2, +T8 | 150 MHz | +0.325 | 156 MHz | 13 | `c_cnt_reg` -> `acc_reg` (REBASE multiply) |
| 2026-09-26 | N16 D16 BLK4, +T8 | 150 MHz | +0.415 | 158 MHz | 15 | `c_cnt_reg` -> `acc_reg` (REBASE multiply) |
| 2026-09-26 | N16 D16 BLK8, +T8 | 150 MHz | +0.297 | 155 MHz | 17 | `s_blk_reg` -> `l_rebased` (EXPF: score, exp ROM, running sum) |
| 2026-09-26 | N16 D16 BLK16, +T8 | 150 MHz | -0.278 | 142 MHz | 20 | `gen_lane[15].u_dot/acc_reg` (DSP) -> `m_new_r_reg` (16-way max tree) |
| 2026-09-26 | N16 D16 BLK4 FOLD_PAR, +T8 | 150 MHz | +0.484 | 160 MHz | 13 | `b_cnt_reg` -> FOLD multiplier input (V row select) |
| 2026-09-26 | N16 D16 BLK16 FOLD_PAR, +T8 | 150 MHz | -1.254 | 125 MHz | 19 | `gen_lane[14].u_dot/acc_reg` (DSP) -> `m_new_r_reg` |
| 2026-09-26 | N16 D16 BLK4, +T1 | 150 MHz | +0.073 | 150 MHz | 1 | `j_base_reg` -> `gen_lane[2].u_dot/b_reg` (K row select fanout) |
| 2026-09-26 | N16 D16 BLK16, +T1 | 150 MHz | +0.279 | 155 MHz | 14 | `s_blk_reg` -> `m_new_r_reg` (max tree, now register to register) |
| 2026-09-26 | N16 D16 BLK4 FOLD_PAR, +T1 | 150 MHz | +0.328 | 156 MHz | 11 | `b_cnt_reg` -> FOLD multiplier input |
| 2026-09-26 | N16 D16 BLK8 FOLD_PAR, +T1 | 150 MHz | +0.363 | 157 MHz | 12 | `b_cnt_reg` -> FOLD multiplier input |
| 2026-09-26 | N16 D16 BLK16 FOLD_PAR, +T1 | 150 MHz | +0.108 | 151 MHz | 13 | `b_cnt_reg` -> FOLD multiplier input |

fmax = 1000 / (period - WNS). The "150 MHz" clock is really 148.1 MHz (6.75 ns),
which is why a build with positive slack can show an fmax of 149.8. About 55% of every path is routing. The BLK=8
build of the T2+T3 sweep failed inside Vivado (a Tcl interpreter error while
generating the PS IP), not in the design, and was rerun with the next batch.

After T2+T3 the lane max is gone from the top of the report. BLK=16 went from
65 to 129 MHz, and fmax no longer falls with BLK. The new critical path is
the output scaling in SCALE_OUT: `acc[c_cnt]` through the 37 x 25-bit
reciprocal multiply (a two-DSP cascade), rounding, saturation and the write
into `o_mem`, all in one cycle.

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
| T1 | Register `d4_s` before the max (one extra SCORE_WAIT cycle per block) | takes the DSP clock-to-out and first routing hop off the path | done (2026-09-26, new MAXR state reads `s_blk`; BLK16 142 -> 155 MHz, every build now meets 150 MHz) |
| T2 | Replace the linear max with a balanced tree | depth from BLK compares to log2(BLK); should flatten fmax across BLK | done (2026-09-26, with T3: BLK16 65 -> 129 MHz, fmax now flat in BLK) |
| T3 | Pipeline the max: register `m_new_c`, compute `exp(m_run - m_new)` next cycle | splits the cone in two; probably the single biggest fmax win | done (2026-09-26, new CORR state, +1 cycle per block, bit-exact) |
| T4 | Make the exp ROM synchronous (registered output, can sit in BRAM) | removes the ROM from the combinational path; costs a cycle in EXPF | idea |
| T5 | Use the DSP's internal pipeline registers (MREG/PREG) in `dot4` and the rebase multiplies | frees fabric levels, DSPs are nearly free here (22 of 216 at BLK16) | idea |
| T6 | Default PL clock 150 -> 100 MHz | makes BLK <= 4 timing-clean | done (2026-09-25, all N16 BLK<=4 builds) |
| T7 | Re-target 150 MHz once T2 + T3 are in | the goal the scripts were first written for | done (2026-09-26, with T1 and T8: all of BLK 2 to 16, serial and FOLD_PAR, meet 150 MHz) |
| T10 | Retime the FOLD multiplier input (register the V row select on `b_cnt`) | the worst path in every FOLD_PAR build | idea |
| T8 | Pipeline the output scaling (operand, product, round/saturate/write) | the critical path after T2+T3; writes land 2 cycles late, which costs no cycles | done (2026-09-26: BLK 2, 4, 8 now meet 150 MHz, zero added cycles) |
| T9 | Build with `FPGA_WORK` at a short path | the first T8 batch failed on Windows' 260-char path limit and silently reused the old IP | done (2026-09-26) |

### Cycles (throughput)

| # | idea | expected effect | status |
|---|---|---|---|
| C1 | Parallelise FOLD: DV MACs instead of one, so a block folds in BLK cycles, not BLK*DV | FOLD is BLK*DV cycles per block, i.e. N*DV per row whatever BLK is. BLK currently only speeds up the dot products, so it cannot pay off until this is done | done (2026-09-26, `FOLD_PAR=1`: N16 BLK16 compute 1,520 cy vs 5,600 serial, bit-exact, 67 DSPs, meets 150 MHz) |
| C2 | Parallelise REBASE the same way (DV multiplies in one cycle) | saves DV-1 cycles per block | done, part of `FOLD_PAR=1` |
| C3 | Overlap the next block's dot products with this block's EXPF/FOLD | hides the dot4 latency | idea |
| C4 | Pack two 16-bit elements per 32-bit stream beat | halves load and store beats; matters when the DMA is the limit (`CYC_LOAD_STALL`) | idea. With `FOLD_PAR=1` at N16 BLK16, load (1,147 cy) is now 38% of the run, so this is the next cycle win |

### Area

| # | idea | expected effect | status |
|---|---|---|---|
| A1 | Q/K/V memories are flops (24.5k FF at N16 BLK2, vs 7.8k at N4). Move them to BRAM / LUTRAM | big FF cut as N grows; needs the load path and lane reads to tolerate a read cycle | idea |

## How to add a row

After a build, take the first path under `Max Delay Paths` in `timing.txt`
(Slack, Source, Destination, Logic Levels) and the LUT/FF/DSP lines from
`utilization.txt`, and add them here. When a backlog item is tried, record the
build tag and what WNS did, even if it made things worse.
