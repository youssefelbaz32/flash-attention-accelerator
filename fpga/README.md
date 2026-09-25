# FPGA flow: ZUBoard 1CG

Target part `xczu1cg-sbva484-1-e`. Everything here is run from the project root.

```bash
vivado -mode batch -source fpga/package_ip.tcl          # package the IP
vivado -mode batch -source fpga/build_bd.tcl            # default N=4 D=4 BLK=2
vivado -mode batch -source fpga/build_bd.tcl -tclargs 8 16 16 16 100   # BLK N D DV MHz
./fpga/sweep.sh                                         # the whole Pareto curve
```

On Windows, run the sweep from Git Bash with the launcher named explicitly:
`VIVADO=/c/Xilinx/Vivado/2024.1/bin/vivado.bat ./fpga/sweep.sh`.

Verified with Vivado 2024.1. Each build writes `attn_<tag>.bit`, `.hwh` and
`.xsa` to `fpga/build/`. PYNQ needs the `.bit` and `.hwh` side by side.

The project is part-based, not board-based: the Avnet ZUBoard board files are
not installed, so the PS DDR and MIO settings in the `.xsa` are Vivado's
defaults, not the board's. That does not matter under PYNQ, which boots with
its own PS configuration and only loads the PL. It would matter for a bare-metal
or PetaLinux flow built from this `.xsa`.

## Results (Vivado 2024.1, 100 MHz PL clock)

| N | D | BLK | LUT | FF | DSP | BRAM | WNS (ns) | fmax (MHz) |
|---|---|---|---|---|---|---|---|---|
| 4 | 4 | 2 | 5,323 | 7,812 | 8 | 3 | +1.97 | 124.5 |
| 16 | 16 | 2 | 11,159 | 24,505 | 8 | 3 | +1.03 | 111.4 |
| 16 | 16 | 4 | 11,735 | 25,621 | 10 | 3 | +0.18 | 101.8 |
| 16 | 16 | 8 | 12,771 | 27,830 | 14 | 3 | -0.47 | 95.5 |
| 16 | 16 | 16 | 13,642 | 32,144 | 22 | 3 | -5.38 | 65.0 |

Counts are the whole design, PS glue and DMA included (the part has 37,440 LUTs,
74,880 FFs, 216 DSPs). fmax is `1000 / (10 - WNS)`. BLK 8 and 16 miss timing
at 100 MHz; their bitstreams exist but should not go on the board as is.

The limit is the datapath, not the size of the part. The critical path starts
at a lane's dot-product DSP, goes through the max across lanes and ends in the
online-softmax rebase (`l_run`). It is 23 logic levels at BLK=2 and 50 at
BLK=16, so each doubling of BLK costs clock. The first build, at 150 MHz,
failed by 0.97 ns, which is why the default clock is now 100 MHz. Pipelining
the cross-lane max is the fix that would move the knee.

## What gets built

```
PS DDR --HP0--> AXI DMA MM2S --AXIS 32b--> accelerator --AXIS 32b--> DMA S2MM --HP0--> PS DDR
PS LPD --AXI-Lite--> accelerator control registers
accelerator irq --> xlconcat --> PS pl_ps_irq0[0]
```

DMA is in simple mode, not scatter-gather: one contiguous buffer per direction
is all this needs and it is far easier to drive from PYNQ.

## Two things that will waste your afternoon if you skip them

**Add `exp_lut.hex` to the project sources.** `$readmemh` resolves a bare
filename against the project, not against the disk. If the file is not added,
the ROM elaborates full of X, the accelerator returns zeros, and there is no
error message anywhere. `package_ip.tcl` adds it and marks it as a Memory
Initialization File.

**The port names in `attention_axi.sv` are load-bearing.** Vivado infers AXI
interfaces from `s_axi_lite_*`, `s_axis_*` and `m_axis_*`. Rename any of them
and packaging silently produces loose scalar ports instead of interfaces, which
you then map by hand. `package_ip.tcl` checks for all four inferred interfaces
and prints an error if one is missing, so this fails loudly rather than three
steps later in block design.

## On the board

Avnet publishes a PYNQ v3.0.1 image for this board:
<https://github.com/Avnet/ZUBoard_1CG-PYNQ>

```bash
sudo python3 python/09_pynq_driver.py --bit fpga/build/attn_N16_D16_BLK4.bit
```

The driver reads `BUILD_ID` and the `PARAM` registers first and refuses to run
on a shape mismatch, then diffs the hardware output against
`python/05_online_softmax_model.py` running in the same process. The golden
model is the self test; there is no wire format between the two sides to get
wrong.

## What to record at each sweep point

| | where it comes from |
|---|---|
| WNS, and therefore fmax | `report_timing_summary`, written to `timing.txt` |
| LUT / FF / DSP / BRAM | `report_utilization`, written to `utilization.txt` |
| cycles per phase | the accelerator's own counters, read by the driver |
| load stall fraction | `CYC_LOAD_STALL / CYC_LOAD` |

The last one is the one to watch. If it climbs as BLK increases, the DMA has
become the limit and further lanes are wasted silicon.

Timing paths and the optimization backlog are tracked in
[docs/07_fpga_timing_and_optimizations.md](../docs/07_fpga_timing_and_optimizations.md).
