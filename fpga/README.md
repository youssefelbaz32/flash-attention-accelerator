# FPGA flow: ZUBoard 1CG

Target part `xczu1cg-sbva484-1-e`. Everything here is run from the project root.

```bash
vivado -mode batch -source fpga/package_ip.tcl          # package the IP
vivado -mode batch -source fpga/build_bd.tcl            # default N=4 D=4 BLK=2
vivado -mode batch -source fpga/build_bd.tcl -tclargs 8 16 16 16   # BLK N D DV
./fpga/sweep.sh                                         # the whole Pareto curve
```

**Not yet run.** These scripts are written against the documented Vivado TCL API
but have not been executed, because there is no Vivado on the machine they were
written on. Expect to fix something the first time. The most likely candidates
are the `apply_bd_automation` config strings, which change between Vivado
versions, and the exact `CONFIG.PSU__*` names for the ZUBoard board preset.

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
sudo python3 python/09_pynq_driver.py --bit fpga/build/attn_N16_D16_BLK8.bit
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
