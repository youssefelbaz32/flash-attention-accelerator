# Setting up on a Windows laptop with an NVIDIA GPU

Short version: **use WSL2 for everything software, keep Vivado on native
Windows.** Triton does not support native Windows, `run_all.sh` is bash, and the
CPU emulation harness wants g++. WSL2 solves all three at once and passes the
GPU through. Vivado stays native because programming the board needs the USB
JTAG drivers.

## 1. WSL2 with CUDA

Windows 11, or Windows 10 21H2+. In PowerShell as admin:

```powershell
wsl --install -d Ubuntu-22.04
```

Install the normal Windows NVIDIA driver. **Do not install a driver inside
WSL** -- the Windows driver is passed through, and installing another one in
the guest breaks it. Then inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential git python3-pip iverilog verilator
pip install numpy torch triton
```

CUDA toolkit inside WSL (needed for `nvcc`):

```bash
sudo apt install -y nvidia-cuda-toolkit      # simplest
nvidia-smi                                   # should list your GPU
```

Then:

```bash
git clone https://github.com/youssefelbaz32/flash-attention-accelerator
cd flash-attention-accelerator
./preflight.sh        # tells you exactly what is missing, and your sm_XX
./run_all.sh          # everything that needs no GPU or FPGA
```

## 2. The GPU runs

```bash
nvcc -O3 -arch=native -o build/flash cuda/06_attention_flash.cu
./build/flash                 # toy dims vs data/O_golden.npy, expect ~1.19e-07
./build/flash bench           # the real-dimension sweep, under a minute
ncu --set full -o flash_prof ./build/flash 4096 128 0
python3 python/06_triton_attention.py
```

`-arch=native` needs CUDA 11.5 or newer. On anything older, `./preflight.sh`
prints the exact `sm_XX` for your card.

Expect the whole GPU side to take an afternoon including fixing things.

## 3. Vivado, on native Windows

**Check licensing first.** Confirm XCZU1CG is covered by the free Vivado ML
Standard edition before committing to a 1-3 hour, ~100 GB install. Some Zynq
UltraScale+ parts require ML Enterprise.

```
vivado -mode batch -source fpga/package_ip.tcl
vivado -mode batch -source fpga/build_bd.tcl
```

**These scripts have never been run anywhere.** They are written against the
documented TCL API. Budget a few hours. The two most likely breakages:

- `apply_bd_automation` config strings change between Vivado versions
- the ZUBoard `CONFIG.PSU__*` preset names are version and board-file specific

If the board files are not installed, get them from
<https://github.com/Avnet/bdf> and point Vivado at them.

## 4. Why "just flashing" is not the whole job

There is no bitstream in this repo. The pipeline is:

| Step | Status | Time |
|---|---|---|
| RTL written and simulated | **done**, bit-exact at N=4 and N=16 | -- |
| IP packaging | script written, never run | mins + debugging |
| Block design | script written, never run | mins + debugging |
| Synthesis | not started | ~5 min |
| Place and route | not started | ~10 min |
| **Timing closure at 150 MHz** | **unknown** | 0, or a redesign |
| Bitstream generation | not started | ~2 min |
| **Flashing the board** | not started | **~2 min** |
| PYNQ image on SD | not started | 30-60 min |
| First working DMA transfer | not started | hours |

Flashing really is two minutes. It is step 8 of 10.

Simulation and synthesis are different tools with different rules. The RTL is
clean of latches, multi-driver nets and blocking/nonblocking mixups (checked
with `verilator -Wall`), but three things can only be settled by actually
running synthesis:

1. six `initial` blocks containing `$error` parameter checks, which simulators
   honour and synthesis may ignore or reject
2. `$readmemh` path resolution, which is why `package_ip.tcl` adds
   `exp_lut.hex` to the project explicitly
3. **timing closure**, which is simply unknown until place and route reports
   WNS. If it fails at 150 MHz the fix is a lower clock or a pipeline stage,
   and that is a real design change

## 5. On the board

Avnet's PYNQ v3.0.1 image: <https://github.com/Avnet/ZUBoard_1CG-PYNQ>

```bash
sudo python3 python/09_pynq_driver.py --bit fpga/build/attn_N16_D16_BLK8.bit
```

The driver reads `BUILD_ID` and the `PARAM` registers first and refuses to run
on a shape mismatch, then diffs against the bit-exact model in the same Python
process. The `armed` gate means the cycle counters cannot under-report while
you are still debugging the DMA, which is exactly when you would not notice.

## Realistic schedule

| | Optimistic | Realistic |
|---|---|---|
| WSL2 and toolchain | 1 hr | 2-3 hrs |
| All GPU results | 1 hr | half a day |
| Vivado install to first bitstream | 3 hrs | 1 day |
| First working board run | 2 hrs | 1 day |
| Sweep, power, writeup | 2 hrs | half a day |

The GPU half is a much better return on time, and it needs nothing from the
FPGA half.
