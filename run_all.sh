#!/usr/bin/env bash
# One-command reproduction of every result in this repo that does not need a GPU
# or an FPGA. Run from the project root:  ./run_all.sh
#
# Needs: python3 + numpy, iverilog (-g2012), a C++17 compiler.
# GPU-only steps (M3/M4/M8 on hardware, Nsight, Triton) are listed at the end.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

RTL="rtl/dot4.sv rtl/qkt.sv rtl/row_max.sv rtl/exp_rom.sv rtl/softmax.sv rtl/pv.sv"
hr() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

hr "M1/M2  Python float golden + Q8.8 fixed-point"
python3 python/01_golden_model.py | tail -3
python3 python/02_fixed_point_model.py

hr "M5 spec  bit-exact integer model + golden vectors"
python3 python/04_rtl_fixed_model.py | grep -E "error|LUT|scaling|row sums"

hr "M8 spec  online softmax model"
BLK=2 python3 python/05_online_softmax_model.py | grep -E "error|delta|LSB|golden"

hr "M5 RTL  dot4 strict testbench (protocol + saturation + backpressure)"
iverilog -g2012 -o build/dot4_strict.vvp rtl/dot4.sv rtl/tb_dot4_strict.sv 2>/dev/null
vvp build/dot4_strict.vvp | tail -9

hr "M5 RTL  qkt vs golden, LANES sweep"
for L in 1 2 4; do
  iverilog -g2012 -DLANES_OVERRIDE=$L -o build/qkt_L$L.vvp $RTL rtl/tb_qkt.sv 2>/dev/null
  printf "  LANES=%s  " $L; vvp build/qkt_L$L.vvp | grep -E "latency|MATCH|FAIL" | tr -d '\n'; echo
done

hr "M5 RTL  full pipeline, every stage checked, LANES sweep"
for L in 1 2 4; do
  iverilog -g2012 -DLANES_OVERRIDE=$L -o build/top_L$L.vvp $RTL rtl/attention_top.sv rtl/tb_attention_top.sv 2>/dev/null
  echo "  --- LANES=$L"; vvp build/top_L$L.vvp | grep -E "PASS|FAIL|latency|per-stage|COMPLETE" | sed 's/^/  /'
done

hr "M8 RTL  online-softmax pipeline, BLK sweep"
for B in 1 2 4; do
  BLK=$B python3 python/05_online_softmax_model.py > /dev/null
  iverilog -g2012 -DBLK_OVERRIDE=$B -o build/flash_B$B.vvp rtl/dot4.sv rtl/exp_rom.sv rtl/flash_top.sv rtl/tb_flash_top.sv 2>/dev/null
  echo "  --- BLK=$B"; vvp build/flash_B$B.vvp | grep -E "PASS|FAIL|latency|delta|COMPLETE" | sed 's/^/  /'
done
BLK=2 python3 python/05_online_softmax_model.py > /dev/null   # restore default vectors

hr "beyond the toy dimensions: N = D = DV = 16"
N=16 D=16 DV=16 python3 python/04_rtl_fixed_model.py > /dev/null
iverilog -g2012 -DN_OVERRIDE=16 -DD_OVERRIDE=16 -DDV_OVERRIDE=16 -DLANES_OVERRIDE=4 \
  -o build/top16.vvp $RTL rtl/attention_top.sv rtl/tb_attention_top.sv 2>/dev/null
vvp build/top16.vvp | grep -E "PASS|FAIL|latency|per-stage|COMPLETE" | sed 's/^/  /'
N=16 D=16 DV=16 BLK=4 python3 python/05_online_softmax_model.py > /dev/null
iverilog -g2012 -DN_OVERRIDE=16 -DD_OVERRIDE=16 -DDV_OVERRIDE=16 -DBLK_OVERRIDE=4 \
  -o build/flash16.vvp rtl/dot4.sv rtl/exp_rom.sv rtl/flash_top.sv rtl/tb_flash_top.sv 2>/dev/null
vvp build/flash16.vvp | grep -E "PASS|FAIL|latency|delta|COMPLETE" | sed 's/^/  /'
python3 python/04_rtl_fixed_model.py > /dev/null          # restore N=4 vectors
BLK=2 python3 python/05_online_softmax_model.py > /dev/null

hr "M6 planning  host link budget"
python3 python/07_link_budget.py | tail -9

hr "M8 CUDA  fused kernel verified on CPU (no GPU required)"
c++ -std=c++17 -O2 -DCPU_EMU -I cuda -o build/flash_cpu cuda/test_flash_cpu.cpp
./build/flash_cpu

hr "lint  verilator, all RTL"
for m in dot4 qkt row_max exp_rom softmax pv attention_top flash_top; do
  verilator --lint-only -Wno-fatal --top-module $m $RTL rtl/attention_top.sv rtl/flash_top.sv 2>&1 \
    | grep -E "%(Error|Warning)" | grep -v EOFNEWLINE || true
done
echo "  clean"

cat <<'NOTE'

=== Steps that need hardware ===
  GPU (nvcc):
    nvcc -O3 -arch=sm_80 -o build/naive   cuda/04_attention_naive.cu
    nvcc -O3 -arch=sm_80 -o build/tiled   cuda/05_attention_tiled.cu
    nvcc -O3 -arch=sm_80 -o build/flash   cuda/06_attention_flash.cu
    ./build/flash            # toy dims vs data/O_golden.npy, expect ~1.19e-07
    ./build/flash bench      # the real-dimension sweep
    ncu --set full -o flash_prof ./build/flash 4096 128 0

  GPU (triton):
    pip install triton && python3 python/06_triton_attention.py

  FPGA (Vivado):  M7 -- see docs/project_log.md
NOTE
