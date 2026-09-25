#!/usr/bin/env bash
# Check the toolchain BEFORE you start, so a missing package is a line of output
# rather than a confusing build error forty minutes in.
#
#   ./preflight.sh
#
# Exits non-zero if anything required for the CPU-only flow is missing. GPU and
# FPGA tools are reported but never fail the check, because plenty of useful
# work does not need them.
cd "$(dirname "$0")"
ok=0; warn=0

say()  { printf "  %-34s %s\n" "$1" "$2"; }
good() { say "$1" "OK    $2"; }
bad()  { say "$1" "MISS  $2"; ok=1; }
note() { say "$1" "--    $2"; warn=1; }

echo "=== required: CPU-only flow (run_all.sh) ==="
command -v python3 >/dev/null && good "python3" "$(python3 --version 2>&1)" || bad "python3" "install python 3.10+"
python3 -c "import numpy" 2>/dev/null && good "numpy" "$(python3 -c 'import numpy;print(numpy.__version__)')" || bad "numpy" "pip install numpy"
command -v iverilog >/dev/null && good "iverilog" "$(iverilog -V 2>&1 | head -1 | cut -c1-40)" || bad "iverilog" "apt install iverilog   (needs -g2012 support)"
command -v verilator >/dev/null && good "verilator" "$(verilator --version 2>&1)" || bad "verilator" "apt install verilator"

CXX=""
for c in g++ clang++ c++; do command -v $c >/dev/null && { CXX=$c; break; }; done
[ -n "$CXX" ] && good "C++17 compiler" "$CXX" || bad "C++17 compiler" "apt install build-essential"

echo
echo "=== optional: GPU flow ==="
if command -v nvcc >/dev/null; then
  good "nvcc" "$(nvcc --version | grep release | sed 's/.*release //')"
  if command -v nvidia-smi >/dev/null; then
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$cc" ]; then
      good "GPU" "$gpu  (sm_$cc)"
      say "" "      build with: nvcc -O3 -arch=sm_$cc   (or -arch=native)"
    else
      note "nvidia-smi" "no compute_cap; use -arch=native"
    fi
  else
    note "nvidia-smi" "not found; cannot detect the GPU arch"
  fi
else
  note "nvcc" "no CUDA toolkit; the kernel still verifies on CPU"
fi
python3 -c "import triton" 2>/dev/null && good "triton" "$(python3 -c 'import triton;print(triton.__version__)')" \
  || note "triton" "pip install triton  (LINUX/WSL2 only; not supported on native Windows)"
python3 -c "import torch" 2>/dev/null && good "torch" "$(python3 -c 'import torch;print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')" \
  || note "torch" "pip install torch"

echo
echo "=== optional: FPGA flow ==="
command -v vivado >/dev/null && good "vivado" "$(vivado -version 2>/dev/null | head -1)" \
  || note "vivado" "not on PATH; the TCL in fpga/ has never been run anywhere"

echo
if [ $ok -ne 0 ]; then
  echo "  FAIL: something required is missing. run_all.sh will not complete."
  exit 1
fi
echo "  CPU-only flow is ready:  ./run_all.sh"
[ $warn -ne 0 ] && echo "  some optional tools are missing; see the '--' lines above"
exit 0
