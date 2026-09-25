#!/usr/bin/env bash
# Every GPU experiment, end to end, unattended.
#
#   ./gpu_run.sh
#
# Writes results/gpu_<date>.md in a form you can paste straight into the blog or
# the README. Safe to re-run. Nothing here needs a GUI.
#
# DESIGN RULE: no single failure aborts the run. Profiling in particular is the
# most likely thing to be blocked (permissions, WSL2), and it would be absurd to
# lose a working benchmark sweep because of it. Each stage records PASS, FAIL or
# SKIP with the reason, and the summary at the end tells you what to chase.

cd "$(dirname "$0")"
mkdir -p build results
OUT="results/gpu_$(date +%Y%m%d_%H%M).md"
exec > >(tee "$OUT") 2>&1

hr(){ printf '\n\n## %s\n\n' "$1"; }
run(){ printf '```\n'; "$@" 2>&1; local rc=$?; printf '```\n'; return $rc; }

echo "# GPU results, $(date -u '+%Y-%m-%d %H:%M UTC')"
echo
echo "Host: \`$(uname -srm)\`"

# ---------------------------------------------------------------- environment
hr "Environment"
GPU="unknown"; ARCH=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  [ -n "$CC" ] && ARCH="-arch=sm_$CC"
  echo "- GPU: **$GPU**  (sm_$CC)"
  echo "- Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  echo "- **No nvidia-smi. Nothing here will run.** Stopping."
  exit 1
fi
command -v nvcc >/dev/null || { echo "- **No nvcc.** Stopping."; exit 1; }
echo "- nvcc: $(nvcc --version | grep release | sed 's/.*release //')"
# -arch=native needs CUDA 11.5+; fall back to the detected sm_XX
nvcc --help 2>/dev/null | grep -q 'native' && ARCH="-arch=native"
echo "- building with: \`$ARCH\`"
grep -qi microsoft /proc/version 2>/dev/null && { WSL=1; echo "- running under **WSL2** (profiling may be restricted)"; } || WSL=0

STATUS=""
mark(){ STATUS="$STATUS\n| $1 | $2 | $3 |"; }

# ---------------------------------------------------------------- build
hr "Build"
BUILT=1
for k in 04_attention_naive:naive 05_attention_tiled:tiled 06_attention_flash:flash; do
  src="cuda/${k%%:*}.cu"; bin="build/${k##*:}"
  if nvcc -O3 $ARCH -o "$bin" "$src" 2>&1 | tail -5; then
    echo "- built \`$bin\`"
  else
    echo "- **FAILED** to build \`$src\`"; BUILT=0
  fi
done
[ $BUILT -eq 1 ] && mark "build" "PASS" "all three kernels" || mark "build" "FAIL" "see log"

# ---------------------------------------------------------------- correctness
hr "Correctness at the project's toy dimensions"
echo "Expect \`~1.19e-07\`, one float32 ULP near unity, the same figure M3 and M4 reported."
if run ./build/flash; then mark "flash toy check" "PASS" "vs data/O_golden.npy"
else mark "flash toy check" "FAIL" "see log"; fi
for b in naive tiled; do
  [ -x build/$b ] && { echo; echo "### $b"; run ./build/$b >/dev/null 2>&1 \
    && mark "$b toy check" "PASS" "" || mark "$b toy check" "FAIL" ""; }
done

# ---------------------------------------------------------------- benchmark
hr "Benchmark sweep"
echo "N in {128, 512, 2048, 4096} x D in {32, 64, 128} x causal, 20 iterations each."
echo "Correctness is checked against a float64 CPU reference where N <= 512."
if run ./build/flash bench; then mark "bench sweep" "PASS" "24 cases"
else mark "bench sweep" "FAIL" "see log"; fi

# ---------------------------------------------------------------- triton
hr "Triton, and the head-to-head against PyTorch SDPA"
if python3 -c "import triton" 2>/dev/null; then
  echo "SDPA dispatches to real FlashAttention-2 / cuDNN. The number to report is"
  echo "**percent of SDPA**, not a win: this kernel uses no tensor cores."
  if run python3 python/06_triton_attention.py; then mark "triton" "PASS" "vs SDPA"
  else mark "triton" "FAIL" "see log"; fi
else
  echo "\`triton\` not installed. \`pip install triton\` (Linux/WSL2 only)."
  mark "triton" "SKIP" "not installed"
fi

# ---------------------------------------------------------------- profiling
hr "Nsight Compute"
if ! command -v ncu >/dev/null 2>&1; then
  echo "\`ncu\` not on PATH. It ships with the CUDA toolkit; on Windows use \`ncu.exe\`."
  mark "nsight" "SKIP" "ncu not found"
elif [ "$WSL" = "1" ]; then
  echo "Under WSL2. Nsight Compute support there is limited, so this is attempted"
  echo "but a failure is expected rather than alarming. If it fails, profile from"
  echo "native Windows with a Windows-built binary."
  if ncu --set full --target-processes all -o results/flash_prof \
         ./build/flash 4096 128 0 >/dev/null 2>&1; then
    mark "nsight" "PASS" "results/flash_prof.ncu-rep"
  else
    echo; echo "Profiling failed under WSL2, as expected. Not a code problem."
    mark "nsight" "SKIP" "WSL2 restriction"
  fi
else
  echo "Needs GPU counter access: if this reports ERR_NVGPUCTRPERM, re-run with sudo."
  if run ncu --set full -o results/flash_prof ./build/flash 4096 128 0; then
    mark "nsight" "PASS" "results/flash_prof.ncu-rep"
    echo
    echo "### Key metrics"
    run ncu --import results/flash_prof.ncu-rep --page details \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__occupancy_limit_registers,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
  else
    mark "nsight" "FAIL" "try: sudo ncu ..."
  fi
fi

# ---------------------------------------------------------------- summary
hr "Summary"
printf '| stage | result | note |\n|---|---|---|'
printf "$STATUS\n"
echo
echo "Raw log: \`$OUT\`"
echo
echo "Next: paste the bench table into the M8 blog entry and the README, and"
echo "replace the seven \`pending GPU\` placeholders on the portfolio."
