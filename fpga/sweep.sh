#!/usr/bin/env bash
# The Pareto sweep: one bitstream per (N, D, BLK) point, with WNS and
# utilization captured for each. This is the artifact that turns the simulated
# cycle counts into an area-versus-latency argument.
#
#   ./fpga/sweep.sh
#
# Each point is a full implementation run, so budget roughly 15-30 minutes per
# point on this part. Run it overnight.
set -euo pipefail
cd "$(dirname "$0")/.."

# VIVADO lets this run from Git Bash on Windows, where the launcher is vivado.bat.
VIVADO=${VIVADO:-vivado}
mkdir -p fpga/build

"$VIVADO" -mode batch -nojournal -nolog -source fpga/package_ip.tcl

# BLK must divide N. 216 DSPs on this part means BLK can go much higher than the
# 1/2/4 that simulation swept; the point of going to 16 is to find where fmax
# starts to fall off, which is the knee of the curve.
for cfg in "2 16 16 16" "4 16 16 16" "8 16 16 16" "16 16 16 16"; do
  set -- $cfg
  echo "=== BLK=$1 N=$2 D=$3 DV=$4 ==="
  # -nojournal/-nolog go BEFORE -tclargs, or they become tclargs.
  "$VIVADO" -mode batch -nojournal -nolog -source fpga/build_bd.tcl -tclargs "$@" \
    | tee "fpga/build/log_N$2_BLK$1.txt" | grep -E "^  (clock|WNS|fmax|bitstream|xsa)" || true
done

echo
echo "results:"
grep -HE "^  (WNS|fmax)" fpga/build/log_*.txt || true
