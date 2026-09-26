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

# BLK must divide N. Fields: BLK N D DV MHz FOLD_PAR. Serial fold at two lane
# counts, then the parallel fold, where more lanes actually buy cycles.
for cfg in "4 16 16 16 150 0" "16 16 16 16 150 0" \
           "4 16 16 16 150 1" "8 16 16 16 150 1" "16 16 16 16 150 1"; do
  set -- $cfg
  echo "=== BLK=$1 N=$2 D=$3 DV=$4 ${5}MHz FOLD_PAR=$6 ==="
  # -nojournal/-nolog go BEFORE -tclargs, or they become tclargs.
  "$VIVADO" -mode batch -nojournal -nolog -source fpga/build_bd.tcl -tclargs "$@" \
    | tee "fpga/build/log_N$2_BLK$1_FP$6.txt" | grep -E "^  (clock|WNS|fmax|bitstream|xsa)" || true
done

echo
echo "results:"
grep -HE "^  (WNS|fmax)" fpga/build/log_*.txt || true
