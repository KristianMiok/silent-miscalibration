#!/usr/bin/env bash
# EDS-2026-0083 revision: re-run every revision analysis in dependency order and check
# that the committed outputs (reports/rev_eds, figures/rev_eds) are reproduced unchanged.
# Record-level surfaces under data/rev_eds are reused when present; delete that directory
# to regenerate them from scratch (several hours). Nothing is committed.
# Usage: scripts/reproduce_rev_eds.sh [python]    (default: .venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${1:-.venv/bin/python}"
LOG="${TMPDIR:-/tmp}/reproduce_rev_eds_$(date +%Y%m%d_%H%M%S).log"
STEPS=(
  rev_eds_item1_protocol_a_oos
  rev_eds_item1_protocol_a_join
  rev_eds_gate_protocol_b_repro
  rev_eds_protocol_b_cv
  rev_eds_item1_protocol_b_join
  rev_eds_diag_upstream_missingness
  rev_eds_memorization_control_b
  rev_eds_item3_null_directional
  rev_eds_item2_calibration
  rev_eds_item2b_fixed_map_calibration
  rev_eds_item4_composition
  rev_eds_item4b_metadata
  rev_eds_worked_case_null
  rev_eds_protocol_b_calibrated
  rev_eds_item6_tables
  rev_eds_null_coverage_flips
  rev_eds_check_interval
  rev_eds_odonata_null
  rev_eds_worked_case_null_coverage
  rev_eds_protocol_a_null_panel
  rev_eds_figures
)
fail=0
echo "reproduce_rev_eds at $(git rev-parse --short HEAD), $("$PY" --version 2>&1); full log: $LOG"
for s in "${STEPS[@]}"; do
  t0=$(date +%s)
  echo "===== $s" >>"$LOG"
  if "$PY" "scripts/$s.py" >>"$LOG" 2>&1; then st=ok; else st=FAILED; fail=$((fail + 1)); fi
  printf '%-40s %-7s %6ss\n' "$s" "$st" "$(( $(date +%s) - t0 ))"
done
changed=$(git status --porcelain -- reports/rev_eds figures/rev_eds)
echo
if [ -z "$changed" ]; then echo "committed outputs reproduced unchanged"; else echo "outputs differing from the committed state:"; echo "$changed"; fi
echo "failed steps: $fail"
[ "$fail" -eq 0 ] && [ -z "$changed" ]
