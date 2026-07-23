#!/usr/bin/env bash
# Regenerate the analyst board the PM layer reads.
#
# The PM does not run the analysts — it replays them from disk, so this is the only
# place analyst spend happens and the PM can then be iterated on for free. Every leg
# must share one config: `ViewBoard` refuses to assemble a board whose legs disagree
# on model, window, or prompt arm, because a "meeting" of analysts run under different
# arms is a comparison of arms wearing a meeting's clothes.
#
# Roughly $2.40 per driver over a 10-year window on sonnet, so ~$17 for all seven.
# To repair a single leg, pass its name:  ./scripts/run_board.sh inflation
#
# Usage:  ./scripts/run_board.sh [driver ...]
set -euo pipefail
cd "$(dirname "$0")/.."

START="${START:-2016-01-01}"
END="${END:-2025-12-31}"
MODEL="${MODEL:-claude-sonnet-5}"
SUFFIX="${SUFFIX:-_on}"
MEMORY="${MEMORY:---memory}"
OUTDIR="${OUTDIR:-reports/ab}"

DRIVERS=("$@")
if [ ${#DRIVERS[@]} -eq 0 ]; then
  DRIVERS=(inflation inflation_expectations labor_tightness term_premium
           financial_conditions balance_sheet curve_slope)
fi

mkdir -p "$OUTDIR"
echo "[board] drivers=${DRIVERS[*]}"
echo "[board] window=$START..$END model=$MODEL suffix=$SUFFIX"

for d in "${DRIVERS[@]}"; do
  out="$OUTDIR/${d}${SUFFIX}.jsonl"
  if [ -f "$out" ]; then
    cp "$out" "$out.bak"
    echo "[board] $d — existing run backed up to $out.bak"
  fi
  echo "[board] $d — running"
  python3 -m src.run_analyst_ic \
    --driver "$d" --start "$START" --end "$END" --model "$MODEL" $MEMORY \
    --out "$out" > "$OUTDIR/_${d}${SUFFIX}.log" 2>&1
  n=$(wc -l < "$out" | tr -d ' ')
  echo "[board] $d — $n records"
done

echo "[board] done. Verify with:"
echo "  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_pm_board.py -q"
