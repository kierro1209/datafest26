#!/usr/bin/env bash
# Patient journey predictability EDA — run from datafest26 repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${INPUT:-data/processed/event_enriched.csv.gz}"
OUT="${OUT:-visuals/eda_story}"
MAX_ROWS="${MAX_ROWS:-}"

EXTRA=()
if [[ -n "$MAX_ROWS" ]]; then
  EXTRA+=(--max-rows "$MAX_ROWS")
fi

python eda/journey_eda.py --input "$INPUT" --output-dir "$OUT" "${EXTRA[@]}"

python eda/next_event_baselines.py --input "$INPUT" --output-dir "$OUT" "${EXTRA[@]}"

python eda/resource_pressure_eda.py --input "$INPUT" --output-dir "$OUT/resource" --quadrant-chart "${EXTRA[@]}"

echo "Done. Figures under $OUT and $OUT/resource"
