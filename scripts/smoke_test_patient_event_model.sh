#!/usr/bin/env bash
# Fast end-to-end check: subset of sequences, temporal train/val/test, 3 epochs,
# checkpoint + artifact JSON, then a capped inference export.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${SMOKE_OUT:-data/processed/patient_event_model_smoke}"

python modelling/train_patient_event_model.py \
  --smoke-test \
  --input token_sequence_model/patient_sequences_with_external_features.pt \
  --vocab-json token_sequence_model/final_token_format.json \
  --output-dir "$OUT"

python modelling/infer_patient_event_model.py \
  --checkpoint "$OUT/patient_event_model.pt" \
  --output "$OUT/predictions_smoke.csv.gz" \
  --split test \
  --batch-size 16 \
  --max-rows 2000

echo "Smoke OK: trained -> $OUT/patient_event_model.pt, infer -> $OUT/predictions_smoke.csv.gz"
