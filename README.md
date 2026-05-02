# datafest26

## Training the patient event sequence model

Training reads **only** a tokenized sequence artifact and its vocabulary metadata:

| Input | Role |
|--------|------|
| `patient_sequences*.pt` | Integer token streams per patient (`type_ids`, `gap_ids`, …) produced upstream |
| `sequence_model_vocab*.json` | Mappings and metadata merged with the `.pt` (must match how that `.pt` was built) |

No raw CSV or live tokenization happens in the trainer.

### Run

From the repo root:

```bash
python modelling/train_patient_event_model.py \
  --input modelling/token_sequence_model/patient_sequences_encounter_only_with_sdoh_status_and_fips.pt \
  --vocab-json modelling/token_sequence_model/sequence_model_vocab_encounter_only_with_sdoh_status_and_fips.json \
  --output-dir data/processed/patient_event_model
```

If `--vocab-json` is omitted or missing, the script tries a **companion** file next to `--input`:  
`patient_sequences_<suffix>.pt` → `sequence_model_vocab_<suffix>.json`.  
Override with `--vocab-json` whenever your vocab filename does not follow that pattern or you need an explicit pairing.

### Split strategies (`--split-strategy`)

- **`temporal` (default)** — Within each patient timeline, target positions are split into train / validation / test regions (early vs middle vs late events). Loss is computed only on train positions; metrics use val and test positions separately. Fractions are normalized if they do not sum to 1:
  - `--temporal-train-frac` (default `0.7`)
  - `--temporal-val-frac` (default `0.15`)
  - `--temporal-test-frac` (default `0.15`)
  - Optional `--top-k-accuracy K` for top-*K* hit rate on val/test when `K > 1`.

- **`patient_holdout`** — Random split of **whole patient sequences** into train vs validation (`--valid-fraction`, default `0.1`). No separate test split in this mode.

### Useful training flags

| Flag | Notes |
|------|--------|
| `--epochs`, `--batch-size`, `--lr` | Training loop |
| `--max-seq-len` | Truncate sequences |
| `--backbone` | `transformer` (default), `gru`, or `lstm` |
| `--d-model`, `--n-layers`, `--n-heads`, `--dropout` | Model size |
| `--w-type`, `--w-gap`, … | Per-task loss weights |

### Outputs (`--output-dir`)

- `patient_event_model.pt` — weights and run args  
- `patient_event_model_artifacts.json` — vocab copies, metadata, training history (JSON lines printed each epoch include `train`, `valid`, and under temporal split also `test`)

See [`tokenization.md`](tokenization.md) for how sequence `.pt` artifacts are produced upstream.
