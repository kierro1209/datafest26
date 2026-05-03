# datafest26

## Training the patient event sequence model

Training reads **only**:

| Input | Role |
|--------|------|
| `patient_sequences*.pt` | Saved patient sequences with integer token streams per timestep (`type_ids`, `gap_ids`, …) and optional tensors documented in the vocab JSON |
| **`token_sequence_model/final_token_format.json`** (recommended default) | **Single** vocabulary artifact: all categorical maps (`*_to_id`), `special_tokens`, and **`metadata`** describing SDOH fields, patient numerics, external feature layout, and lineage |

There is no separate “internal vs external” vocab file—use **`final_token_format.json`** together with **`patient_sequences_with_external_features.pt`** so mappings stay consistent.

### Default paths

```text
--input   token_sequence_model/patient_sequences_with_external_features.pt
--vocab-json token_sequence_model/final_token_format.json
```

If `--vocab-json` is missing, the script falls back to a **companion** next to `--input` (for `patient_sequences_with_external_features.pt` this resolves to `final_token_format.json`).

### What the model consumes from the `.pt`

- **Core factorized streams** — `type_ids`, `event_description_ids`, `group_code_ids`, `diagnosis_value_ids`, `gap_ids`, `setting_ids`, `dept_type_ids`, `dept_specialty_ids`, `facility_size_ids`, `region_ids`, SDOH status streams.
- **Patient context** — `patient_context_ids` (categorical embeddings broadcast to the sequence) and `patient_context_values` for static numerics. Numeric columns are taken from vocab metadata (`patient_context_numeric_fields_found`) when present (e.g. `PopulationValue`, `CENTLAT`, `CENTLON`), otherwise legacy keys such as `patient_lat` / `patient_lon` / `patient_population` if present.
- **External numeric features (per timestep)** — When metadata sets `has_external_numeric_features`, the loader reads dimension from `external_numeric_features_per_token_dim` (and/or infers it from the first sequence’s tensor). The sequence field name defaults to `external_feature_tensor` (`external_feature_tensor_sequence_key` in metadata). Each row is projected with a linear layer into `d_model` and **added** at every position (input conditioning only; no extra CE head). Names of dimensions are documented under `external_numeric_features_dim_index` in `final_token_format.json`.
- **Optional extra categorical streams** — Any additional `<base>_ids` column aligned with `type_ids` (excluding core and SDOH streams) gets an embedding, head, and loss weighted by `--w-external`.

### Train / validation / test behavior (`--split-strategy`)

- **`temporal` (default)** — For **each patient sequence**, target positions (aligned with events along time, after `--max-seq-len` truncation) are partitioned into **train → validation → test** regions using `--temporal-train-frac`, `--temporal-val-frac`, `--temporal-test-frac`. Training loss uses **train** positions only; reported metrics use **val** and **test** positions separately (no gradient on val/test targets). Optional `--top-k-accuracy K` adds top-*K* accuracy on val/test for every prediction head.
- **`patient_holdout`** — Random split of **entire patient sequences** into train vs validation (`--valid-fraction`). No separate test split in this mode (`test` in logs is `null`).

### Run

From the repo root:

```bash
python modelling/train_patient_event_model.py \
  --input token_sequence_model/patient_sequences_with_external_features.pt \
  --vocab-json token_sequence_model/final_token_format.json \
  --output-dir data/processed/patient_event_model
```

### Useful flags

| Flag | Notes |
|------|--------|
| `--epochs`, `--batch-size`, `--lr` | Training loop |
| `--max-seq-len` | Truncate sequences (also affects temporal split boundaries) |
| `--backbone` | `transformer` (default), `gru`, `lstm` |
| `--d-model`, `--n-layers`, `--n-heads`, `--dropout` | Model size |
| `--w-type`, `--w-gap`, … | Per-head CE weights |
| `--w-external` | Weight for each auto-discovered extra `<base>_ids` stream |

### Outputs (`--output-dir`)

- `patient_event_model.pt` — checkpoint and args  
- `patient_event_model_artifacts.json` — vocab copies, metadata, training history  

Epoch logs include `train`, `valid`, and under temporal split also `test`.

### Exporting per-timestep predictions

After training, run inference to write **CSV** (optionally gzip) with top-1 predictions vs targets for every head:

```bash
python modelling/infer_patient_event_model.py \
  --checkpoint data/processed/patient_event_model/patient_event_model.pt \
  --output data/processed/patient_event_model/predictions_test.csv.gz \
  --split test
```

`--input` / `--vocab-json` default to the paths stored in the checkpoint. Use **`--split`** with **`temporal`** training: `val`, `test`, `both`, or `train`. With **`patient_holdout`**, exports validation patients only (`all_holdout_valid`). Add **`--decode-labels`** for string labels via inverse vocab maps. **`--max-rows`** caps rows for quick checks.

See [`tokenization.md`](tokenization.md) for upstream sequence generation when applicable.
