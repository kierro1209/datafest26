# DataFest table-build scripts

## Files

- `src/datafest_config.py` — exact raw schemas, expected file names, SDOH domains, final table list.
- `src/build_datafest_tables.py` — full SQLite-backed build pipeline.
- `scripts/run_pipeline.sh` — convenience command to run the full pipeline.

## Expected project layout

```text
data/
  raw/
    departments.csv
    diagnosis.csv
    encounters.csv
    patients.csv
    providers.csv
    social_determinants.csv
    tigercensuscodes.csv
  interim/
  processed/
src/
  datafest_config.py
  build_datafest_tables.py
scripts/
  run_pipeline.sh
```

## Run

```bash
bash scripts/run_pipeline.sh
```

Or directly:

```bash
python src/build_datafest_tables.py --raw-dir data/raw --export
```

## Outputs

The pipeline creates a SQLite database at:

```text
data/interim/datafest_pipeline.sqlite
```

and, with `--export`, compressed CSV outputs in:

```text
data/processed/
```

Core outputs:

```text
encounter_enriched_with_sdoh.csv.gz
patient_timeline_event.csv.gz
patient_timeline_sequence.csv.gz
patient_timeline_transition.csv.gz
patient_timeline_summary.csv.gz
model_feature_matrix_patient.csv.gz
```

Supporting diagnosis outputs:

```text
diagnosis_episode.csv.gz
journey_episode.csv.gz
journey_event_sequence.csv.gz
journey_transition.csv.gz
diagnosis_group_summary.csv.gz
```

QC output:

```text
data/processed/join_validation_report.md
```

## Notes

- Raw/staging column names are exact and validated before loading.
- All joins follow `datafest_relation_model.md`.
- All final tables follow the updated patient-timeline-centered `data_structure_design.md`.
- SDOH answers are preserved as answer tokens; no risk scoring is assumed.
- Provider roles remain separate: `ProviderDurableKey`, `AttendingProviderDurableKey`, and `DischargeProviderDurableKey`.

## Event-level joined dataset

The updated pipeline also builds:

```text
event_enriched.csv.gz
```

This is a long event table. With the currently documented CSVs, the only event-like rows available are:

1. one `ENCOUNTER` event per `EncounterKey`; and
2. one `SDOH_RESPONSE` event per answered SDOH question linked to an encounter.

Each row carries the full joined encounter context: patient info, diagnosis info, department/hospital info, provider-role info, geography info, encounter info, and SDOH context. If a lab/procedure such as a blood test is represented only through `Type`, `VisitType`, or `VisitTypeDescription`, it appears as an `ENCOUNTER` event. A true one-row-per-blood-test table cannot be generated unless a separate lab/procedure/orders CSV exists.
