# Sequence Model Tokenization README

## Purpose

This folder contains the final tokenized patient sequences for the next-encounter sequence model.

The project goal is to predict a patient's next clinical encounter and the information needed for resource allocation:

- **WHAT:** visit type, visit description, diagnosis group, and detailed diagnosis
- **WHEN:** time gap until the next clinical encounter
- **WHERE:** care setting, department type, department specialty, department volume, and region
- **CONTEXT:** static patient profile, approximate home-area geography, and rolling SDOH status known so far

The final design is:

```text
one patient -> one ordered sequence of clinical encounters
```

The final artifact is **encounter-only**. SDOH response rows are **not** prediction targets and are **not** sequence steps. Instead, SDOH information is converted into rolling latest-status streams that provide context at each encounter.

---

## Final model-ready files

Use these final artifacts for modeling:

```text
patient_sequences_encounter_only_with_sdoh_status_and_fips.pt
sequence_model_vocab_encounter_only_with_sdoh_status_and_fips.json
```

The main modeling file is:

```text
data/processed/sequence_model/patient_sequences_encounter_only_with_sdoh_status_and_fips.pt
```

The matching vocab/metadata file is:

```text
data/processed/sequence_model/sequence_model_vocab_encounter_only_with_sdoh_status_and_fips.json
```

Earlier intermediate artifacts may exist, but they are not the recommended modeling input.

---

## Important design change: no grand event token

Earlier artifacts used a composite event token called `event_token_ids`.

That old token looked conceptually like:

```text
EVENT_COMPOSITE::SRC=ENCOUNTER|TYPE=OFFICE_VISIT|DESC=FOLLOW_UP|DXG=E11|SETTING=OP_FACE|DEPT_TYPE=CLINIC|DEPT_SPEC=ENDOCRINOLOGY|VOL=HIGH|GAP=8_30D|SDOH=UNKNOWN
```

The final artifact **removes** this grand token.

The final model should **not** rely on:

```text
event_token_ids
```

Instead, the event information is factorized into separate encounter-level streams:

```text
type_ids
event_description_ids
group_code_ids
diagnosis_value_ids
setting_ids
dept_type_ids
dept_specialty_ids
facility_size_ids
region_ids
gap_ids
```

This makes the model more interpretable and lets us predict each part of the next encounter separately.

---

## Main PyTorch artifact structure

The `.pt` file contains a Python dictionary.

At the top level, it has fields like:

```json
{
  "sequences": ["list of patient sequence dictionaries"],
  "gap_to_id": {},
  "setting_to_id": {},
  "dept_type_to_id": {},
  "facility_size_to_id": {},
  "region_to_id": {},
  "group_code_to_id": {},
  "diagnosis_value_to_id": {},
  "type_to_id": {},
  "event_description_to_id": {},
  "dept_specialty_to_id": {},
  "patient_context_to_id": {},
  "sdoh_status_to_id": {},
  "metadata": {}
}
```

Each item in `sequences` is one patient.

Each patient sequence contains:

1. Encounter-level lists: one item per clinical encounter.
2. Rolling SDOH status lists: one item per clinical encounter.
3. Static patient context dictionaries: one dictionary per patient.

---

## Final sequence shape

A final sequence may look like this:

```json
{
  "patient_id": "P001",

  "type_ids": [12, 12, 30, 4],
  "event_description_ids": [55, 81, 102, 9],
  "group_code_ids": [41, 41, 41, 88],
  "diagnosis_value_ids": [892, 892, 901, 1440],

  "setting_ids": [5, 5, 5, 0],
  "dept_type_ids": [2, 2, 2, 1],
  "dept_specialty_ids": [14, 14, 23, 7],
  "facility_size_ids": [3, 3, 4, 5],
  "region_ids": [6, 6, 6, 8],

  "gap_ids": [0, 3, 4, 2],

  "sdoh_transportation_needs_latest_status_ids": [0, 2, 2, 2],
  "sdoh_food_insecurity_latest_status_ids": [0, 0, 0, 2],
  "sdoh_housing_stability_latest_status_ids": [0, 0, 0, 0],
  "sdoh_financial_resource_strain_latest_status_ids": [0, 0, 1, 1],

  "patient_context_ids": {
    "PatientBirthYearBin": 12,
    "SexAssignedAtBirth": 2,
    "OmbRace": 4,
    "OmbEthnicity": 1,
    "SmokingStatus": 3,
    "MaritalStatus": 2
  },

  "patient_context_values": {
    "patient_lat": 39.05,
    "patient_lon": -95.67,
    "patient_population": 1320.0,
    "patient_census_block_group_fips": "201770045001"
  }
}
```

For this fake patient, there are 4 clinical encounters.

All encounter-level lists are aligned by position. For example, index `2` means the patient's third encounter. These all describe the same third encounter:

```text
type_ids[2]
event_description_ids[2]
group_code_ids[2]
diagnosis_value_ids[2]
setting_ids[2]
dept_type_ids[2]
dept_specialty_ids[2]
facility_size_ids[2]
region_ids[2]
gap_ids[2]
every sdoh_*_latest_status_ids[2]
```

`patient_context_ids` and `patient_context_values` are patient-level dictionaries. They are not encounter-level lists.

---

## Encounter-only sequence design

Earlier preprocessing had two kinds of rows in `event_stage`:

```text
ENCOUNTER
SDOH_RESPONSE
```

In the final artifact:

```text
ENCOUNTER rows -> kept as sequence steps
SDOH_RESPONSE rows -> removed as sequence steps
```

SDOH is still preserved, but as rolling latest-status context.

This means the model predicts the next clinical encounter, not the next SDOH survey response.

This is better for resource allocation because clinical encounters drive provider demand, department demand, specialty demand, diagnosis-specific planning, and location planning.

---

## Encounter-level WHAT streams

### `type_ids`

Broad encounter type.

Example decoded labels might include:

```text
OFFICE_VISIT
HOSPITAL_ENCOUNTER
ED_VISIT
TELEPHONE
APPOINTMENT
```

Used to predict what kind of encounter is likely next.

### `event_description_ids`

More specific encounter or visit description.

Usually based on `VisitTypeDescription` or `event_description`.

Example decoded labels might include:

```text
FOLLOW_UP
NEW_PATIENT
ROUTINE_VISIT
PROCEDURE_VISIT
EMERGENCY_VISIT
```

Used to predict a more specific visit subtype.

### `group_code_ids`

Integer-coded `GroupCode`.

This is the broad diagnosis group.

Used to predict the next encounter's broad diagnosis category.

### `diagnosis_value_ids`

Integer-coded `DiagnosisValue`.

This is the detailed diagnosis value.

Used to predict the next encounter's detailed diagnosis.

This is important for resource allocation because it gives a more specific clinical signal than `GroupCode`.

---

## Encounter-level WHERE streams

### `setting_ids`

Care setting for the encounter.

Possible labels include:

```text
ED
INPATIENT
HOSP_ADMIT
HOSP_OP
OBS
OP_FACE
NONE
UNKNOWN
```

Used to predict the next encounter's care setting.

### `dept_type_ids`

Department type.

This captures the broad department category.

Used to predict where the next encounter may occur at a department-type level.

### `dept_specialty_ids`

Department specialty.

This captures specialty context such as:

```text
CARDIOLOGY
ENDOCRINOLOGY
FAMILY_MEDICINE
EMERGENCY_MEDICINE
RADIOLOGY
ORTHOPEDICS
```

This is especially useful for provider and specialty capacity planning.

### `facility_size_ids`

Department volume bin.

Despite the name, this is a department event-volume proxy, not true physical facility size.

Possible labels include:

```text
VERY_LOW
LOW
MID
HIGH
VERY_HIGH
MISSING
UNKNOWN
```

Used as a resource-intensity proxy.

### `region_ids`

Rough department geography label.

The tokenizer builds region labels from available department location fields, prioritizing:

```text
department_County -> department_City -> department_PostalCode
```

Examples may include:

```text
COUNTY_SHAWNEE
CITY_TOPEKA
ZIP_66604
UNKNOWN
```

Used to predict the next encounter's approximate service region.

---

## Encounter-level WHEN stream

### `gap_ids`

Time gap category since the previous clinical encounter for the same patient.

After the encounter-only patch, `gap_ids` are recomputed between encounters only.

Possible labels:

```text
START
0D
1_7D
8_30D
31_90D
91_180D
181_365D
365PLUS
UNKNOWN
```

Used to predict when the next encounter is likely to happen.

---

## Rolling SDOH latest-status streams

SDOH response events are not sequence steps in the final artifact.

Instead, each SDOH domain becomes a rolling latest-status stream.

Example fields:

```text
sdoh_transportation_needs_latest_status_ids
sdoh_food_insecurity_latest_status_ids
sdoh_housing_stability_latest_status_ids
sdoh_financial_resource_strain_latest_status_ids
sdoh_utilities_latest_status_ids
sdoh_stress_latest_status_ids
sdoh_depression_latest_status_ids
sdoh_social_connections_latest_status_ids
sdoh_physical_activity_latest_status_ids
sdoh_alcohol_use_latest_status_ids
sdoh_intimate_partner_violence_latest_status_ids
```

Each SDOH status stream is an encounter-level list aligned with the encounter sequence.

The value at position `i` means:

```text
latest known status for that SDOH domain at or before encounter i
```

The status mapping is:

```json
{
  "UNKNOWN_NOT_YET_MEASURED": 0,
  "NEGATIVE_NO_NEED": 1,
  "POSITIVE_NEED_OR_RISK": 2,
  "OTHER_DECLINED_UNABLE_UNSPECIFIED": 3
}
```

Example:

```json
{
  "sdoh_transportation_needs_latest_status_ids": [0, 2, 2, 2]
}
```

Meaning:

```text
encounter 1: transportation status not known yet
encounter 2: transportation need/risk observed
encounter 3: latest known transportation status remains positive
encounter 4: latest known transportation status remains positive
```

Example:

```json
{
  "sdoh_food_insecurity_latest_status_ids": [0, 0, 0, 2]
}
```

Meaning:

```text
encounters 1-3: food insecurity not known yet
encounter 4: food insecurity need/risk observed
```

This design avoids future leakage because the value at each encounter uses only information known at or before that encounter.

### Note on SDOH status quality

The SDOH status labels are generated using a generic text-based heuristic.

This means the fields are useful for modeling and exploration, but they may benefit from future domain-specific refinement.

---

## Static patient context

Static patient context is stored once per patient, not once per encounter.

### `patient_context_ids`

`patient_context_ids` contains one categorical ID per patient-level context field.

Example:

```json
{
  "PatientBirthYearBin": 12,
  "SexAssignedAtBirth": 2,
  "OmbRace": 4,
  "OmbEthnicity": 1,
  "SmokingStatus": 3,
  "MaritalStatus": 2
}
```

These are static patient-level conditioning features.

Typical fields may include:

```text
PatientBirthYearBin
SexAssignedAtBirth
OmbRace
OmbEthnicity
MaritalStatus
SmokingStatus
VitalStatus
MyChartStatus
SexualOrientation
patient_geography_known_flag
patient_block_population_bin
```

Some static SDOH summary fields may also exist in `patient_context_ids` because they were added during earlier patient-context patching.

If present, those static SDOH summary fields should be treated as legacy patient-level summaries. The preferred final SDOH representation is the rolling `sdoh_*_latest_status_ids` streams.

### `patient_context_values`

`patient_context_values` contains numeric or raw patient-level geographic context.

Example:

```json
{
  "patient_lat": 39.05,
  "patient_lon": -95.67,
  "patient_population": 1320.0,
  "patient_census_block_group_fips": "201770045001"
}
```

Interpretation:

```text
patient_lat:
  Latitude of the centroid of the patient's Census block group.
  This is not exact home latitude.

patient_lon:
  Longitude of the centroid of the patient's Census block group.
  This is not exact home longitude.

patient_population:
  Census population count of the patient's home Census block group.

patient_census_block_group_fips:
  Patient's Census block group identifier.
  Stored as a string identifier, not as a continuous numeric feature.
```

Only a subset of patients have non-missing Census geography fields. Missing values are expected.

---

## Patient-level vs journey-level

The final design is patient-level:

```text
one patient -> one encounter sequence
```

It is not journey-level:

```text
one patient + one DiagnosisValue -> one journey sequence
```

Reason: the project goal is to predict the patient's next clinical encounter overall. A patient can have multiple active diagnosis journeys, and cross-journey context can matter for resource allocation.

DiagnosisValue is still preserved and predicted through `diagnosis_value_ids`, but the sequence unit remains the full patient timeline.

---

## Autoregressive model target interpretation

The model should be autoregressive over encounter positions.

At each timestep, the model predicts the next encounter's labels.

WHAT:

```text
type_ids
event_description_ids
group_code_ids
diagnosis_value_ids
```

WHERE:

```text
setting_ids
dept_type_ids
dept_specialty_ids
facility_size_ids
region_ids
```

WHEN:

```text
gap_ids
```

The model also receives context:

```text
rolling SDOH latest-status streams
static patient_context_ids
static patient_context_values
```

The model should not use `event_token_ids`, because the final artifact removes the grand composite event token.

---

## How to load the final `.pt` file

Use:

```python
import torch

path = "data/processed/sequence_model/patient_sequences_encounter_only_with_sdoh_status_and_fips.pt"
artifact = torch.load(path, map_location="cpu")

sequences = artifact["sequences"]
metadata = artifact["metadata"]

print(metadata)
print(len(sequences))
print(sequences[0].keys())
```

Inspect one patient:

```python
s = sequences[0]

print(s["patient_id"])
print(len(s["gap_ids"]))
print(s["type_ids"][:5])
print(s["diagnosis_value_ids"][:5])
print(s["patient_context_ids"])
print(s["patient_context_values"])
```

Check event-level alignment:

```python
fields = [
    "type_ids",
    "event_description_ids",
    "group_code_ids",
    "diagnosis_value_ids",
    "setting_ids",
    "dept_type_ids",
    "dept_specialty_ids",
    "facility_size_ids",
    "region_ids",
    "gap_ids"
]

n = len(s["gap_ids"])

for f in fields:
    assert len(s[f]) == n, (f, len(s[f]), n)

for f in artifact["metadata"]["sdoh_status_fields"]:
    assert len(s[f]) == n, (f, len(s[f]), n)

print("All encounter-level streams align.")
```

---

## How to decode IDs back to labels

The artifact stores mappings like:

```text
type_to_id
diagnosis_value_to_id
gap_to_id
setting_to_id
```

To decode, reverse the mapping:

```python
id_to_type = {v: k for k, v in artifact["type_to_id"].items()}
id_to_gap = {v: k for k, v in artifact["gap_to_id"].items()}
id_to_dx = {v: k for k, v in artifact["diagnosis_value_to_id"].items()}

s = artifact["sequences"][0]

print([id_to_type[i] for i in s["type_ids"][:5]])
print([id_to_gap[i] for i in s["gap_ids"][:5]])
print([id_to_dx[i] for i in s["diagnosis_value_ids"][:5]])
```

For patient context fields:

```python
ctx_maps = artifact["patient_context_to_id"]

field = "SmokingStatus"
id_to_smoking = {v: k for k, v in ctx_maps[field].items()}

s = artifact["sequences"][0]
smoking_id = s["patient_context_ids"][field]

print(id_to_smoking[smoking_id])
```

---

## Recommended model heads

The model should include heads for:

```text
type
event description
group code
diagnosis value
gap
setting
department type
department specialty
department volume / facility size proxy
region
```

The model should also accept:

```text
rolling SDOH latest-status streams as encounter-level covariates
patient_context_ids as static categorical covariates
patient_context_values as static numeric/raw geographic covariates
```

This matches the project goal:

```text
Predict the next clinical encounter's WHAT, WHEN, and WHERE so resources and providers can be planned ahead.
```

---

## Current build quality checks

Important metadata to verify:

```text
diagnosis_patch_length_mismatch = 0
encounter_only_length_mismatch = 0
event_token_ids not present in final patient sequences
event_source_ids not present in final patient sequences
sdoh_domain_ids not present in final patient sequences
```
