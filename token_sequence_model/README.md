# Sequence Model Tokenization README

## Purpose

This folder contains the tokenized patient sequences used for the next-event sequence model.

The goal is to predict the next patient event or visit and the information needed for resource planning:

- WHAT: visit/event type and diagnosis
- WHEN: time until the next event
- WHERE: care setting, department type, department volume, and region

The tokenization is patient-sequence based:

one patient -> one ordered sequence of events

This is different from a journey-only design:

one patient + one DiagnosisValue -> one journey sequence

We keep patient-level sequences because a patient can have multiple diagnosis journeys, and the next visit may depend on their full recent history.

---

## Final output files

The final model-ready files are:

- patient_sequences_for_model_with_diagnosis.pt
- sequence_model_vocab_with_diagnosis.json
- sequence_model_events.csv.gz

Use this file for modeling:

- patient_sequences_for_model_with_diagnosis.pt

The earlier file, patient_sequences_for_model.pt, does not include the patched diagnosis streams.

---

## Main PyTorch artifact

patient_sequences_for_model_with_diagnosis.pt contains a dictionary with:

- sequences
- vocab
- gap_to_id
- setting_to_id
- dept_type_to_id
- facility_size_to_id
- region_to_id
- group_code_to_id
- diagnosis_value_to_id
- metadata

Each item in sequences is one patient:

{
"patient_id": "...",
"event_token_ids": [...],
"gap_ids": [...],
"setting_ids": [...],
"dept_type_ids": [...],
"facility_size_ids": [...],
"region_ids": [...],
"group_code_ids": [...],
"diagnosis_value_ids": [...]
}

All arrays are aligned by event position. Position i in every array describes the same patient event.

---

## Event vs encounter

In this project:

encounter = one recorded interaction in the raw encounter table

event = one timeline item used by the sequence model

One encounter can create multiple events:

- 1 encounter row -> 1 ENCOUNTER event
- 1 SDOH answer row -> 1 SDOH_RESPONSE event

So a patient sequence can include both clinical encounter events and SDOH response events.

For predicting next visits, we may later evaluate primarily on ENCOUNTER events, but SDOH events are useful context.

---

## What an event token looks like

Each event has a controlled composite token.

Example:

EVENT_COMPOSITE::SRC=ENCOUNTER|GRAIN=ENCOUNTER|TYPE=OFFICE_VISIT|DESC=FOLLOW_UP|DXG=E11|SETTING=OP_FACE|DEPT_TYPE=CLINIC|DEPT_SPEC=ENDOCRINOLOGY|VOL=HIGH|GAP=8_30D|SDOH=UNKNOWN

This long token is mapped to an integer ID.

Example:

EVENT_COMPOSITE::SRC=ENCOUNTER|... -> 12345

The patient sequence stores the integer in event_token_ids.

---

## Main event token components

The main event token is intentionally broad and controlled. It includes:

- SRC: event source, such as ENCOUNTER or SDOH_RESPONSE
- GRAIN: event grain, such as ENCOUNTER or SDOH_RESPONSE
- TYPE: visit/event type
- DESC: visit type description or event description
- DXG: diagnosis group code
- SETTING: care setting
- DEPT_TYPE: department type
- DEPT_SPEC: department specialty
- VOL: department event-volume bin
- GAP: time gap bin from previous event
- SDOH: SDOH domain for SDOH response events

We intentionally do not put exact DiagnosisValue inside the main event token, because DiagnosisValue is high-cardinality. Instead, DiagnosisValue is preserved separately in diagnosis_value_ids.

---

## Sequence fields

### event_token_ids

Main broad event-state sequence.

This captures a compact summary of WHAT, WHERE, and WHEN for each event.

Used to predict the next broad event state.

### gap_ids

Time gap category before each event.

Examples:

- START
- 0D
- 1_7D
- 8_30D
- 31_90D
- 91_180D
- 181_365D
- 365PLUS
- UNKNOWN

Used for next-event WHEN prediction.

### setting_ids

Care setting for each event.

Examples:

- ED
- INPATIENT
- HOSP_ADMIT
- HOSP_OP
- OBS
- OP_FACE
- NONE
- UNKNOWN

Used for next-event WHERE / care setting prediction.

### dept_type_ids

Department type for each event.

Used for next-event WHERE / department category prediction.

### facility_size_ids

Department event-volume bin for each event.

Despite the name, this is currently a department volume proxy, not true physical facility size.

Examples:

- VERY_LOW
- LOW
- MID
- HIGH
- VERY_HIGH
- MISSING
- UNKNOWN

Used for resource planning and WHERE/resource-intensity prediction.

### region_ids

Rough department geography label.

The tokenizer creates region labels from available location fields, prioritizing:

department_County -> department_City -> department_PostalCode

Examples:

- COUNTY_SHAWNEE
- CITY_TOPEKA
- ZIP_66604
- UNKNOWN

Used for next-event WHERE / geography prediction.

### group_code_ids

Integer-coded GroupCode for each event.

This is the broad diagnosis category.

Used for next-event WHAT diagnosis-group prediction.

### diagnosis_value_ids

Integer-coded DiagnosisValue for each event.

This is the detailed diagnosis value used to track patient condition/journey more specifically.

Used for next-event WHAT detailed-diagnosis prediction.

This is important for resource allocation because it gives a more specific clinical signal than GroupCode.

---

## Why DiagnosisValue is separate

DiagnosisValue is important, but it has many unique values.

Current build metadata:

- group_code_classes: 1,634
- diagnosis_value_classes: 21,141

Putting DiagnosisValue directly inside event_token_ids would make the event vocabulary much larger and sparser.

So the design is:

event_token_ids = broad event state using GroupCode

diagnosis_value_ids = detailed diagnosis stream

This preserves detailed diagnosis information without making the main event token too sparse.

---

## Patient-level vs journey-level

Current design:

one patient -> one full event sequence

Not current design:

one patient + one DiagnosisValue -> one journey sequence

Reason: our goal is to predict the patient's next visit/event overall. A patient may have multiple active diagnosis journeys, and cross-journey context can matter.

---

## Model target interpretation

The model is autoregressive.

For a patient with:

event_token_ids = [42, 91, 17]

The model receives:

input_ids = [BOS, 42, 91]

and predicts:

target_event = [42, 91, 17]

So it learns:

- given start -> predict event 1
- given event 1 -> predict event 2
- given event 2 -> predict event 3

At the same time, the model predicts aligned auxiliary targets:

- gap_ids -> WHEN
- setting_ids -> WHERE care setting
- dept_type_ids -> WHERE department type
- facility_size_ids -> WHERE department volume/resource proxy
- region_ids -> WHERE geography
- group_code_ids -> WHAT broad diagnosis group
- diagnosis_value_ids -> WHAT detailed diagnosis

---

## Current build metadata

The successful tokenization produced:

n_events_processed: 11,883,143
n_patients_seen: 363,271
n_training_sequences: 319,094
event_vocab_size: 136,482
gap_classes: 9
setting_classes: 8
dept_type_classes: 9
facility_size_classes: 7
region_classes: 11
group_code_classes: 1,634
diagnosis_value_classes: 21,141
diagnosis_patch_length_mismatch: 0

The key validation point is:

diagnosis_patch_length_mismatch = 0

That means group_code_ids and diagnosis_value_ids aligned correctly with the patient event sequences.

---

## Recommended modeling input

Use:

patient_sequences_for_model_with_diagnosis.pt

The model should include heads for:

- event token
- gap
- setting
- department type
- facility size / department volume
- region
- group code
- diagnosis value

This matches the project goal:

Predict the next visit's WHAT, WHEN, and WHERE so resources and providers can be planned ahead.
