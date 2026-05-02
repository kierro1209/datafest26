# Sequence Model Tokenization README

## Purpose

This folder contains the final tokenized patient sequences for the next-encounter sequence model.

The project goal is to predict a patient's next clinical encounter and the information needed for resource planning:

- WHAT: visit type, visit description, diagnosis group, and detailed diagnosis
- WHEN: time gap until the next encounter
- WHERE: care setting, department type, department specialty, department volume, and region
- CONTEXT: static patient profile, approximate home-area geography, and rolling SDOH status known so far

The final design is encounter-sequence based:

one patient -> one ordered sequence of clinical encounters

SDOH response rows are not treated as prediction targets. Instead, SDOH information is converted into rolling latest-status streams that provide context for each encounter.

---

## Final model-ready files

Use these final artifacts for modeling:

- patient_sequences_encounter_only_with_sdoh_status_and_fips.pt
- sequence_model_vocab_encounter_only_with_sdoh_status_and_fips.json

Earlier intermediate artifacts may exist, but the final modeling file is:

patient_sequences_encounter_only_with_sdoh_status_and_fips.pt

---

## Main PyTorch artifact

patient_sequences_encounter_only_with_sdoh_status_and_fips.pt contains a dictionary with:

- sequences
- gap_to_id
- setting_to_id
- dept_type_to_id
- facility_size_to_id
- region_to_id
- group_code_to_id
- diagnosis_value_to_id
- type_to_id
- event_description_to_id
- dept_specialty_to_id
- patient_context_to_id
- sdoh_status_to_id
- metadata

Each item in sequences is one patient.

Each patient sequence contains encounter-level streams, rolling SDOH status streams, and static patient context.

---

## Final sequence shape

A final sequence may look like this:

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
"CensusBlockGroupFipsCode": "201770045001"
}
}

All encounter-level lists are aligned by position.

For example, index 2 means the patient's third encounter:

- type_ids[2]
- event_description_ids[2]
- group_code_ids[2]
- diagnosis_value_ids[2]
- setting_ids[2]
- dept_type_ids[2]
- dept_specialty_ids[2]
- facility_size_ids[2]
- region_ids[2]
- gap_ids[2]
- every sdoh\_\*\_latest_status_ids[2]

all describe the same encounter.

patient_context_ids and patient_context_values are patient-level static context dictionaries, not event-level lists.

## Encounter-level WHAT streams

### type_ids

Broad encounter type.

This describes the general type of visit or clinical interaction.

Example decoded labels might include:

- OFFICE_VISIT
- HOSPITAL_ENCOUNTER
- ED_VISIT
- TELEPHONE
- APPOINTMENT

Used for predicting what kind of encounter is likely next.

### event_description_ids

More specific encounter or visit description.

This is usually based on VisitTypeDescription or event_description.

Example decoded labels might include:

- FOLLOW_UP
- NEW_PATIENT
- ROUTINE_VISIT
- PROCEDURE_VISIT
- EMERGENCY_VISIT

Used for predicting a more specific visit subtype.

### group_code_ids

Integer-coded GroupCode.

This is the broad diagnosis group.

Used for predicting the next encounter's broad diagnosis category.

### diagnosis_value_ids

Integer-coded DiagnosisValue.

This is the detailed diagnosis value.

Used for predicting the next encounter's detailed diagnosis.

This is important for resource planning because it gives a more specific clinical signal than GroupCode.

---

## Encounter-level WHERE streams

### setting_ids

Care setting for the encounter.

Possible labels include:

- ED
- INPATIENT
- HOSP_ADMIT
- HOSP_OP
- OBS
- OP_FACE
- NONE
- UNKNOWN

Used for predicting the next encounter's care setting.

### dept_type_ids

Department type.

This captures the broad department category.

Used for predicting where the next encounter may occur at a department-type level.

### dept_specialty_ids

Department specialty.

This captures specialty context such as cardiology, endocrinology, family medicine, emergency medicine, etc.

This is especially useful for provider and specialty capacity planning.

### facility_size_ids

Department volume bin.

Despite the name, this is a department event-volume proxy, not true physical facility size.

Possible labels include:

- VERY_LOW
- LOW
- MID
- HIGH
- VERY_HIGH
- MISSING
- UNKNOWN

Used as a resource-intensity proxy.

### region_ids

Rough department geography label.

The tokenizer builds region labels from available department location fields, prioritizing:

department_County -> department_City -> department_PostalCode

Examples may include:

- COUNTY_SHAWNEE
- CITY_TOPEKA
- ZIP_66604
- UNKNOWN

Used for predicting the next encounter's approximate service region.

---

## Encounter-level WHEN stream

### gap_ids

Time gap category since the previous clinical encounter for the same patient.

After the encounter-only patch, gap_ids are recomputed between encounters only.

Possible labels:

- START
- 0D
- 1_7D
- 8_30D
- 31_90D
- 91_180D
- 181_365D
- 365PLUS
- UNKNOWN

Used for predicting when the next encounter is likely to happen.

---

## Rolling SDOH latest-status streams

SDOH response events are not sequence steps in the final artifact. Instead, each SDOH domain becomes a rolling latest-status stream.

Example fields:

- sdoh_transportation_needs_latest_status_ids
- sdoh_food_insecurity_latest_status_ids
- sdoh_housing_stability_latest_status_ids
- sdoh_financial_resource_strain_latest_status_ids
- sdoh_utilities_latest_status_ids
- sdoh_stress_latest_status_ids
- sdoh_depression_latest_status_ids
- sdoh_social_connections_latest_status_ids
- sdoh_physical_activity_latest_status_ids
- sdoh_alcohol_use_latest_status_ids
- sdoh_intimate_partner_violence_latest_status_ids

Each SDOH status stream is an encounter-level list aligned with the encounter sequence.

The value at position i means:

latest known status for that SDOH domain at or before encounter i

The status mapping is:

- 0 = UNKNOWN_NOT_YET_MEASURED
- 1 = NEGATIVE_NO_NEED
- 2 = POSITIVE_NEED_OR_RISK
- 3 = OTHER_DECLINED_UNABLE_UNSPECIFIED

Example:

sdoh_transportation_needs_latest_status_ids = [0, 2, 2, 2]

means:

- encounter 1: transportation status not known yet
- encounter 2: transportation need/risk observed
- encounter 3: latest known status remains positive
- encounter 4: latest known status remains positive

Example:

sdoh_food_insecurity_latest_status_ids = [0, 0, 0, 2]

means:

- encounters 1-3: food insecurity not known yet
- encounter 4: food insecurity need/risk observed

This design avoids future leakage because the value at each encounter uses only information known at or before that encounter.

---

## Static patient context

### patient_context_ids

patient_context_ids contains one categorical ID per patient-level context field.

Example:

{
"PatientBirthYearBin": 12,
"SexAssignedAtBirth": 2,
"OmbRace": 4,
"OmbEthnicity": 1,
"SmokingStatus": 3,
"MaritalStatus": 2
}

These are static patient-level conditioning features, not event-level lists.

Typical fields may include:

- PatientBirthYearBin
- SexAssignedAtBirth
- OmbRace
- OmbEthnicity
- MaritalStatus
- SmokingStatus
- VitalStatus
- MyChartStatus
- SexualOrientation
- patient_geography_known_flag
- patient_block_population_bin

Some SDOH summary fields may also exist in earlier artifacts, but the preferred final SDOH representation is the rolling latest-status streams.

### patient_context_values

patient_context_values contains numeric or raw patient-level geographic context.

Example:

{
"patient_lat": 39.05,
"patient_lon": -95.67,
"patient_population": 1320.0,
"CensusBlockGroupFipsCode": "201770045001"
}

Important interpretation:

- patient_lat is the latitude of the centroid of the patient's Census block group, not exact home latitude.
- patient_lon is the longitude of the centroid of the patient's Census block group, not exact home longitude.
- patient_population is the Census population count of the patient's home Census block group.
- CensusBlockGroupFipsCode is the patient's Census block group identifier. It is stored as a string, not as a numeric continuous feature.

## Model target interpretation

The model should be autoregressive over encounter positions.

For each patient, at each timestep, the model uses previous encounters and context to predict the next encounter's:

WHAT:

- type_ids
- event_description_ids
- group_code_ids
- diagnosis_value_ids

WHERE:

- setting_ids
- dept_type_ids
- dept_specialty_ids
- facility_size_ids
- region_ids

WHEN:

- gap_ids

CONTEXT:

- rolling SDOH latest-status streams
- static patient_context_ids
- static patient_context_values

The model should no longer rely on event_token_ids, because the final artifact removes the grand composite event token.

---

## Recommended modeling input

Use:

patient_sequences_encounter_only_with_sdoh_status_and_fips.pt

The model should include heads for:

- type
- event description
- group code
- diagnosis value
- gap
- setting
- department type
- department specialty
- department volume / facility size proxy
- region

The model should also accept:

- rolling SDOH latest-status streams as encounter-level covariates
- patient_context_ids as static categorical covariates
- patient_context_values as static numeric/raw geographic covariates

This matches the project goal:

Predict the next clinical encounter's WHAT, WHEN, and WHERE so resources and providers can be planned ahead.
