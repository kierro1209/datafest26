# GPT Sequence Model Output Shape

## Purpose

This document describes the expected model output shape after training is complete for the next-encounter GPT-style sequence model.

The model is trained to predict the next clinical encounter's:

- WHAT
- WHEN
- WHERE

as defined in `tokenization.md`.

---

## Prediction task

For each patient sequence, the model consumes the prior encounter history and produces logits for the next encounter attributes at each timestep.

The outputs are multi-head predictions over aligned encounter positions.

Conceptually:

```text
input history up to timestep t -> predict encounter attributes at timestep t+1
```

---

## Output tensor structure

Let:

- `B` = batch size
- `T` = sequence length
- `D_type` = number of type classes
- `D_event_description` = number of event description classes
- `D_group_code` = number of group code classes
- `D_diagnosis_value` = number of diagnosis value classes
- `D_gap` = number of gap classes
- `D_setting` = number of setting classes
- `D_dept_type` = number of department type classes
- `D_dept_specialty` = number of department specialty classes
- `D_facility_size` = number of facility size classes
- `D_region` = number of region classes

After training, a forward pass returns a dictionary of logits with shape:

```text
{
  "type":               [B, T, D_type],
  "event_description":  [B, T, D_event_description],
  "group_code":         [B, T, D_group_code],
  "diagnosis_value":    [B, T, D_diagnosis_value],
  "gap":                [B, T, D_gap],
  "setting":            [B, T, D_setting],
  "dept_type":          [B, T, D_dept_type],
  "dept_specialty":     [B, T, D_dept_specialty],
  "facility_size":      [B, T, D_facility_size],
  "region":             [B, T, D_region]
}
```

These are raw logits, not decoded labels.

---

## Output heads by planning objective

### WHAT heads

These outputs describe what kind of encounter is expected next.

- `type`
  - broad encounter class
  - shape: `[B, T, D_type]`

- `event_description`
  - specific visit subtype such as follow-up, surgery, check-up, procedure visit, emergency visit, etc.
  - shape: `[B, T, D_event_description]`
  - this is the most specific head for predicting the next visit type in the way a planner would usually ask the question

- `group_code`
  - broad diagnosis grouping
  - shape: `[B, T, D_group_code]`

- `diagnosis_value`
  - detailed diagnosis value
  - shape: `[B, T, D_diagnosis_value]`

### WHEN head

This output predicts when the next encounter is likely to happen.

- `gap`
  - predicts the next encounter gap bucket
  - shape: `[B, T, D_gap]`

Typical decoded labels include buckets such as:

- `START`
- `0D`
- `1_7D`
- `8_30D`
- `31_90D`
- `91_180D`
- `181_365D`
- `365PLUS`
- `UNKNOWN`

### WHERE heads

These outputs describe where the next encounter is likely to occur.

- `setting`
  - care setting
  - shape: `[B, T, D_setting]`

- `dept_type`
  - department type
  - shape: `[B, T, D_dept_type]`

- `dept_specialty`
  - department specialty
  - shape: `[B, T, D_dept_specialty]`

- `facility_size`
  - department volume / facility size proxy
  - shape: `[B, T, D_facility_size]`

- `region`
  - approximate service geography
  - shape: `[B, T, D_region]`

---

## Per-patient interpretation

For one patient with sequence length `T`, each timestep position returns one prediction for each output head.

For example, position `t` produces:

```text
(type[t], event_description[t], group_code[t], diagnosis_value[t],
 gap[t], setting[t], dept_type[t], dept_specialty[t], facility_size[t], region[t])
```

which represents the model's predicted next encounter attributes after observing the patient's history up to that point.

---

## Decoded output shape after inference

After applying `argmax` or another decoding strategy to each head, the decoded output for one patient timestep is conceptually:

```text
{
  "what": {
    "type": <predicted type label or id>,
    "event_description": <predicted event description label or id>,
    "group_code": <predicted group code label or id>,
    "diagnosis_value": <predicted diagnosis value label or id>
  },
  "when": {
    "gap": <predicted gap label or id>
  },
  "where": {
    "setting": <predicted setting label or id>,
    "dept_type": <predicted department type label or id>,
    "dept_specialty": <predicted department specialty label or id>,
    "facility_size": <predicted facility size label or id>,
    "region": <predicted region label or id>
  }
}
```

For a full batch, the decoded output is a sequence of these predictions across all timesteps and patients.

---

## Most important head for visit-type forecasting

If the main question is:

```text
What will the next event be?
```

in the sense of:

- check-up
- surgery
- follow-up
- procedure visit
- emergency visit

then the most important output head is:

- `event_description`

The other WHAT heads provide supporting structure:

- `type` gives a broader visit category
- `group_code` and `diagnosis_value` capture disease context

---

## Training artifact vs model output

The training input artifact stores aligned patient histories like:

- `type_ids`
- `event_description_ids`
- `group_code_ids`
- `diagnosis_value_ids`
- `setting_ids`
- `dept_type_ids`
- `dept_specialty_ids`
- `facility_size_ids`
- `region_ids`
- `gap_ids`
- rolling SDOH status streams
- static patient context

The trained model output is not another patient sequence object.

Instead, it is a dictionary of prediction logits over the next-token classes for each target head.

---

## Final summary

After GPT-style training completes, the model output shape is a multi-head logits dictionary:

```text
{
  "type":               [B, T, D_type],
  "event_description":  [B, T, D_event_description],
  "group_code":         [B, T, D_group_code],
  "diagnosis_value":    [B, T, D_diagnosis_value],
  "gap":                [B, T, D_gap],
  "setting":            [B, T, D_setting],
  "dept_type":          [B, T, D_dept_type],
  "dept_specialty":     [B, T, D_dept_specialty],
  "facility_size":      [B, T, D_facility_size],
  "region":             [B, T, D_region]
}
```

This output jointly represents the model's predicted WHAT, WHEN, and WHERE for the next clinical encounter.
