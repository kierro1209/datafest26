# Schema-to-RL mapping

This document ties **DataFest encounter-side schema** (see `docs/data_structure_design.md` and `docs/relation_model.md`) to a **reinforcement learning** formulation for **scheduling / resource allocation**. It also defines how the **upstream GPT-style module’s 3-D output** (planner inputs) fits into that MDP.

---

## 1. Assumed GPT output: 3-D planner input vector

The sequence model is assumed to emit **one 3-vector per predicted future contact** (or per decision tick), interpreted as:

| Index | Name | Type | Meaning |
|------:|------|------|---------|
| 0 | `token_diagnosis_id` | non-negative integer | Discrete **clinical need / primary diagnosis** token (vocabulary index). Joins conceptually to `diagnosis.DiagnosisKey` or a derived token id; in synthetic data it is an unconstrained index. |
| 1 | `arrival_fraction_day` | float in **[0, 1)** | **When** the patient is expected to need service: fractional offset within a **planning day** (0 = start of horizon, 1 wraps to next day in env logic). Maps from predicted clock time without storing raw PHI-style timestamps in the RL artifact. |
| 2 | `routing_hint_id` | non-negative integer | **Where / how intense** care is: index into a small set of **resource archetypes** aligned with `departments.DepartmentType` and encounter flags (ED vs inpatient vs outpatient). See `mapping.ROUTING_HINT_BUCKETS`. |

**Not in the CSVs:** hard capacities (beds, staff counts). The RL environment should take those as **config** (constants or scenario files), optionally calibrated from historical occupancy, not as columns from `encounters.csv`.

---

## 2. Raw schema → environment state (hospital side)

State features are anything the scheduler **knows when allocating** in simulation. Suggested mapping from tables:

| Signal | Source columns | Role in RL |
|--------|----------------|------------|
| Current time / horizon position | Simulated clock; optional anchor from real `Date` / `AdmissionInstant` distribution | Drives arrivals of GPT-scored patients; step boundaries. |
| Per-unit occupancy or queue length | Derived from **rolling** join of active encounters: same `DepartmentKey` (or `DepartmentType`), `AdmissionInstant` ≤ now < `DischargeInstant` | **Constraint** checks and state for queueing. |
| Department capacity slack | **Not in data** — env parameter `capacity_by_department_type` or `capacity_by_DepartmentKey` | Upper bound on concurrent patients per bucket. |
| Staffing slack | **Optional proxy:** count distinct `ProviderDurableKey` (or `AttendingProviderDurableKey`) per day per `PrimaryDepartment` / specialty from `providers.csv` + `encounters.csv` | Noisy capacity proxy; treat as soft constraint or ignore in v1. |
| Scheduled GPT patients not yet seen | Buffer of pending 3-vectors sorted by `arrival_fraction_day` | Demand pipeline fed by synthetic or live GPT outputs. |

**Encounter-derived service time:** If `DischargeInstant` and `AdmissionInstant` are populated, duration informs **service-time distribution** by `DepartmentKey` or `DepartmentType` for the simulator (offline fit), not a single-column “constraint.”

---

## 3. Raw schema → actions

Actions depend on your chosen scheduling granularity. Typical mappings:

| Action concept | Example encoding | Schema anchor |
|----------------|------------------|---------------|
| Assign to department bucket | Discrete `0 … num_units-1` | `departments.DepartmentKey` grouped by `DepartmentType` / site. |
| Assign slot / start time | Discrete slot index within day | Uses same time discretization as `arrival_fraction_day`. |
| Reject / overflow / redirect | Extra action or masked illegal moves | Penalize in reward when occupancy would exceed capacity. |

**Action mask:** Illegal if target unit at capacity or if patient `routing_hint_id` is incompatible with unit (domain rule you define, e.g. ICU-only hints).

---

## 4. Rewards and costs (schema-agnostic, clinically motivated)

| Component | Suggested definition | Data used for calibration |
|-----------|----------------------|---------------------------|
| Wait time | Simulated queue delay until service starts | LOS / inter-arrival from encounters |
| Overflow / diversion penalty | Large penalty if assignment exceeds capacity | Env params only |
| Lateness vs predicted need | Penalty if service starts far from GPT `arrival_fraction_day` | GPT vector dim 1 |
| Wrong-site penalty | If assignment contradicts `routing_hint_id` | GPT vector dim 2 |

---

## 5. Join path: one GPT 3-vector → patient episode stub

For training with **only** synthetic GPT vectors, no join is required.

When combining with real data later:

1. **Dim 0** ↔ `encounters.PrimaryDiagnosisKey` or hashed `DiagnosisValue` token id (after vocabulary build).
2. **Dim 1** ↔ discretized `AdmitHour`/`AdmitMinute` or delta from a reference midnight (normalized to [0,1)).
3. **Dim 2** ↔ `departments.DepartmentType` (via `encounters.DepartmentKey` → `departments`) collapsed to `ROUTING_HINT_BUCKETS` index in `mapping.py`.

---

## 6. File layout in `rl/`

| Path | Purpose |
|------|---------|
| `schema_to_rl_mapping.md` | This document. |
| `mapping.py` | Canonical names, bucket list, bounds, helpers for decoding GPT dims. |
| `generate_synthetic_gpt_vectors.py` | CLI to write CSV of 3-D rows for offline RL training. |
| `synthetic/` | Default output directory for generated CSVs (regenerable). |
| `schedule_env.py` | Gymnasium MDP: assign doctor + start slot per patient; masks illegal overlaps. |
| `schedule_policy.py` | MLP policy over `doctor × slot` actions with legal-action masking. |
| `train_doctor_scheduler.py` | REINFORCE training + CSV export (`rl/output/schedule.csv`). |
| `output/` | Exported day schedules (regenerable). |

---

## 7. Doctor-day scheduler (implemented)

**Input:** CSV rows with `token_diagnosis_id`, `arrival_fraction_day`, `routing_hint_id` (first *N* rows after sorting by arrival = one synthetic day).

**Decision:** For each patient in arrival order, pick a **doctor** (from a small default pool with eligibility by `routing_hint_id`) and a **start slot** so that:

- `[start_slot, end_slot)` does not intersect any prior assignment on the **same doctor** (no double-booking).
- Optional **department concurrency**: per `routing_hint_id`, at most `DEFAULT_DEPT_CONCURRENT_CAP[h]` overlapping patients system-wide (shared service-line pressure).

**Output CSV columns:** `schedule_date`, patient + GPT fields, `doctor_id`, `doctor_label`, `department_assigned`, `resource_line` (department + doctor id), slot indices, `start_time_hhmm` / `end_time_hhmm` (from `--slot-minutes`, default 30), plus **`episode_total_reward`** (same sum of step rewards as the RL trainer; repeated on each row for convenience).

**Training:** REINFORCE on masked categorical actions (`train_doctor_scheduler.py`). `--greedy-only` exports a minimum-wait feasible schedule without learning.

---

## 8. Version note

**v0** uses fixed routing bucket ordering in code. If you add new `DepartmentType` labels from EDA, update `ROUTING_HINT_BUCKETS` and bump a `MAPPING_VERSION` constant in `mapping.py` so old policies and new data stay comparable.

Default doctors and department caps in `schedule_env.py` are **synthetic**; swap in `providers.csv`-driven pools when you define a join from `PrimarySpecialty` / `PrimaryDepartment` to `routing_hint_id`.
