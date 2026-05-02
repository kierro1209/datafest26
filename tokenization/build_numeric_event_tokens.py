#!/usr/bin/env python3
"""
Numeric event tokenizer v2 for WHAT / WHEN / WHERE forecasting.

Input:
  data/processed/event_enriched.csv.gz

Outputs:
  data/processed/token_vocabulary_v2.csv
  data/processed/event_tokenized_v2.csv.gz
  data/processed/patient_tokenized_sequence_v2.csv.gz
  data/processed/next_event_training_examples_v2.csv.gz
  docs/tokenization_feature_spec_v2.md

Design:
  - event-level, not encounter-level
  - one event_enriched row -> one composite event token string -> one integer event_token_id
  - token pieces are organized around:
      WHAT: event type + diagnosis + visit description
      WHEN: patient-level previous-event gap + calendar timing
      WHERE: care setting + department + provider role + geography/proximity/volume proxies
  - preserves interpretability by saving atomic token strings for each event
  - creates next-event targets for model training: next token, next gap, next department/provider/setting/diagnosis fields

Progress bars use tqdm when installed (`pip install tqdm`); otherwise staging/pass prints behave as before.

Resume without redoing long steps (same --work-db): `--skip-stage` reuses `event_stage`; `--skip-event-work` reuses `event_work`.

Important limitation:
  event_enriched contains patient home geography centroid (CENTLAT/CENTLON) and department census tract/city/county,
  but not department latitude/longitude. Therefore, this script does NOT compute true distance.
  It creates a safe proximity token using CensusBlockGroupFipsCode startswith(department_CensusTract).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Callable

import pandas as pd

try:
    from tqdm import tqdm as _tqdm_fn
except ImportError:
    _tqdm_fn = None

CSV_ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

# Set False via --no-progress in main()
SHOW_PROGRESS = True


class _NoProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, desc: str, **kwargs) -> None:
        pass


def _progress_bar(*, total: int | None, desc: str, unit: str = "row"):
    if not SHOW_PROGRESS or _tqdm_fn is None:
        return _NoProgressBar()
    kwargs: dict = {"desc": desc, "unit": unit}
    if total is not None:
        kwargs["total"] = total
    return _tqdm_fn(**kwargs)

TOKEN_VERSION = "v2_what_when_where_event_composite_numeric"

SPECIAL_TOKENS = [
    (0, "[PAD]", "SPECIAL", "special", "padding"),
    (1, "[PATIENT_START]", "SPECIAL", "special", "patient_start"),
    (2, "[PATIENT_END]", "SPECIAL", "special", "patient_end"),
    (3, "[UNK]", "SPECIAL", "special", "unknown"),
    (4, "[MISSING]", "SPECIAL", "special", "missing"),
]

CARE_SETTING_FLAGS = [
    "IsEdVisit",
    "IsHospitalAdmission",
    "IsHospitalOutpatientVisit",
    "IsInpatientAdmission",
    "IsObservation",
    "IsOutpatientFaceToFaceVisit",
]

# Exact columns expected from the user's current event_enriched.csv.gz when present.
IDENTITY_TIME_COLUMNS = [
    "event_id", "event_source", "event_grain", "event_EncounterKey", "event_PatientDurableKey",
    "event_date", "event_time", "event_index_within_encounter",
    "EncounterKey", "PatientDurableKey", "Date", "AdmissionInstant",
    "AdmitYear", "AdmitMonth", "AdmitDay", "AdmitHour", "AdmitMinute",
    "DischargeInstant", "DischargeYear", "DischargeMonth", "DischargeDay", "DischargeHour", "DischargeMinute",
]

PATIENT_CONTEXT_COLUMNS = [
    "PatientBirthYearBin", "SexAssignedAtBirth", "FirstRace", "OmbRace", "OmbEthnicity",
    "MaritalStatus", "SmokingStatus", "VitalStatus", "MyChartStatus", "SexualOrientation",
    "CensusBlockGroupFipsCode", "patient_geography_known_flag", "patient_birth_year_bin_missing_class",
    "mychart_status_missing_class", "smoking_status_missing_class",
    "GEOID", "PopulationValue", "CENTLAT", "CENTLON", "patient_block_population_bin",
    "sdoh_any_observed", "sdoh_num_questions_answered", "sdoh_num_domains_answered",
    "sdoh_AlcoholUse_observed", "sdoh_Depression_observed", "sdoh_FinancialResourceStrain_observed",
    "sdoh_FoodInsecurity_observed", "sdoh_HousingStability_observed", "sdoh_IntimatePartnerViolence_observed",
    "sdoh_PhysicalActivity_observed", "sdoh_SocialConnections_observed", "sdoh_Stress_observed",
    "sdoh_TransportationNeeds_observed", "sdoh_Utilities_observed",
    "sdoh_domain_answer_tokens", "sdoh_domains_observed",
]

WHAT_COLUMNS = [
    "event_type", "event_subtype", "event_description", "event_domain", "event_value",
    "event_display_name", "event_answer_text", "event_sdoh_answer_token",
    "Type", "VisitType", "VisitTypeDescription", "AdmissionSource", "AdmissionType",
    "PrimaryDiagnosisKey", "DiagnosisKey", "DiagnosisValue", "DiagnosisName", "GroupCode", "GroupName",
    "primary_diagnosis_missing_class",
]

WHERE_COLUMNS = [
    "DepartmentKey", "DepartmentName", "DepartmentSpecialty", "DepartmentType",
    "department_Address", "department_City", "department_County", "department_PostalCode", "department_CensusTract",
    "City", "County", "PostalCode", "CensusTract",
    "department_key_missing_class",
    "ProviderDurableKey", "AttendingProviderDurableKey", "DischargeProviderDurableKey",
    "provider_key_missing_class", "attending_provider_key_missing_class", "discharge_provider_key_missing_class",
    "provider_ClinicianTitle", "provider_PrimarySpecialty", "provider_Type", "provider_PrimaryDepartment",
    "provider_OfficeCity", "provider_OfficePostalCode",
    "attending_provider_ClinicianTitle", "attending_provider_PrimarySpecialty", "attending_provider_Type",
    "attending_provider_PrimaryDepartment", "attending_provider_OfficeCity", "attending_provider_OfficePostalCode",
    "discharge_provider_ClinicianTitle", "discharge_provider_PrimarySpecialty", "discharge_provider_Type",
    "discharge_provider_PrimaryDepartment", "discharge_provider_OfficeCity", "discharge_provider_OfficePostalCode",
    *CARE_SETTING_FLAGS,
]

ALL_SOURCE_COLUMNS = list(dict.fromkeys(IDENTITY_TIME_COLUMNS + PATIENT_CONTEXT_COLUMNS + WHAT_COLUMNS + WHERE_COLUMNS))

STAGE_COLUMNS = [
    "event_id", "PatientDurableKey", "EncounterKey", "event_date", "event_time", "event_index_within_encounter",
    "event_source", "event_grain",
    *PATIENT_CONTEXT_COLUMNS, *WHAT_COLUMNS, *WHERE_COLUMNS,
]


def choose_encoding(path: Path) -> str:
    for enc in CSV_ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, nrows=0, encoding=enc)
            return enc
        except Exception:
            continue
    return "latin1"


def qident(x: str) -> str:
    return '"' + x.replace('"', '""') + '"'


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _ensure_event_stage_ready(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "event_stage"):
        raise SystemExit(
            "work DB has no table `event_stage`. Run without --skip-stage first to load from --input, "
            "or point --work-db at a database that already contains a populated event_stage."
        )
    n = conn.execute("SELECT COUNT(*) FROM event_stage").fetchone()[0]
    if n == 0:
        raise SystemExit("event_stage exists but is empty; cannot use --skip-stage.")


def _ensure_event_work_ready(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "event_work"):
        raise SystemExit(
            "work DB has no table `event_work`. Run without --skip-event-work first to build it, "
            "or point --work-db at a database that already contains event_work."
        )
    n = conn.execute("SELECT COUNT(*) FROM event_work").fetchone()[0]
    if n == 0:
        raise SystemExit("event_work exists but is empty; cannot use --skip-event-work.")
    if _table_exists(conn, "event_stage"):
        n_stage = conn.execute("SELECT COUNT(*) FROM event_stage").fetchone()[0]
        if n_stage != n:
            print(
                f"[WARN] event_stage={n_stage:,} rows vs event_work={n:,} rows — "
                "if unexpected, rebuild without --skip-event-work.",
                flush=True,
            )


def missing_columns_from_input_header(input_path: Path) -> list[str]:
    """Same `missing` list as load_event_stage, without re-reading the full CSV."""
    enc = choose_encoding(input_path)
    header = list(pd.read_csv(input_path, nrows=0, encoding=enc).columns)
    return [c for c in ALL_SOURCE_COLUMNS if c not in header]


def normalize_value(value: object, max_len: int = 80) -> str:
    if value is None:
        return "MISSING"
    try:
        if pd.isna(value):
            return "MISSING"
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return "MISSING"
    missing_map = {
        "*UNSPECIFIED": "ASKED_NOT_ANSWERED_OR_UNABLE",
        "*UNKNOWN": "ASKED_NOT_ANSWERED_OR_UNABLE",
        "*NOT APPLICABLE": "ASKED_NOT_ANSWERED_OR_UNABLE",
        "UNSPECIFIED": "NOT_RECORDED_OR_UNKNOWN",
        "UNKNOWN": "NOT_RECORDED_OR_UNKNOWN",
        "NOT APPLICABLE": "STRUCTURAL_NOT_APPLICABLE",
    }
    if s.upper() in missing_map:
        return missing_map[s.upper()]
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "MISSING"


def trueish(value: object) -> bool:
    return normalize_value(value) in {"1", "TRUE", "T", "YES", "Y"}


def parse_float(value: object) -> float | None:
    try:
        s = str(value).replace(",", "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def bin_count(value: object) -> str:
    x = parse_float(value)
    if x is None:
        return "MISSING"
    n = int(x)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3PLUS"


def bin_population(value: object) -> str:
    x = parse_float(value)
    if x is None or x <= 0:
        return "MISSING"
    if x < 500:
        return "LT500"
    if x < 1500:
        return "500_1499"
    if x < 3000:
        return "1500_2999"
    return "3000PLUS"


def bin_hour(value: object) -> str:
    try:
        h = int(float(str(value).strip()))
    except Exception:
        return "MISSING"
    if 0 <= h <= 5:
        return "NIGHT"
    if 6 <= h <= 11:
        return "MORNING"
    if 12 <= h <= 17:
        return "AFTERNOON"
    if 18 <= h <= 23:
        return "EVENING"
    return "MISSING"


def bin_month(value: object) -> str:
    try:
        m = int(float(str(value).strip()))
    except Exception:
        return "MISSING"
    if 1 <= m <= 3:
        return "Q1"
    if 4 <= m <= 6:
        return "Q2"
    if 7 <= m <= 9:
        return "Q3"
    if 10 <= m <= 12:
        return "Q4"
    return "MISSING"


def event_month_bin(row: pd.Series) -> str:
    """Return calendar-quarter token from AdmitMonth when present, else event_date.

    This intentionally does not require restaging. Older event_stage tables may not
    contain AdmitMonth, but they should contain event_date from the staged input.
    """
    m = row.get("AdmitMonth")
    if m is not None and str(m).strip():
        out = bin_month(m)
        if out != "MISSING":
            return out

    dt = pd.to_datetime(row.get("event_date"), errors="coerce")
    if pd.isna(dt):
        return "MISSING"
    return bin_month(dt.month)


def event_hour_bin(row: pd.Series) -> str:
    """Return time-of-day token from AdmitHour when present, else event_time.

    This intentionally does not require restaging. Older event_stage tables may not
    contain AdmitHour, but they should contain event_time when the source had time data.
    """
    h = row.get("AdmitHour")
    if h is not None and str(h).strip():
        out = bin_hour(h)
        if out != "MISSING":
            return out

    dt = pd.to_datetime(row.get("event_time"), errors="coerce")
    if pd.isna(dt):
        return "MISSING"
    return bin_hour(dt.hour)


def first_present(row: pd.Series, names: list[str], default: object = "") -> object:
    for name in names:
        if name in row.index:
            val = row.get(name)
            if val is not None and str(val).strip() != "":
                return val
    return default


def setting_value_from_row(row: pd.Series) -> str:
    labels = []
    mapping = {
        "IsEdVisit": "ED",
        "IsHospitalAdmission": "HOSP_ADMIT",
        "IsHospitalOutpatientVisit": "HOSP_OP",
        "IsInpatientAdmission": "INPATIENT",
        "IsObservation": "OBS",
        "IsOutpatientFaceToFaceVisit": "OP_FACE",
    }
    for col, lab in mapping.items():
        if trueish(row.get(col)):
            labels.append(lab)
    return "__".join(labels) if labels else "NONE"


def department_tract(row: pd.Series) -> str:
    return str(first_present(row, ["department_CensusTract", "CensusTract"], "")).strip()


def department_county(row: pd.Series) -> str:
    return str(first_present(row, ["department_County", "County"], "")).strip()


def department_city(row: pd.Series) -> str:
    return str(first_present(row, ["department_City", "City"], "")).strip()


def department_postal(row: pd.Series) -> str:
    return str(first_present(row, ["department_PostalCode", "PostalCode"], "")).strip()


def proximity_to_department_tract(row: pd.Series) -> str:
    patient_bg = normalize_value(row.get("CensusBlockGroupFipsCode"), max_len=30)
    dept_tract_raw = department_tract(row)
    dept_tract = normalize_value(dept_tract_raw, max_len=30)
    if patient_bg == "MISSING" or dept_tract == "MISSING":
        return "UNKNOWN"
    # Census block group GEOID normally begins with the 11-digit census tract GEOID.
    if patient_bg.startswith(dept_tract):
        return "SAME_TRACT"
    return "DIFFERENT_TRACT"


def volume_bin(n: object) -> str:
    x = parse_float(n)
    if x is None:
        return "MISSING"
    if x < 100:
        return "VERY_LOW"
    if x < 1000:
        return "LOW"
    if x < 10000:
        return "MID"
    if x < 100000:
        return "HIGH"
    return "VERY_HIGH"


def volume_ord(label: object) -> int:
    return {"MISSING": 0, "VERY_LOW": 1, "LOW": 2, "MID": 3, "HIGH": 4, "VERY_HIGH": 5}.get(str(label), 0)


def transfer_pattern(row: pd.Series) -> str:
    idx = parse_float(row.get("patient_event_index"))
    if idx == 1:
        return "START"
    cur_dept = normalize_value(row.get("DepartmentKey"), max_len=60)
    prev_dept = normalize_value(row.get("previous_DepartmentKey"), max_len=60)
    if cur_dept != "MISSING" and prev_dept != "MISSING" and cur_dept == prev_dept:
        return "SAME_DEPARTMENT"
    cur_v = volume_ord(row.get("department_volume_bin"))
    prev_v = volume_ord(row.get("previous_department_volume_bin"))
    cur_setting = setting_value_from_row(row)
    prev_setting = normalize_value(row.get("previous_setting_value"), max_len=80)
    if prev_setting in {"ED", "ED__OP_FACE", "ED__HOSP_ADMIT", "ED__INPATIENT"} and cur_v > prev_v:
        return "ED_TO_LARGER_FACILITY"
    if cur_v > prev_v:
        return "LOCAL_TO_LARGER"
    if cur_v < prev_v and prev_v > 0:
        return "LARGER_TO_LOCAL"
    if cur_setting == prev_setting and cur_setting != "NONE":
        return "SAME_SETTING"
    return "OTHER"


def standardize_chunk(df: pd.DataFrame, row_offset: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    def col_or_blank(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    out["event_id"] = col_or_blank("event_id") if "event_id" in df.columns else pd.Series([f"ROW_EVENT_{row_offset+i}" for i in range(len(df))], index=df.index)
    out["PatientDurableKey"] = col_or_blank("event_PatientDurableKey")
    fallback_patient = col_or_blank("PatientDurableKey")
    out["PatientDurableKey"] = out["PatientDurableKey"].where(out["PatientDurableKey"].str.len() > 0, fallback_patient)
    out["EncounterKey"] = col_or_blank("event_EncounterKey")
    fallback_enc = col_or_blank("EncounterKey")
    out["EncounterKey"] = out["EncounterKey"].where(out["EncounterKey"].str.len() > 0, fallback_enc)
    out["event_date"] = col_or_blank("event_date")
    fallback_date = col_or_blank("Date")
    out["event_date"] = out["event_date"].where(out["event_date"].str.len() > 0, fallback_date)
    out["event_time"] = col_or_blank("event_time")
    fallback_time = col_or_blank("AdmissionInstant")
    out["event_time"] = out["event_time"].where(out["event_time"].str.len() > 0, fallback_time)
    out["event_index_within_encounter"] = col_or_blank("event_index_within_encounter")
    out["event_source"] = col_or_blank("event_source").where(col_or_blank("event_source").str.len() > 0, "UNKNOWN_EVENT_SOURCE")
    out["event_grain"] = col_or_blank("event_grain").where(col_or_blank("event_grain").str.len() > 0, "EVENT")

    for c in PATIENT_CONTEXT_COLUMNS + WHAT_COLUMNS + WHERE_COLUMNS:
        if c not in out.columns:
            out[c] = col_or_blank(c)
    return out[STAGE_COLUMNS]


def load_event_stage(conn: sqlite3.Connection, input_path: Path, chunksize: int) -> list[str]:
    enc = choose_encoding(input_path)
    header = list(pd.read_csv(input_path, nrows=0, encoding=enc).columns)
    usecols = [c for c in ALL_SOURCE_COLUMNS if c in header]
    missing = [c for c in ALL_SOURCE_COLUMNS if c not in header]
    print(f"[TOKEN] input={input_path}")
    print(f"[TOKEN] encoding={enc}")
    print(f"[TOKEN] using_source_columns={len(usecols)} missing_optional_columns={len(missing)}")

    conn.execute("DROP TABLE IF EXISTS event_stage")
    conn.commit()
    offset = 0
    first = True
    pbar = _progress_bar(total=None, desc="Stage CSV → event_stage", unit="row")
    try:
        for chunk in pd.read_csv(
            input_path,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            chunksize=chunksize,
            encoding=enc,
        ):
            std = standardize_chunk(chunk, offset)
            std.to_sql("event_stage", conn, if_exists="replace" if first else "append", index=False)
            first = False
            n = len(std)
            offset += n
            pbar.update(n)
    finally:
        pbar.close()
    print(f"[TOKEN] staged rows: {offset:,}")
    if first:
        raise RuntimeError("No rows found in input.")
    conn.executescript("""
    CREATE INDEX IF NOT EXISTS idx_event_stage_patient_time
      ON event_stage(PatientDurableKey, event_date, event_time, EncounterKey, event_index_within_encounter, event_id);
    CREATE INDEX IF NOT EXISTS idx_event_stage_department
      ON event_stage(DepartmentKey);
    """)
    conn.commit()
    return missing


def build_event_work(conn: sqlite3.Connection) -> None:
    print("[TOKEN] building event_work with within-patient gaps, department volume bins, and next-event targets")
    print(
        "[TOKEN] running one large SQLite query over event_stage (often several minutes for ~10M+ rows; "
        "heartbeats print below while it runs)",
        flush=True,
    )
    _progress_calls = {"n": 0}
    _t0 = time.monotonic()

    def _sql_progress() -> int:
        """SQLite invokes this every N VM instructions; return non-zero to cancel."""
        _progress_calls["n"] += 1
        if SHOW_PROGRESS:
            elapsed = time.monotonic() - _t0
            print(
                f"[TOKEN] event_work SQL still running… {_progress_calls['n']} checkpoints, "
                f"{elapsed / 60.0:.1f} min elapsed",
                flush=True,
            )
        return 0

    instruction_interval = 8_000_000
    if SHOW_PROGRESS:
        conn.set_progress_handler(_sql_progress, instruction_interval)
    try:
        conn.executescript("""
    DROP TABLE IF EXISTS department_volume;
    CREATE TABLE department_volume AS
    SELECT
      COALESCE(NULLIF(TRIM(DepartmentKey), ''), 'MISSING') AS DepartmentKey_norm,
      COUNT(*) AS department_event_count,
      CASE
        WHEN COUNT(*) < 100 THEN 'VERY_LOW'
        WHEN COUNT(*) < 1000 THEN 'LOW'
        WHEN COUNT(*) < 10000 THEN 'MID'
        WHEN COUNT(*) < 100000 THEN 'HIGH'
        ELSE 'VERY_HIGH'
      END AS department_volume_bin
    FROM event_stage
    GROUP BY COALESCE(NULLIF(TRIM(DepartmentKey), ''), 'MISSING');

    DROP TABLE IF EXISTS event_work;
    CREATE TABLE event_work AS
    WITH base AS (
      SELECT
        s.*,
        COALESCE(v.department_event_count, 0) AS department_event_count,
        COALESCE(v.department_volume_bin, 'MISSING') AS department_volume_bin,
        CASE
          WHEN lower(trim(coalesce(s.IsEdVisit,''))) IN ('1','true','t','yes','y') THEN 'ED'
          WHEN lower(trim(coalesce(s.IsInpatientAdmission,''))) IN ('1','true','t','yes','y') THEN 'INPATIENT'
          WHEN lower(trim(coalesce(s.IsHospitalAdmission,''))) IN ('1','true','t','yes','y') THEN 'HOSP_ADMIT'
          WHEN lower(trim(coalesce(s.IsObservation,''))) IN ('1','true','t','yes','y') THEN 'OBS'
          WHEN lower(trim(coalesce(s.IsHospitalOutpatientVisit,''))) IN ('1','true','t','yes','y') THEN 'HOSP_OP'
          WHEN lower(trim(coalesce(s.IsOutpatientFaceToFaceVisit,''))) IN ('1','true','t','yes','y') THEN 'OP_FACE'
          ELSE 'NONE'
        END AS setting_value_simple
      FROM event_stage s
      LEFT JOIN department_volume v
        ON COALESCE(NULLIF(TRIM(s.DepartmentKey), ''), 'MISSING') = v.DepartmentKey_norm
    ), ordered AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS patient_event_index,
        LAG(event_date) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS previous_patient_event_date,
        LAG(DepartmentKey) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS previous_DepartmentKey,
        LAG(department_volume_bin) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS previous_department_volume_bin,
        LAG(setting_value_simple) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS previous_setting_value,
        LEAD(event_id) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS next_event_id,
        LEAD(event_date) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS next_event_date,
        LEAD(event_source) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_event_source,
        LEAD(Type) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_Type,
        LEAD(GroupCode) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_GroupCode,
        LEAD(DiagnosisValue) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_DiagnosisValue,
        LEAD(DepartmentType) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_DepartmentType,
        LEAD(DepartmentSpecialty) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_DepartmentSpecialty,
        LEAD(department_volume_bin) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_department_volume_bin,
        LEAD(provider_Type) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_provider_Type,
        LEAD(provider_PrimarySpecialty) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_provider_PrimarySpecialty,
        LEAD(setting_value_simple) OVER (
          PARTITION BY PatientDurableKey
          ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
        ) AS target_next_setting_value
      FROM base
    ), gaps AS (
      SELECT
        *,
        CASE
          WHEN previous_patient_event_date IS NULL THEN NULL
          WHEN julianday(event_date) IS NULL OR julianday(previous_patient_event_date) IS NULL THEN NULL
          ELSE CAST(julianday(event_date) - julianday(previous_patient_event_date) AS INTEGER)
        END AS days_since_previous_patient_event,
        CASE
          WHEN next_event_date IS NULL THEN NULL
          WHEN julianday(next_event_date) IS NULL OR julianday(event_date) IS NULL THEN NULL
          ELSE CAST(julianday(next_event_date) - julianday(event_date) AS INTEGER)
        END AS target_next_gap_days
      FROM ordered
    )
    SELECT
      *,
      CASE
        WHEN patient_event_index = 1 THEN 'START'
        WHEN days_since_previous_patient_event IS NULL THEN 'UNKNOWN'
        WHEN days_since_previous_patient_event = 0 THEN '0D'
        WHEN days_since_previous_patient_event BETWEEN 1 AND 7 THEN '1_7D'
        WHEN days_since_previous_patient_event BETWEEN 8 AND 30 THEN '8_30D'
        WHEN days_since_previous_patient_event BETWEEN 31 AND 90 THEN '31_90D'
        WHEN days_since_previous_patient_event BETWEEN 91 AND 180 THEN '91_180D'
        WHEN days_since_previous_patient_event BETWEEN 181 AND 365 THEN '181_365D'
        WHEN days_since_previous_patient_event > 365 THEN '365PLUS'
        ELSE 'UNKNOWN'
      END AS gap_bin,
      CASE
        WHEN target_next_gap_days IS NULL THEN NULL
        WHEN target_next_gap_days = 0 THEN '0D'
        WHEN target_next_gap_days BETWEEN 1 AND 7 THEN '1_7D'
        WHEN target_next_gap_days BETWEEN 8 AND 30 THEN '8_30D'
        WHEN target_next_gap_days BETWEEN 31 AND 90 THEN '31_90D'
        WHEN target_next_gap_days BETWEEN 91 AND 180 THEN '91_180D'
        WHEN target_next_gap_days BETWEEN 181 AND 365 THEN '181_365D'
        WHEN target_next_gap_days > 365 THEN '365PLUS'
        ELSE 'UNKNOWN'
      END AS target_next_gap_bin
    FROM gaps;

    CREATE INDEX IF NOT EXISTS idx_event_work_patient_idx ON event_work(PatientDurableKey, patient_event_index);
    CREATE INDEX IF NOT EXISTS idx_event_work_event_id ON event_work(event_id);
    """)
    finally:
        # Two-arg form required (Python 3.11+): None handler + n=0 clears the progress hook.
        conn.set_progress_handler(None, 0)
    conn.commit()
    print("[TOKEN] event_work SQL finished.", flush=True)


def iter_event_work(conn: sqlite3.Connection, chunksize: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_sql_query(
        "SELECT * FROM event_work ORDER BY PatientDurableKey, patient_event_index",
        conn,
        chunksize=chunksize,
    )


def make_atomic_tokens(row: pd.Series, include_high_cardinality: bool = False) -> list[str]:
    tokens: list[str] = []

    # Patient context
    tokens.append(f"PAT_AGEBIN_{normalize_value(row.get('PatientBirthYearBin'))}")
    tokens.append(f"PAT_SEX_{normalize_value(row.get('SexAssignedAtBirth'))}")
    tokens.append(f"PAT_RACE_{normalize_value(row.get('OmbRace'))}")
    tokens.append(f"PAT_ETH_{normalize_value(row.get('OmbEthnicity'))}")
    tokens.append(f"PAT_MARITAL_{normalize_value(row.get('MaritalStatus'))}")
    tokens.append(f"PAT_SMOKE_{normalize_value(row.get('SmokingStatus'))}")
    tokens.append(f"PAT_VITAL_{normalize_value(row.get('VitalStatus'))}")
    tokens.append(f"PAT_MYCHART_{normalize_value(row.get('MyChartStatus'))}")
    tokens.append(f"PAT_ORIENT_{normalize_value(row.get('SexualOrientation'))}")
    tokens.append(f"PAT_GEO_KNOWN_{'YES' if trueish(row.get('patient_geography_known_flag')) else 'NO'}")
    popbin = first_present(row, ["patient_block_population_bin"], "")
    tokens.append(f"PAT_HOME_POPBIN_{normalize_value(popbin) if str(popbin).strip() else bin_population(row.get('PopulationValue'))}")
    tokens.append(f"PAT_SDOH_OBS_{'YES' if trueish(row.get('sdoh_any_observed')) else 'NO'}")
    tokens.append(f"PAT_SDOH_QCOUNT_{bin_count(row.get('sdoh_num_questions_answered'))}")
    tokens.append(f"PAT_SDOH_DOMAINCOUNT_{bin_count(row.get('sdoh_num_domains_answered'))}")
    for d in re.split(r"[,|;]+", str(row.get("sdoh_domains_observed") or "")):
        dnorm = normalize_value(d)
        if dnorm != "MISSING":
            tokens.append(f"PAT_SDOH_DOMAIN_{dnorm}")

    # WHAT: source, encounter/event type, diagnosis
    tokens.append(f"EVT_SRC_{normalize_value(row.get('event_source'))}")
    tokens.append(f"EVT_GRAIN_{normalize_value(row.get('event_grain'))}")
    tokens.append(f"EVT_TYPE_{normalize_value(first_present(row, ['event_type', 'Type']))}")
    tokens.append(f"EVT_SUBTYPE_{normalize_value(first_present(row, ['event_subtype', 'VisitType']))}")
    tokens.append(f"EVT_DESC_{normalize_value(first_present(row, ['event_description', 'VisitTypeDescription']))}")
    tokens.append(f"EVT_ADMISSION_SOURCE_{normalize_value(row.get('AdmissionSource'))}")
    tokens.append(f"EVT_ADMISSION_TYPE_{normalize_value(row.get('AdmissionType'))}")
    tokens.append(f"EVT_DX_MISSING_CLASS_{normalize_value(row.get('primary_diagnosis_missing_class'))}")
    tokens.append(f"EVT_DXG_{normalize_value(row.get('GroupCode'))}")
    if include_high_cardinality:
        tokens.append(f"EVT_DX_{normalize_value(row.get('DiagnosisValue'))}")
    # SDOH response rows are event rows, so preserve their event-specific content at controlled granularity.
    tokens.append(f"EVT_DOMAIN_{normalize_value(row.get('event_domain'))}")
    if include_high_cardinality:
        tokens.append(f"EVT_VALUE_{normalize_value(row.get('event_value'))}")

    # WHEN: gap to prior event in same patient + calendar timing of current event.
    tokens.append(f"EVT_GAP_{normalize_value(row.get('gap_bin'))}")
    tokens.append(f"EVT_MONTH_{event_month_bin(row)}")
    tokens.append(f"EVT_HOURBIN_{event_hour_bin(row)}")

    # WHERE: setting, department, geography/proximity proxy, observed department volume, provider role metadata.
    tokens.append(f"EVT_SETTING_{setting_value_from_row(row)}")
    tokens.append(f"EVT_DEPT_TYPE_{normalize_value(row.get('DepartmentType'))}")
    tokens.append(f"EVT_DEPT_SPEC_{normalize_value(row.get('DepartmentSpecialty'))}")
    tokens.append(f"EVT_DEPT_KEY_MISSING_CLASS_{normalize_value(row.get('department_key_missing_class'))}")
    tokens.append(f"EVT_DEPT_COUNTY_{normalize_value(department_county(row))}")
    if include_high_cardinality:
        tokens.append(f"EVT_DEPT_CITY_{normalize_value(department_city(row))}")
        tokens.append(f"EVT_DEPT_POSTAL_{normalize_value(department_postal(row))}")
    tokens.append(f"EVT_DEPT_TRACT_PROX_{proximity_to_department_tract(row)}")
    tokens.append(f"EVT_DEPT_VOLUME_{normalize_value(row.get('department_volume_bin'))}")
    tokens.append(f"EVT_PROVIDER_TYPE_{normalize_value(row.get('provider_Type'))}")
    tokens.append(f"EVT_ATTENDING_TYPE_{normalize_value(row.get('attending_provider_Type'))}")
    tokens.append(f"EVT_DISCHARGE_TYPE_{normalize_value(row.get('discharge_provider_Type'))}")
    if include_high_cardinality:
        tokens.append(f"EVT_PROVIDER_SPEC_{normalize_value(row.get('provider_PrimarySpecialty'))}")
        tokens.append(f"EVT_ATTENDING_SPEC_{normalize_value(row.get('attending_provider_PrimarySpecialty'))}")
        tokens.append(f"EVT_DISCHARGE_SPEC_{normalize_value(row.get('discharge_provider_PrimarySpecialty'))}")
    tokens.append(f"EVT_TRANSFER_{transfer_pattern(row)}")

    seen = set()
    out = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def composite_token(row: pd.Series, include_high_cardinality: bool = False) -> str:
    return "EVENT_COMPOSITE::" + "|".join(make_atomic_tokens(row, include_high_cardinality=include_high_cardinality))


def build_vocabulary(conn: sqlite3.Connection, output_dir: Path, chunksize: int, min_event_freq: int, include_high_cardinality: bool) -> dict[str, int]:
    print("[TOKEN] pass 1/2: vocabulary")
    n_rows = conn.execute("SELECT COUNT(*) FROM event_work").fetchone()[0]
    event_counter: Counter[str] = Counter()
    atomic_counter: Counter[str] = Counter()
    pbar = _progress_bar(total=n_rows, desc="Vocabulary (pass 1/2)", unit="row")
    try:
        for chunk in iter_event_work(conn, chunksize):
            for _, row in chunk.iterrows():
                atomic = make_atomic_tokens(row, include_high_cardinality=include_high_cardinality)
                event_counter["EVENT_COMPOSITE::" + "|".join(atomic)] += 1
                atomic_counter.update(atomic)
            pbar.update(len(chunk))
    finally:
        pbar.close()

    rows = []
    token_to_id = {}
    for tid, ts, kind, field, val in SPECIAL_TOKENS:
        token_to_id[ts] = tid
        rows.append({"token_id": tid, "token_string": ts, "token_kind": kind, "field_name": field, "field_value": val, "frequency": None})
    next_id = 5
    for ts, freq in sorted(event_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_event_freq:
            continue
        token_to_id[ts] = next_id
        rows.append({"token_id": next_id, "token_string": ts, "token_kind": "EVENT_COMPOSITE", "field_name": "event_composite", "field_value": ts.replace("EVENT_COMPOSITE::", "", 1), "frequency": freq})
        next_id += 1
    for ts, freq in sorted(atomic_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if ts in token_to_id:
            continue
        token_to_id[ts] = next_id
        rows.append({"token_id": next_id, "token_string": ts, "token_kind": "ATOMIC_DEBUG", "field_name": ts.split("_", 2)[0], "field_value": ts, "frequency": freq})
        next_id += 1
    out = output_dir / "token_vocabulary_v2.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[TOKEN] wrote {out} with {len(rows):,} rows")
    return token_to_id


def build_event_tokenized(conn: sqlite3.Connection, output_dir: Path, token_to_id: dict[str, int], chunksize: int, include_high_cardinality: bool) -> None:
    print("[TOKEN] pass 2/2: encode events")
    out_path = output_dir / "event_tokenized_v2.csv.gz"
    train_path = output_dir / "next_event_training_examples_v2.csv.gz"
    for p in [out_path, train_path]:
        if p.exists():
            p.unlink()
    conn.execute("DROP TABLE IF EXISTS token_event_min")
    conn.execute("""
      CREATE TABLE token_event_min (
        PatientDurableKey TEXT,
        event_id TEXT,
        EncounterKey TEXT,
        event_date TEXT,
        patient_event_index INTEGER,
        event_token_id INTEGER,
        event_composite_token TEXT
      )
    """)
    conn.commit()

    first_events = True
    first_train = True
    total = 0
    n_rows = conn.execute("SELECT COUNT(*) FROM event_work").fetchone()[0]
    pbar = _progress_bar(total=n_rows, desc="Encode events (pass 2/2)", unit="row")
    try:
        for chunk in iter_event_work(conn, chunksize):
            event_rows = []
            min_rows = []
            train_rows = []
            for _, row in chunk.iterrows():
                atomic = make_atomic_tokens(row, include_high_cardinality=include_high_cardinality)
                comp = "EVENT_COMPOSITE::" + "|".join(atomic)
                tid = token_to_id.get(comp, token_to_id["[UNK]"])
                rec = {
                    "event_id": row.get("event_id"),
                    "PatientDurableKey": row.get("PatientDurableKey"),
                    "EncounterKey": row.get("EncounterKey"),
                    "event_date": row.get("event_date"),
                    "event_time": row.get("event_time"),
                    "patient_event_index": row.get("patient_event_index"),
                    "event_source": row.get("event_source"),
                    "event_grain": row.get("event_grain"),
                    "days_since_previous_patient_event": row.get("days_since_previous_patient_event"),
                    "gap_bin": row.get("gap_bin"),
                    "event_token_id": tid,
                    "event_composite_token": comp,
                    "atomic_token_strings": json.dumps(atomic, ensure_ascii=False),
                    "atomic_token_ids": json.dumps([token_to_id.get(a, token_to_id["[UNK]"]) for a in atomic]),
                    "n_atomic_tokens": len(atomic),
                    "token_version": TOKEN_VERSION,
                    "next_event_id": row.get("next_event_id"),
                    "target_next_gap_days": row.get("target_next_gap_days"),
                    "target_next_gap_bin": row.get("target_next_gap_bin"),
                    "target_next_event_source": row.get("target_next_event_source"),
                    "target_next_Type": row.get("target_next_Type"),
                    "target_next_GroupCode": row.get("target_next_GroupCode"),
                    "target_next_DiagnosisValue": row.get("target_next_DiagnosisValue"),
                    "target_next_DepartmentType": row.get("target_next_DepartmentType"),
                    "target_next_DepartmentSpecialty": row.get("target_next_DepartmentSpecialty"),
                    "target_next_department_volume_bin": row.get("target_next_department_volume_bin"),
                    "target_next_provider_Type": row.get("target_next_provider_Type"),
                    "target_next_provider_PrimarySpecialty": row.get("target_next_provider_PrimarySpecialty"),
                    "target_next_setting_value": row.get("target_next_setting_value"),
                }
                event_rows.append(rec)
                min_rows.append({
                    "PatientDurableKey": row.get("PatientDurableKey"),
                    "event_id": row.get("event_id"),
                    "EncounterKey": row.get("EncounterKey"),
                    "event_date": row.get("event_date"),
                    "patient_event_index": int(row.get("patient_event_index")),
                    "event_token_id": int(tid),
                    "event_composite_token": comp,
                })
                if row.get("next_event_id") not in (None, ""):
                    train_rows.append({
                        "PatientDurableKey": row.get("PatientDurableKey"),
                        "current_event_id": row.get("event_id"),
                        "current_patient_event_index": row.get("patient_event_index"),
                        "current_event_token_id": tid,
                        "next_event_id": row.get("next_event_id"),
                        "target_next_gap_days": row.get("target_next_gap_days"),
                        "target_next_gap_bin": row.get("target_next_gap_bin"),
                        "target_next_event_source": row.get("target_next_event_source"),
                        "target_next_Type": row.get("target_next_Type"),
                        "target_next_GroupCode": row.get("target_next_GroupCode"),
                        "target_next_DiagnosisValue": row.get("target_next_DiagnosisValue"),
                        "target_next_DepartmentType": row.get("target_next_DepartmentType"),
                        "target_next_DepartmentSpecialty": row.get("target_next_DepartmentSpecialty"),
                        "target_next_department_volume_bin": row.get("target_next_department_volume_bin"),
                        "target_next_provider_Type": row.get("target_next_provider_Type"),
                        "target_next_provider_PrimarySpecialty": row.get("target_next_provider_PrimarySpecialty"),
                        "target_next_setting_value": row.get("target_next_setting_value"),
                        "token_version": TOKEN_VERSION,
                    })
            pd.DataFrame(event_rows).to_csv(out_path, index=False, mode="w" if first_events else "a", header=first_events, compression="gzip")
            pd.DataFrame(min_rows).to_sql("token_event_min", conn, if_exists="append", index=False)
            if train_rows:
                pd.DataFrame(train_rows).to_csv(train_path, index=False, mode="w" if first_train else "a", header=first_train, compression="gzip")
                first_train = False
            first_events = False
            total += len(event_rows)
            pbar.update(len(chunk))
    finally:
        pbar.close()
    print(f"[TOKEN] encoded event rows: {total:,}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_event_min_patient_idx ON token_event_min(PatientDurableKey, patient_event_index)")
    conn.commit()
    print(f"[TOKEN] wrote {out_path}")
    print(f"[TOKEN] wrote {train_path}")


def build_next_event_training_examples(conn: sqlite3.Connection, output_dir: Path, chunksize: int) -> None:
    """Build next-event supervision after every event token id is known.

    The original in-loop writer could emit auxiliary next-event fields, but it could
    not reliably include target_next_event_token_id because the next row might appear
    in a later chunk. This post-pass joins each current event to the next event for
    the same patient by patient_event_index + 1, preserving the auxiliary WHAT /
    WHEN / WHERE targets from event_work.
    """
    print("[TOKEN] building next-event training examples with target_next_event_token_id")
    train_path = output_dir / "next_event_training_examples_v2.csv.gz"
    if train_path.exists():
        train_path.unlink()

    query = """
    SELECT
      cur.PatientDurableKey,
      cur.event_id AS current_event_id,
      cur.patient_event_index AS current_patient_event_index,
      cur.event_token_id AS current_event_token_id,
      cur.event_composite_token AS current_event_composite_token,

      nxt.event_id AS next_event_id,
      nxt.patient_event_index AS next_patient_event_index,
      nxt.event_token_id AS target_next_event_token_id,
      nxt.event_composite_token AS target_next_event_composite_token,

      ew.target_next_gap_days,
      ew.target_next_gap_bin,
      ew.target_next_event_source,
      ew.target_next_Type,
      ew.target_next_GroupCode,
      ew.target_next_DiagnosisValue,
      ew.target_next_DepartmentType,
      ew.target_next_DepartmentSpecialty,
      ew.target_next_department_volume_bin,
      ew.target_next_provider_Type,
      ew.target_next_provider_PrimarySpecialty,
      ew.target_next_setting_value,
      'v2_what_when_where_event_composite_numeric' AS token_version
    FROM token_event_min cur
    JOIN token_event_min nxt
      ON cur.PatientDurableKey = nxt.PatientDurableKey
     AND nxt.patient_event_index = cur.patient_event_index + 1
    LEFT JOIN event_work ew
      ON cur.event_id = ew.event_id
    ORDER BY cur.PatientDurableKey, cur.patient_event_index
    """

    first = True
    total = 0
    n_examples = conn.execute("""
        SELECT COUNT(*)
        FROM token_event_min cur
        JOIN token_event_min nxt
          ON cur.PatientDurableKey = nxt.PatientDurableKey
         AND nxt.patient_event_index = cur.patient_event_index + 1
    """).fetchone()[0]
    pbar = _progress_bar(total=n_examples, desc="Next-event examples", unit="row")
    try:
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
            chunk.to_csv(
                train_path,
                index=False,
                mode="w" if first else "a",
                header=first,
                compression="gzip",
            )
            first = False
            n = len(chunk)
            total += n
            pbar.update(n)
    finally:
        pbar.close()
    print(f"[TOKEN] next-event training examples written: {total:,}")
    print(f"[TOKEN] wrote {train_path}")


def validate_tokenization_outputs(output_dir: Path, sample_n: int = 100_000) -> None:
    """Print lightweight sanity checks for patient / WHAT / WHEN / WHERE coverage."""
    event_path = output_dir / "event_tokenized_v2.csv.gz"
    vocab_path = output_dir / "token_vocabulary_v2.csv"
    train_path = output_dir / "next_event_training_examples_v2.csv.gz"

    if not event_path.exists() or not vocab_path.exists():
        print("[TOKEN VALIDATION] skipped: tokenized event or vocabulary file not found")
        return

    df = pd.read_csv(event_path, nrows=sample_n)
    vocab = pd.read_csv(vocab_path)

    required_prefixes = {
        "PAT_": "patient context",
        "EVT_TYPE_": "WHAT: event type",
        "EVT_DESC_": "WHAT: description",
        "EVT_DXG_": "WHAT: diagnosis group",
        "EVT_GAP_": "WHEN: previous gap",
        "EVT_MONTH_": "WHEN: calendar quarter",
        "EVT_HOURBIN_": "WHEN: time of day",
        "EVT_SETTING_": "WHERE: care setting",
        "EVT_DEPT_TYPE_": "WHERE: department type",
        "EVT_DEPT_SPEC_": "WHERE: department specialty",
        "EVT_DEPT_VOLUME_": "WHERE: department volume",
        "EVT_PROVIDER_TYPE_": "WHERE: provider type",
    }

    print("\n[TOKEN VALIDATION] Basic output shape")
    print(f"[TOKEN VALIDATION] sampled_event_rows={len(df):,}")
    print(f"[TOKEN VALIDATION] vocab_rows={len(vocab):,}")
    if "event_token_id" in df.columns:
        print(f"[TOKEN VALIDATION] sampled_UNK_event_token_rate={(df['event_token_id'] == 3).mean():.4%}")

    print("\n[TOKEN VALIDATION] Required token prefix coverage")
    for prefix, meaning in required_prefixes.items():
        coverage = df["atomic_token_strings"].str.contains(prefix, regex=False, na=False).mean()
        print(f"[TOKEN VALIDATION] {prefix:<20} {meaning:<32} coverage={coverage:.2%}")

    print("\n[TOKEN VALIDATION] WHEN missing-token rates")
    for token in ["EVT_GAP_MISSING", "EVT_MONTH_MISSING", "EVT_HOURBIN_MISSING"]:
        rate = df["atomic_token_strings"].str.contains(token, regex=False, na=False).mean()
        print(f"[TOKEN VALIDATION] {token:<25} rate={rate:.2%}")

    if train_path.exists():
        train = pd.read_csv(train_path, nrows=sample_n)
        has_target = "target_next_event_token_id" in train.columns
        null_rate = train["target_next_event_token_id"].isna().mean() if has_target else 1.0
        print("\n[TOKEN VALIDATION] Next-event target")
        print(f"[TOKEN VALIDATION] has_target_next_event_token_id={has_target}")
        print(f"[TOKEN VALIDATION] sampled_target_next_event_token_id_null_rate={null_rate:.4%}")


def build_patient_sequences(conn: sqlite3.Connection, output_dir: Path, chunksize: int) -> None:
    print("[TOKEN] building patient sequences")
    out_path = output_dir / "patient_tokenized_sequence_v2.csv.gz"
    if out_path.exists():
        out_path.unlink()
    query = """
    WITH ordered AS (
      SELECT * FROM token_event_min ORDER BY PatientDurableKey, patient_event_index
    ), grouped AS (
      SELECT
        PatientDurableKey,
        COUNT(*) AS n_events,
        MIN(event_date) AS first_event_date,
        MAX(event_date) AS last_event_date,
        GROUP_CONCAT(event_token_id, ',') AS event_token_ids_no_specials,
        GROUP_CONCAT(event_id, ',') AS event_ids,
        GROUP_CONCAT(EncounterKey, ',') AS encounter_keys
      FROM ordered
      GROUP BY PatientDurableKey
    )
    SELECT
      PatientDurableKey,
      n_events,
      first_event_date,
      last_event_date,
      n_events + 2 AS n_tokens_with_specials,
      '[' || '1,' || event_token_ids_no_specials || ',2' || ']' AS patient_token_ids,
      event_token_ids_no_specials,
      event_ids,
      encounter_keys,
      'v2_what_when_where_event_composite_numeric' AS token_version
    FROM grouped
    ORDER BY PatientDurableKey
    """
    n_patients = conn.execute("SELECT COUNT(DISTINCT PatientDurableKey) FROM token_event_min").fetchone()[0]
    first = True
    total = 0
    pbar = _progress_bar(total=n_patients, desc="Patient sequences", unit="patient")
    try:
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
            chunk.to_csv(out_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
            first = False
            n = len(chunk)
            total += n
            pbar.update(n)
    finally:
        pbar.close()
    print(f"[TOKEN] patient sequences written: {total:,}")
    print(f"[TOKEN] wrote {out_path}")


def write_feature_spec(docs_dir: Path, input_path: Path, missing: list[str], include_high_cardinality: bool) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "tokenization_feature_spec_v2.md"
    content = f"""# Tokenization Feature Spec v2: WHAT / WHEN / WHERE

## Input

```text
{input_path}
```

## Central rule

```text
one event_enriched row -> one composite event token -> one event_token_id
```

## Output files

```text
token_vocabulary_v2.csv
event_tokenized_v2.csv.gz
patient_tokenized_sequence_v2.csv.gz
next_event_training_examples_v2.csv.gz
```

## Token groups

### WHAT tokens

```text
EVT_SRC_{{event_source}}
EVT_GRAIN_{{event_grain}}
EVT_TYPE_{{event_type or Type}}
EVT_SUBTYPE_{{event_subtype or VisitType}}
EVT_DESC_{{event_description or VisitTypeDescription}}
EVT_ADMISSION_SOURCE_{{AdmissionSource}}
EVT_ADMISSION_TYPE_{{AdmissionType}}
EVT_DX_MISSING_CLASS_{{primary_diagnosis_missing_class}}
EVT_DXG_{{GroupCode}}
EVT_DX_{{DiagnosisValue}}  # only when --include-high-cardinality is used
EVT_DOMAIN_{{event_domain}}
EVT_VALUE_{{event_value}}  # only when --include-high-cardinality is used
```

### WHEN tokens

```text
EVT_GAP_{{gap_bin}}
EVT_MONTH_{{Q1/Q2/Q3/Q4}}
EVT_HOURBIN_{{NIGHT/MORNING/AFTERNOON/EVENING}}
```

### WHERE tokens

```text
EVT_SETTING_{{ED/HOSP_ADMIT/HOSP_OP/INPATIENT/OBS/OP_FACE/NONE}}
EVT_DEPT_TYPE_{{DepartmentType}}
EVT_DEPT_SPEC_{{DepartmentSpecialty}}
EVT_DEPT_COUNTY_{{department_County or County}}
EVT_DEPT_TRACT_PROX_{{SAME_TRACT/DIFFERENT_TRACT/UNKNOWN}}
EVT_DEPT_VOLUME_{{observed department volume bin}}
EVT_PROVIDER_TYPE_{{provider_Type}}
EVT_ATTENDING_TYPE_{{attending_provider_Type}}
EVT_DISCHARGE_TYPE_{{discharge_provider_Type}}
EVT_TRANSFER_{{START/SAME_DEPARTMENT/ED_TO_LARGER_FACILITY/LOCAL_TO_LARGER/LARGER_TO_LOCAL/SAME_SETTING/OTHER}}
```

### Patient context tokens

```text
PAT_AGEBIN_{{PatientBirthYearBin}}
PAT_SEX_{{SexAssignedAtBirth}}
PAT_RACE_{{OmbRace}}
PAT_ETH_{{OmbEthnicity}}
PAT_MARITAL_{{MaritalStatus}}
PAT_SMOKE_{{SmokingStatus}}
PAT_VITAL_{{VitalStatus}}
PAT_MYCHART_{{MyChartStatus}}
PAT_ORIENT_{{SexualOrientation}}
PAT_GEO_KNOWN_{{YES/NO}}
PAT_HOME_POPBIN_{{patient_block_population_bin or PopulationValue bin}}
PAT_SDOH_OBS_{{YES/NO}}
PAT_SDOH_QCOUNT_{{0/1/2/3PLUS}}
PAT_SDOH_DOMAINCOUNT_{{0/1/2/3PLUS}}
PAT_SDOH_DOMAIN_{{Domain}}
```

## Modeling targets produced

```text
target_next_event_token_id
target_next_event_composite_token
target_next_gap_days
target_next_gap_bin
target_next_event_source
target_next_Type
target_next_GroupCode
target_next_DiagnosisValue
target_next_DepartmentType
target_next_DepartmentSpecialty
target_next_department_volume_bin
target_next_provider_Type
target_next_provider_PrimarySpecialty
target_next_setting_value
```

## High-cardinality mode

```text
include_high_cardinality = {include_high_cardinality}
```

When this is false, exact `DiagnosisValue`, exact SDOH answer value, department city/postal, and provider specialties are not placed into the composite event token. They are still available in the source data and some are still output as targets.

## Optional columns missing from this input

```text
{chr(10).join(missing) if missing else 'None'}
```

## Geography limitation

`CENTLAT` and `CENTLON` in this table describe patient home census block group geography. The current event_enriched file does not include department latitude/longitude, so this tokenizer does not compute true travel distance. It instead creates `EVT_DEPT_TRACT_PROX` from patient block group vs. department census tract when available.
"""
    out.write_text(content, encoding="utf-8")
    print(f"[TOKEN] wrote {out}")


def build_model_inputs(output_dir: Path, max_seq_len: int, chunksize: int) -> None:
    if max_seq_len <= 0:
        return
    seq_path = output_dir / "patient_tokenized_sequence_v2.csv.gz"
    out_path = output_dir / "patient_token_model_inputs_v2.csv.gz"
    if out_path.exists():
        out_path.unlink()
    first = True
    total = 0
    pbar = _progress_bar(total=None, desc="Model inputs (trunc/pad)", unit="row")
    try:
        for chunk in pd.read_csv(seq_path, dtype=str, chunksize=chunksize):
            rows = []
            for _, row in chunk.iterrows():
                ids = json.loads(row["patient_token_ids"])
                original_len = len(ids)
                ids2 = ids[:max_seq_len]
                mask = [1] * len(ids2)
                if len(ids2) < max_seq_len:
                    pad = max_seq_len - len(ids2)
                    ids2 += [0] * pad
                    mask += [0] * pad
                rows.append({
                    "PatientDurableKey": row["PatientDurableKey"],
                    "input_ids": json.dumps(ids2),
                    "attention_mask": json.dumps(mask),
                    "sequence_length_original": original_len,
                    "sequence_length_model": max_seq_len,
                    "was_truncated": int(original_len > max_seq_len),
                    "token_version": TOKEN_VERSION,
                })
            pd.DataFrame(rows).to_csv(out_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
            first = False
            n = len(rows)
            total += n
            pbar.update(n)
    finally:
        pbar.close()
    print(f"[TOKEN] model input rows written: {total:,}")
    print(f"[TOKEN] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build numeric event tokens for WHAT/WHEN/WHERE next-event forecasting.")
    p.add_argument("--input", type=Path, default=Path("data/processed/event_enriched.csv.gz"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--docs-dir", type=Path, default=Path("docs"))
    p.add_argument("--work-db", type=Path, default=Path("data/interim/event_tokenization_v2.sqlite"))
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--min-event-token-frequency", type=int, default=1)
    p.add_argument("--include-high-cardinality", action="store_true", help="Include exact diagnosis value, event value, department city/postal, and provider specialties in composite token.")
    p.add_argument("--max-seq-len", type=int, default=0)
    p.add_argument("--replace", action="store_true")
    p.add_argument(
        "--skip-stage",
        action="store_true",
        help="Skip loading CSV into event_stage; use existing event_stage in --work-db (continue after staging finished).",
    )
    p.add_argument(
        "--skip-event-work",
        action="store_true",
        help="Skip build_event_work (long SQL); use existing event_work in --work-db (resume after that step finished).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (plain log lines only).",
    )
    return p.parse_args()


def main() -> None:
    global SHOW_PROGRESS
    args = parse_args()
    SHOW_PROGRESS = not args.no_progress
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.skip_stage and args.replace:
        raise SystemExit("--skip-stage cannot be combined with --replace (that deletes --work-db).")
    if args.skip_event_work and args.replace:
        raise SystemExit("--skip-event-work cannot be combined with --replace (that deletes --work-db).")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    args.work_db.parent.mkdir(parents=True, exist_ok=True)
    if args.replace and args.work_db.exists():
        args.work_db.unlink()
    if args.skip_stage and not args.work_db.exists():
        raise SystemExit(f"--work-db not found: {args.work_db}")
    with sqlite3.connect(args.work_db) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=FILE;")
        conn.execute("PRAGMA cache_size=-200000;")
        if args.skip_stage:
            _ensure_event_stage_ready(conn)
            missing = missing_columns_from_input_header(args.input)
            n = conn.execute("SELECT COUNT(*) FROM event_stage").fetchone()[0]
            print(f"[TOKEN] --skip-stage: using existing event_stage ({n:,} rows) in {args.work_db}")
        else:
            missing = load_event_stage(conn, args.input, args.chunksize)
        if args.skip_event_work:
            _ensure_event_work_ready(conn)
            n_ew = conn.execute("SELECT COUNT(*) FROM event_work").fetchone()[0]
            print(f"[TOKEN] --skip-event-work: using existing event_work ({n_ew:,} rows) in {args.work_db}")
        else:
            build_event_work(conn)
        vocab = build_vocabulary(conn, args.output_dir, args.chunksize, args.min_event_token_frequency, args.include_high_cardinality)
        build_event_tokenized(conn, args.output_dir, vocab, args.chunksize, args.include_high_cardinality)
        build_next_event_training_examples(conn, args.output_dir, args.chunksize)
        build_patient_sequences(conn, args.output_dir, args.chunksize)
    build_model_inputs(args.output_dir, args.max_seq_len, args.chunksize)
    write_feature_spec(args.docs_dir, args.input, missing, args.include_high_cardinality)
    validate_tokenization_outputs(args.output_dir)
    print("[TOKEN] done")


if __name__ == "__main__":
    main()
