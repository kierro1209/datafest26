#!/usr/bin/env python3
"""
Build numeric, LLM-style event tokens from DataFest processed event data.

Main design choice
------------------
This script tokenizes at the EVENT grain, not the encounter grain.
Each row in event_enriched becomes exactly one composite numeric event token:

    one event row -> one event_token_id

For interpretability/debugging, the script also saves atomic token pieces for each
row. The patient-level model sequence uses one integer per event, wrapped by
[PATIENT_START] and [PATIENT_END].

Expected primary input:
    data/processed/event_enriched.csv.gz

Outputs:
    data/processed/token_vocabulary.csv
    data/processed/event_tokenized.csv.gz
    data/processed/patient_tokenized_sequence.csv.gz
    data/processed/patient_token_model_inputs.csv.gz   optional, if --max-seq-len > 0
    docs/tokenization_feature_spec.md                  if --docs-dir exists or is provided

Usage:
    python src/build_numeric_event_tokens.py \
      --input data/processed/event_enriched.csv.gz \
      --output-dir data/processed \
      --docs-dir docs

Notes:
    - Uses only pandas + SQLite + stdlib.
    - Reads/writes in chunks.
    - Does not assume SDOH risk scoring. It preserves observed SDOH domains and
      answer tokens as categorical token pieces.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

CSV_ENCODING_CANDIDATES = ["utf-8-sig", "utf-16", "cp1252", "latin1"]

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

# These are exact column names we will use when present in event_enriched.
# event_enriched may have event_* names for event-specific fields and regular
# encounter-enriched names for patient/diagnosis/department/provider context.
ID_TIME_COLUMNS = [
    "event_id",
    "event_source",
    "event_grain",
    "event_EncounterKey",
    "event_PatientDurableKey",
    "event_date",
    "event_time",
    "event_index_within_encounter",
    "EncounterKey",
    "PatientDurableKey",
    "Date",
    "AdmissionInstant",
]

PATIENT_FEATURE_COLUMNS = [
    "PatientBirthYearBin",
    "SexAssignedAtBirth",
    "OmbRace",
    "OmbEthnicity",
    "MaritalStatus",
    "SmokingStatus",
    "VitalStatus",
    "MyChartStatus",
    "SexualOrientation",
    "CensusBlockGroupFipsCode",
    "PopulationValue",
    "sdoh_any_observed",
    "sdoh_num_domains_answered",
    "sdoh_domains_observed",
    "sdoh_domain_answer_tokens",
]

EVENT_FEATURE_COLUMNS = [
    "event_type",
    "event_subtype",
    "event_description",
    "event_domain",
    "event_value",
    "Type",
    "VisitType",
    "VisitTypeDescription",
    "PrimaryDiagnosisKey",
    "DiagnosisKey",
    "DiagnosisValue",
    "DiagnosisName",
    "GroupCode",
    "GroupName",
    "DepartmentKey",
    "DepartmentName",
    "DepartmentSpecialty",
    "DepartmentType",
    "City",
    "County",
    "PostalCode",
    "CensusTract",
    "provider_Type",
    "provider_ClinicianTitle",
    "provider_PrimarySpecialty",
    "attending_provider_Type",
    "attending_provider_ClinicianTitle",
    "attending_provider_PrimarySpecialty",
    "discharge_provider_Type",
    "discharge_provider_ClinicianTitle",
    "discharge_provider_PrimarySpecialty",
    *CARE_SETTING_FLAGS,
]

USED_FEATURE_COLUMNS = ID_TIME_COLUMNS + PATIENT_FEATURE_COLUMNS + EVENT_FEATURE_COLUMNS

STAGE_COLUMNS = [
    # standardized identity/time fields
    "event_id",
    "PatientDurableKey",
    "EncounterKey",
    "event_date",
    "event_time",
    "event_index_within_encounter",
    "event_source",
    "event_grain",
    # patient context
    *PATIENT_FEATURE_COLUMNS,
    # event context
    *EVENT_FEATURE_COLUMNS,
]


def choose_encoding(path: Path) -> str:
    errors: list[str] = []
    for enc in CSV_ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, nrows=0, encoding=enc)
            return enc
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{enc}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not read {path} with candidate encodings. " + " | ".join(errors))


def read_header(path: Path, encoding: str) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding=encoding).columns)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_value(value: object, max_len: int = 80) -> str:
    """Normalize a raw field value into a stable token value."""
    if value is None:
        return "MISSING"
    try:
        if pd.isna(value):
            return "MISSING"
    except Exception:  # noqa: BLE001
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
    if not s:
        return "MISSING"
    return s[:max_len]


def trueish(value: object) -> bool:
    return normalize_value(value) in {"1", "TRUE", "T", "YES", "Y"}


def bin_count(value: object) -> str:
    try:
        n = int(float(str(value).strip()))
    except Exception:  # noqa: BLE001
        return "MISSING"
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3PLUS"


def bin_population(value: object) -> str:
    try:
        n = float(str(value).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return "MISSING"
    # Conservative generic bins; adjust after inspecting distribution.
    if n <= 0:
        return "MISSING"
    if n < 500:
        return "LT500"
    if n < 1000:
        return "500_999"
    if n < 2000:
        return "1000_1999"
    if n < 5000:
        return "2000_4999"
    return "5000PLUS"


def first_present(row: pd.Series, names: list[str], default: object = "") -> object:
    for name in names:
        if name in row.index:
            val = row.get(name)
            if val is not None and not (isinstance(val, float) and pd.isna(val)) and str(val).strip() != "":
                return val
    return default


def standardize_chunk(df: pd.DataFrame, row_offset: int) -> pd.DataFrame:
    """Map whatever event_enriched provides into a fixed staging schema."""
    out = pd.DataFrame(index=df.index)

    def col_or_blank(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    # Standardized ID/time fields.
    if "event_id" in df.columns:
        out["event_id"] = col_or_blank("event_id")
    else:
        out["event_id"] = [f"ROW_EVENT_{row_offset + i}" for i in range(len(df))]

    out["PatientDurableKey"] = col_or_blank("event_PatientDurableKey")
    if (out["PatientDurableKey"].str.len() == 0).any():
        fallback = col_or_blank("PatientDurableKey")
        out["PatientDurableKey"] = out["PatientDurableKey"].where(out["PatientDurableKey"].str.len() > 0, fallback)

    out["EncounterKey"] = col_or_blank("event_EncounterKey")
    if (out["EncounterKey"].str.len() == 0).any():
        fallback = col_or_blank("EncounterKey")
        out["EncounterKey"] = out["EncounterKey"].where(out["EncounterKey"].str.len() > 0, fallback)

    out["event_date"] = col_or_blank("event_date")
    if (out["event_date"].str.len() == 0).any():
        fallback = col_or_blank("Date")
        out["event_date"] = out["event_date"].where(out["event_date"].str.len() > 0, fallback)

    out["event_time"] = col_or_blank("event_time")
    if (out["event_time"].str.len() == 0).any():
        fallback = col_or_blank("AdmissionInstant")
        out["event_time"] = out["event_time"].where(out["event_time"].str.len() > 0, fallback)

    out["event_index_within_encounter"] = col_or_blank("event_index_within_encounter")
    out["event_source"] = col_or_blank("event_source")
    out["event_source"] = out["event_source"].where(out["event_source"].str.len() > 0, "UNKNOWN_EVENT_SOURCE")
    out["event_grain"] = col_or_blank("event_grain")
    out["event_grain"] = out["event_grain"].where(out["event_grain"].str.len() > 0, "EVENT")

    # Carry expected feature columns exactly when present, otherwise blank.
    for c in PATIENT_FEATURE_COLUMNS + EVENT_FEATURE_COLUMNS:
        if c in out.columns:
            continue
        out[c] = col_or_blank(c)

    return out[STAGE_COLUMNS]


def make_setting_value(row: pd.Series) -> str:
    labels = []
    mapping = {
        "IsEdVisit": "ED",
        "IsHospitalAdmission": "HOSP_ADMIT",
        "IsHospitalOutpatientVisit": "HOSP_OP",
        "IsInpatientAdmission": "INPATIENT",
        "IsObservation": "OBS",
        "IsOutpatientFaceToFaceVisit": "OP_FACE",
    }
    for col, label in mapping.items():
        if col in row.index and trueish(row.get(col)):
            labels.append(label)
    return "__".join(labels) if labels else "NONE"


def make_atomic_tokens(row: pd.Series) -> list[str]:
    """Atomic pieces used to construct/debug the one-event composite token."""
    tokens: list[str] = []

    # Patient context. Keep this intentionally low-cardinality.
    tokens.append(f"PAT_AGEBIN_{normalize_value(row.get('PatientBirthYearBin'))}")
    tokens.append(f"PAT_SEX_{normalize_value(row.get('SexAssignedAtBirth'))}")
    tokens.append(f"PAT_RACE_{normalize_value(row.get('OmbRace'))}")
    tokens.append(f"PAT_ETH_{normalize_value(row.get('OmbEthnicity'))}")
    tokens.append(f"PAT_MARITAL_{normalize_value(row.get('MaritalStatus'))}")
    tokens.append(f"PAT_SMOKE_{normalize_value(row.get('SmokingStatus'))}")
    tokens.append(f"PAT_VITAL_{normalize_value(row.get('VitalStatus'))}")
    tokens.append(f"PAT_MYCHART_{normalize_value(row.get('MyChartStatus'))}")
    tokens.append(f"PAT_ORIENT_{normalize_value(row.get('SexualOrientation'))}")

    geo_known = "YES" if normalize_value(row.get("CensusBlockGroupFipsCode")) not in {"MISSING", "ASKED_NOT_ANSWERED_OR_UNABLE", "NOT_RECORDED_OR_UNKNOWN"} else "NO"
    tokens.append(f"PAT_GEO_KNOWN_{geo_known}")
    tokens.append(f"PAT_POPBIN_{bin_population(row.get('PopulationValue'))}")

    sdoh_obs = "YES" if trueish(row.get("sdoh_any_observed")) else "NO"
    tokens.append(f"PAT_SDOH_OBS_{sdoh_obs}")
    tokens.append(f"PAT_SDOH_DOMAINCOUNT_{bin_count(row.get('sdoh_num_domains_answered'))}")

    domains = str(row.get("sdoh_domains_observed") or "").strip()
    if domains:
        # Preserve domains, not answer scoring.
        for d in re.split(r"[,|;]+", domains):
            d_norm = normalize_value(d)
            if d_norm != "MISSING":
                tokens.append(f"PAT_SDOH_DOMAIN_{d_norm}")

    # Event context. These are the main event-level attributes.
    tokens.append(f"EVT_SRC_{normalize_value(row.get('event_source'))}")
    tokens.append(f"EVT_GRAIN_{normalize_value(row.get('event_grain'))}")
    tokens.append(f"EVT_GAP_{normalize_value(row.get('gap_bin'))}")
    tokens.append(f"EVT_TYPE_{normalize_value(first_present(row, ['event_type', 'Type']))}")
    tokens.append(f"EVT_SUBTYPE_{normalize_value(first_present(row, ['event_subtype', 'VisitType']))}")
    tokens.append(f"EVT_DESC_{normalize_value(first_present(row, ['event_description', 'VisitTypeDescription']))}")
    tokens.append(f"EVT_DOMAIN_{normalize_value(row.get('event_domain'))}")
    tokens.append(f"EVT_VALUE_{normalize_value(row.get('event_value'))}")

    # Diagnosis: GroupCode is lower-cardinality and usually best for modeling.
    # DiagnosisValue is included to preserve detail, but can be dropped later if too sparse.
    tokens.append(f"EVT_DXG_{normalize_value(row.get('GroupCode'))}")
    tokens.append(f"EVT_DX_{normalize_value(row.get('DiagnosisValue'))}")

    # Department / hospital context.
    tokens.append(f"EVT_DEPT_TYPE_{normalize_value(row.get('DepartmentType'))}")
    tokens.append(f"EVT_DEPT_SPEC_{normalize_value(row.get('DepartmentSpecialty'))}")
    tokens.append(f"EVT_DEPT_COUNTY_{normalize_value(row.get('County'))}")

    # Provider context. Do not use provider IDs; use role-level metadata only.
    tokens.append(f"EVT_PROVIDER_TYPE_{normalize_value(row.get('provider_Type'))}")
    tokens.append(f"EVT_PROVIDER_SPEC_{normalize_value(row.get('provider_PrimarySpecialty'))}")
    tokens.append(f"EVT_ATTENDING_TYPE_{normalize_value(row.get('attending_provider_Type'))}")
    tokens.append(f"EVT_ATTENDING_SPEC_{normalize_value(row.get('attending_provider_PrimarySpecialty'))}")
    tokens.append(f"EVT_DISCHARGE_TYPE_{normalize_value(row.get('discharge_provider_Type'))}")
    tokens.append(f"EVT_DISCHARGE_SPEC_{normalize_value(row.get('discharge_provider_PrimarySpecialty'))}")

    tokens.append(f"EVT_SETTING_{make_setting_value(row)}")

    # Remove duplicate tokens while preserving order.
    seen = set()
    deduped = []
    for tok in tokens:
        if tok not in seen:
            deduped.append(tok)
            seen.add(tok)
    return deduped


def make_composite_event_token(row: pd.Series) -> str:
    """One event row -> one composite token string -> one integer ID."""
    atomic = make_atomic_tokens(row)
    return "EVENT_COMPOSITE::" + "|".join(atomic)


def gap_bin_from_days(days: object, event_index: object) -> str:
    try:
        idx = int(event_index)
    except Exception:  # noqa: BLE001
        idx = None
    if idx == 1:
        return "START"
    try:
        d = int(float(days))
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    if d == 0:
        return "0D"
    if 1 <= d <= 7:
        return "1_7D"
    if 8 <= d <= 30:
        return "8_30D"
    if 31 <= d <= 90:
        return "31_90D"
    if 91 <= d <= 180:
        return "91_180D"
    if 181 <= d <= 365:
        return "181_365D"
    if d > 365:
        return "365PLUS"
    return "UNKNOWN"


def load_event_stage(conn: sqlite3.Connection, input_path: Path, chunksize: int) -> list[str]:
    print(f"[TOKEN] Reading source: {input_path}")
    encoding = choose_encoding(input_path)
    header = read_header(input_path, encoding)
    usecols = [c for c in USED_FEATURE_COLUMNS if c in header]
    missing = [c for c in USED_FEATURE_COLUMNS if c not in header]

    print(f"[TOKEN] Encoding: {encoding}")
    print(f"[TOKEN] Using {len(usecols)} source columns")
    print(f"[TOKEN] Missing optional columns: {missing}")

    conn.execute("DROP TABLE IF EXISTS event_stage")
    conn.commit()

    offset = 0
    first = True
    for chunk in pd.read_csv(
        input_path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunksize,
        encoding=encoding,
    ):
        std = standardize_chunk(chunk, row_offset=offset)
        std.to_sql("event_stage", conn, if_exists="replace" if first else "append", index=False)
        first = False
        offset += len(std)
        print(f"[TOKEN] staged rows: {offset:,}")

    if first:
        raise RuntimeError("Input file contained no rows.")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_stage_patient_time ON event_stage(PatientDurableKey, event_date, event_time, EncounterKey, event_id)")
    conn.commit()
    return missing


def build_event_work(conn: sqlite3.Connection) -> None:
    print("[TOKEN] Building ordered event_work with patient event index and gap bins")
    conn.executescript(
        """
        DROP TABLE IF EXISTS event_work;
        CREATE TABLE event_work AS
        WITH ordered AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY PatientDurableKey
              ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
            ) AS patient_event_index,
            LAG(event_date) OVER (
              PARTITION BY PatientDurableKey
              ORDER BY event_date, event_time, EncounterKey, event_index_within_encounter, event_id
            ) AS previous_patient_event_date
          FROM event_stage
        ), gaps AS (
          SELECT
            *,
            CASE
              WHEN previous_patient_event_date IS NULL THEN NULL
              WHEN julianday(event_date) IS NULL OR julianday(previous_patient_event_date) IS NULL THEN NULL
              ELSE CAST(julianday(event_date) - julianday(previous_patient_event_date) AS INTEGER)
            END AS days_since_previous_patient_event
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
          END AS gap_bin
        FROM gaps;

        CREATE INDEX IF NOT EXISTS idx_event_work_patient_idx ON event_work(PatientDurableKey, patient_event_index);
        CREATE INDEX IF NOT EXISTS idx_event_work_event_id ON event_work(event_id);
        """
    )
    conn.commit()


def iter_event_work(conn: sqlite3.Connection, chunksize: int) -> Iterable[pd.DataFrame]:
    query = "SELECT * FROM event_work ORDER BY PatientDurableKey, patient_event_index"
    yield from pd.read_sql_query(query, conn, chunksize=chunksize)


def build_vocabulary(conn: sqlite3.Connection, output_dir: Path, chunksize: int, min_event_freq: int) -> dict[str, int]:
    print("[TOKEN] Pass 1/2: building token vocabulary")
    event_counter: Counter[str] = Counter()
    atomic_counter: Counter[str] = Counter()
    total_rows = 0

    for chunk in iter_event_work(conn, chunksize):
        for _, row in chunk.iterrows():
            atomic = make_atomic_tokens(row)
            composite = "EVENT_COMPOSITE::" + "|".join(atomic)
            event_counter[composite] += 1
            atomic_counter.update(atomic)
        total_rows += len(chunk)
        print(f"[TOKEN] vocabulary rows scanned: {total_rows:,}")

    rows: list[dict[str, object]] = []
    token_to_id: dict[str, int] = {}
    for token_id, token_string, token_kind, field_name, field_value in SPECIAL_TOKENS:
        token_to_id[token_string] = token_id
        rows.append({
            "token_id": token_id,
            "token_string": token_string,
            "token_kind": token_kind,
            "field_name": field_name,
            "field_value": field_value,
            "frequency": None,
        })

    next_id = max(t[0] for t in SPECIAL_TOKENS) + 1

    # Composite event tokens are the actual one-token-per-event vocabulary.
    # Sort by frequency descending, then token string for stability.
    for token_string, freq in sorted(event_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_event_freq:
            continue
        token_to_id[token_string] = next_id
        rows.append({
            "token_id": next_id,
            "token_string": token_string,
            "token_kind": "EVENT_COMPOSITE",
            "field_name": "event_composite",
            "field_value": token_string.replace("EVENT_COMPOSITE::", "", 1),
            "frequency": freq,
        })
        next_id += 1

    # Atomic tokens are saved for interpretability/debugging and optional later models.
    for token_string, freq in sorted(atomic_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if token_string in token_to_id:
            continue
        field_name = token_string.split("_", 2)[0] if "_" in token_string else "atomic"
        token_to_id[token_string] = next_id
        rows.append({
            "token_id": next_id,
            "token_string": token_string,
            "token_kind": "ATOMIC",
            "field_name": field_name,
            "field_value": token_string,
            "frequency": freq,
        })
        next_id += 1

    vocab = pd.DataFrame(rows)
    output_path = output_dir / "token_vocabulary.csv"
    vocab.to_csv(output_path, index=False)
    print(f"[TOKEN] Wrote vocabulary: {output_path} ({len(vocab):,} tokens)")
    return token_to_id


def build_event_tokenized(conn: sqlite3.Connection, output_dir: Path, token_to_id: dict[str, int], chunksize: int) -> None:
    print("[TOKEN] Pass 2/2: encoding events as numeric token IDs")
    out_path = output_dir / "event_tokenized.csv.gz"
    if out_path.exists():
        out_path.unlink()

    conn.execute("DROP TABLE IF EXISTS token_event_min")
    conn.execute(
        """
        CREATE TABLE token_event_min (
          PatientDurableKey TEXT,
          event_id TEXT,
          EncounterKey TEXT,
          event_date TEXT,
          patient_event_index INTEGER,
          event_token_id INTEGER,
          event_composite_token TEXT
        )
        """
    )
    conn.commit()

    first = True
    total = 0
    for chunk in iter_event_work(conn, chunksize):
        records: list[dict[str, object]] = []
        min_records: list[dict[str, object]] = []
        for _, row in chunk.iterrows():
            atomic = make_atomic_tokens(row)
            composite = "EVENT_COMPOSITE::" + "|".join(atomic)
            event_token_id = token_to_id.get(composite, token_to_id["[UNK]"])
            atomic_ids = [token_to_id.get(tok, token_to_id["[UNK]"]) for tok in atomic]

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
                "event_composite_token": composite,
                "event_token_id": event_token_id,
                "atomic_token_strings": json.dumps(atomic, ensure_ascii=False),
                "atomic_token_ids": json.dumps(atomic_ids),
                "n_atomic_tokens": len(atomic),
                "token_version": "v1_event_composite_numeric",
            }
            records.append(rec)
            min_records.append({
                "PatientDurableKey": row.get("PatientDurableKey"),
                "event_id": row.get("event_id"),
                "EncounterKey": row.get("EncounterKey"),
                "event_date": row.get("event_date"),
                "patient_event_index": int(row.get("patient_event_index")),
                "event_token_id": int(event_token_id),
                "event_composite_token": composite,
            })

        out_df = pd.DataFrame.from_records(records)
        out_df.to_csv(out_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
        pd.DataFrame.from_records(min_records).to_sql("token_event_min", conn, if_exists="append", index=False)
        first = False
        total += len(out_df)
        print(f"[TOKEN] encoded event rows: {total:,}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_event_min_patient_idx ON token_event_min(PatientDurableKey, patient_event_index)")
    conn.commit()
    print(f"[TOKEN] Wrote event tokens: {out_path}")


def build_patient_sequences(conn: sqlite3.Connection, output_dir: Path) -> None:
    print("[TOKEN] Building patient token-id sequences")
    out_path = output_dir / "patient_tokenized_sequence.csv.gz"
    query = """
    WITH ordered AS (
      SELECT *
      FROM token_event_min
      ORDER BY PatientDurableKey, patient_event_index
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
      'v1_event_composite_numeric' AS token_version
    FROM grouped
    ORDER BY PatientDurableKey;
    """
    df_iter = pd.read_sql_query(query, conn, chunksize=100_000)
    first = True
    total = 0
    if out_path.exists():
        out_path.unlink()
    for chunk in df_iter:
        chunk.to_csv(out_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
        first = False
        total += len(chunk)
        print(f"[TOKEN] patient sequences written: {total:,}")
    print(f"[TOKEN] Wrote patient sequences: {out_path}")


def build_model_inputs(output_dir: Path, max_seq_len: int, chunksize: int) -> None:
    if max_seq_len <= 0:
        return
    print(f"[TOKEN] Building padded model inputs, max_seq_len={max_seq_len}")
    seq_path = output_dir / "patient_tokenized_sequence.csv.gz"
    out_path = output_dir / "patient_token_model_inputs.csv.gz"
    if out_path.exists():
        out_path.unlink()
    first = True
    total = 0

    for chunk in pd.read_csv(seq_path, dtype=str, chunksize=chunksize):
        rows = []
        for _, row in chunk.iterrows():
            ids = json.loads(row["patient_token_ids"])
            original_len = len(ids)
            truncated = ids[:max_seq_len]
            mask = [1] * len(truncated)
            if len(truncated) < max_seq_len:
                pad_n = max_seq_len - len(truncated)
                truncated = truncated + [0] * pad_n
                mask = mask + [0] * pad_n
            rows.append({
                "PatientDurableKey": row["PatientDurableKey"],
                "input_ids": json.dumps(truncated),
                "attention_mask": json.dumps(mask),
                "sequence_length_original": original_len,
                "sequence_length_model": max_seq_len,
                "was_truncated": int(original_len > max_seq_len),
                "token_version": row.get("token_version", "v1_event_composite_numeric"),
            })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
        first = False
        total += len(out_df)
        print(f"[TOKEN] model input rows written: {total:,}")
    print(f"[TOKEN] Wrote model inputs: {out_path}")


def write_feature_spec(docs_dir: Path, input_path: Path, missing_optional_columns: list[str]) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "tokenization_feature_spec.md"
    content = f"""# Numeric Event Tokenization Feature Spec

## Goal

Create LLM-style numeric token IDs at the **event** grain.

The central rule is:

```text
one row in event_enriched -> one composite event token string -> one event_token_id
```

Patient-level sequences are then:

```text
[PATIENT_START], event_token_id_1, event_token_id_2, ..., [PATIENT_END]
```

## Input

```text
{input_path}
```

## Outputs

```text
token_vocabulary.csv
event_tokenized.csv.gz
patient_tokenized_sequence.csv.gz
patient_token_model_inputs.csv.gz  # only if --max-seq-len > 0
```

## Special token IDs

| token_string | token_id |
|---|---:|
| `[PAD]` | 0 |
| `[PATIENT_START]` | 1 |
| `[PATIENT_END]` | 2 |
| `[UNK]` | 3 |
| `[MISSING]` | 4 |

## Patient-context columns used when present

```text
{chr(10).join(PATIENT_FEATURE_COLUMNS)}
```

## Event-context columns used when present

```text
{chr(10).join(EVENT_FEATURE_COLUMNS)}
```

## Identity/time columns used when present

```text
{chr(10).join(ID_TIME_COLUMNS)}
```

## Optional columns not found in this input

```text
{chr(10).join(missing_optional_columns) if missing_optional_columns else 'None'}
```

## Token construction

Each event row is converted into atomic token pieces, including:

```text
PAT_AGEBIN_{{PatientBirthYearBin}}
PAT_SEX_{{SexAssignedAtBirth}}
PAT_RACE_{{OmbRace}}
PAT_ETH_{{OmbEthnicity}}
PAT_MYCHART_{{MyChartStatus}}
PAT_SDOH_OBS_{{YES/NO}}
PAT_SDOH_DOMAINCOUNT_{{0/1/2/3PLUS}}
EVT_SRC_{{event_source}}
EVT_GRAIN_{{event_grain}}
EVT_GAP_{{gap_bin}}
EVT_TYPE_{{event_type or Type}}
EVT_SUBTYPE_{{event_subtype or VisitType}}
EVT_DESC_{{event_description or VisitTypeDescription}}
EVT_DOMAIN_{{event_domain}}
EVT_VALUE_{{event_value}}
EVT_DXG_{{GroupCode}}
EVT_DX_{{DiagnosisValue}}
EVT_DEPT_TYPE_{{DepartmentType}}
EVT_DEPT_SPEC_{{DepartmentSpecialty}}
EVT_PROVIDER_TYPE_{{provider_Type}}
EVT_ATTENDING_TYPE_{{attending_provider_Type}}
EVT_DISCHARGE_TYPE_{{discharge_provider_Type}}
EVT_SETTING_{{care setting flags}}
```

The final composite token is:

```text
EVENT_COMPOSITE::atomic_token_1|atomic_token_2|...|atomic_token_n
```

Then `token_vocabulary.csv` maps this composite token to one integer `event_token_id`.

## Important caveat

This is event-level tokenization over the events available in `event_enriched`. If the raw data do not contain row-level lab/procedure/medication events, the script cannot invent them. It will tokenize the event-like rows that exist, such as encounter events and SDOH-response events.
"""
    out.write_text(content, encoding="utf-8")
    print(f"[TOKEN] Wrote feature spec: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build numeric event-level tokens for DataFest patient timelines.")
    p.add_argument("--input", type=Path, default=Path("data/processed/event_enriched.csv.gz"), help="Path to event_enriched CSV/CSV.GZ")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Directory for token outputs")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"), help="Directory for tokenization_feature_spec.md")
    p.add_argument("--work-db", type=Path, default=Path("data/interim/event_tokenization.sqlite"), help="SQLite work DB path")
    p.add_argument("--chunksize", type=int, default=100_000, help="CSV/SQL chunk size")
    p.add_argument("--min-event-token-frequency", type=int, default=1, help="Map rarer composite event tokens to [UNK] if frequency is below this")
    p.add_argument("--max-seq-len", type=int, default=0, help="If >0, create padded model input arrays of this length")
    p.add_argument("--replace", action="store_true", help="Delete existing work DB before running")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}\nRun the main pipeline first or pass --input to an existing event_enriched file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_db.parent.mkdir(parents=True, exist_ok=True)
    if args.replace and args.work_db.exists():
        args.work_db.unlink()

    with sqlite3.connect(args.work_db) as conn:
        missing_optional_columns = load_event_stage(conn, args.input, args.chunksize)
        build_event_work(conn)
        token_to_id = build_vocabulary(conn, args.output_dir, args.chunksize, args.min_event_token_frequency)
        build_event_tokenized(conn, args.output_dir, token_to_id, args.chunksize)
        build_patient_sequences(conn, args.output_dir)

    build_model_inputs(args.output_dir, args.max_seq_len, args.chunksize)
    write_feature_spec(args.docs_dir, args.input, missing_optional_columns)
    print("[TOKEN] Done.")


if __name__ == "__main__":
    main()
