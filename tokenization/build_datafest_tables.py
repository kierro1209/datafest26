#!/usr/bin/env python3
"""
Build DataFest analysis tables from raw CSVs.

Pipeline outputs:
  Level 1: dim/fact base tables
  Level 2: encounter_enriched, sdoh_encounter_summary, encounter_enriched_with_sdoh
  Level 3: patient timeline/tokenization tables and supporting diagnosis journey tables

Usage from project root:
  python src/build_datafest_tables.py --raw-dir data/raw --export

This script uses only Python stdlib + pandas + SQLite. It keeps joins on disk in
SQLite so the 7.6M-row encounters file does not require one huge in-memory join.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from datafest_config import (
        DB_PATH,
        EXPECTED_COLUMNS,
        FINAL_TABLES,
        INTERIM_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        RAW_FILES,
        SDOH_DOMAINS,
    )
except ImportError:  # allow running from outside src/ with python /path/to/script.py
    sys.path.append(str(Path(__file__).resolve().parent))
    from datafest_config import (  # type: ignore
        DB_PATH,
        EXPECTED_COLUMNS,
        FINAL_TABLES,
        INTERIM_DIR,
        PROCESSED_DIR,
        RAW_DIR,
        RAW_FILES,
        SDOH_DOMAINS,
    )


# ----------------------------- basic utilities -----------------------------

def qident(name: str) -> str:
    """Quote a SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


def run(conn: sqlite3.Connection, sql: str, label: str | None = None) -> None:
    """Execute a SQL script and commit."""
    if label:
        print(f"[SQL] {label}")
    conn.executescript(sql)
    conn.commit()


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def drop_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {qident(table)}")
    conn.commit()


def get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({qident(table)})")]


def normalize_missing_expr(col: str) -> str:
    """SQL expression that maps source values into broad missingness classes."""
    c = qident(col)
    return f"""
    CASE
      WHEN {c} IS NULL OR TRIM({c}) = '' OR UPPER(TRIM({c})) = 'NA' THEN 'system_missing'
      WHEN TRIM({c}) IN ('*Unspecified','*Unknown','*Not Applicable') THEN 'asked_not_answered_or_unable'
      WHEN TRIM({c}) IN ('Unspecified','Unknown') THEN 'not_recorded_or_unknown'
      WHEN TRIM({c}) = 'Not Applicable' THEN 'structural_not_applicable'
      ELSE 'known_value'
    END
    """


def true_expr(col: str) -> str:
    """SQL boolean-ish true expression for source flag fields stored as text."""
    c = f"COALESCE({qident(col)}, '')"
    return f"LOWER(TRIM({c})) IN ('1','true','t','yes','y')"


def safe_token_expr(col: str) -> str:
    return f"COALESCE(NULLIF(TRIM({qident(col)}), ''), 'MISSING')"


def ensure_dirs(raw_dir: Path, interim_dir: Path, processed_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


# --------------------------- schema + staging load --------------------------

def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader)


def validate_raw_schema(raw_dir: Path) -> None:
    print("[CHECK] Validating raw CSV schemas")
    errors: list[str] = []
    for table, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            errors.append(f"Missing required file: {path}")
            continue
        observed = read_header(path)
        expected = EXPECTED_COLUMNS[table]
        if observed != expected:
            missing = [c for c in expected if c not in observed]
            extra = [c for c in observed if c not in expected]
            errors.append(
                f"Schema mismatch for {filename}\n"
                f"  Expected: {expected}\n"
                f"  Observed: {observed}\n"
                f"  Missing: {missing}\n"
                f"  Extra: {extra}"
            )
    if errors:
        raise SystemExit("\n\n".join(errors))
    print("[CHECK] Raw schemas match expected columns")


def load_csv_to_sqlite(
    conn: sqlite3.Connection,
    raw_dir: Path,
    table: str,
    chunksize: int,
    replace: bool,
) -> None:
    if table_exists(conn, table) and not replace:
        print(f"[LOAD] Skipping existing raw table: {table}")
        return
    drop_table(conn, table)
    path = raw_dir / RAW_FILES[table]
    print(f"[LOAD] {path} -> SQLite table {table}")

    total = 0
    for chunk in pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        # Column order has already been validated. Keep exact names.
        chunk.to_sql(table, conn, if_exists="append", index=False)
        total += len(chunk)
        print(f"       loaded {total:,} rows", end="\r")
    print(f"       loaded {total:,} rows")
    conn.commit()


def load_all_raw(conn: sqlite3.Connection, raw_dir: Path, chunksize: int, replace: bool) -> None:
    for table in RAW_FILES:
        load_csv_to_sqlite(conn, raw_dir, table, chunksize, replace)


# ----------------------------- indexes --------------------------------------

def create_indexes(conn: sqlite3.Connection) -> None:
    print("[INDEX] Creating indexes")
    run(
        conn,
        """
        CREATE INDEX IF NOT EXISTS idx_patients_DurableKey
          ON patients(DurableKey);
        CREATE INDEX IF NOT EXISTS idx_diagnosis_DiagnosisKey
          ON diagnosis(DiagnosisKey);
        CREATE INDEX IF NOT EXISTS idx_departments_DepartmentKey
          ON departments(DepartmentKey);
        CREATE INDEX IF NOT EXISTS idx_providers_DurableKey
          ON providers(DurableKey);
        CREATE INDEX IF NOT EXISTS idx_tiger_GEOID
          ON tigercensuscodes(GEOID);

        CREATE INDEX IF NOT EXISTS idx_encounters_EncounterKey
          ON encounters(EncounterKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_PatientDurableKey
          ON encounters(PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_PrimaryDiagnosisKey
          ON encounters(PrimaryDiagnosisKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_DepartmentKey
          ON encounters(DepartmentKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_ProviderDurableKey
          ON encounters(ProviderDurableKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_AttendingProviderDurableKey
          ON encounters(AttendingProviderDurableKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_DischargeProviderDurableKey
          ON encounters(DischargeProviderDurableKey);
        CREATE INDEX IF NOT EXISTS idx_encounters_patient_time
          ON encounters(PatientDurableKey, Date, AdmissionInstant, EncounterKey);

        CREATE INDEX IF NOT EXISTS idx_sdoh_encounter_patient
          ON social_determinants(EncounterKey, PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_sdoh_patient
          ON social_determinants(PatientDurableKey);
        """,
    )


# ----------------------------- level 1 --------------------------------------

def build_level1(conn: sqlite3.Connection) -> None:
    print("[BUILD] Level 1 dim/fact tables")

    run(conn, "DROP TABLE IF EXISTS dim_patient;")
    run(
        conn,
        f"""
        CREATE TABLE dim_patient AS
        SELECT
          DurableKey,
          CensusBlockGroupFipsCode,
          FirstRace,
          MaritalStatus,
          MyChartStatus,
          OmbEthnicity,
          OmbRace,
          SexAssignedAtBirth,
          SexualOrientation,
          SmokingStatus,
          VitalStatus,
          PatientBirthYearBin,
          CASE
            WHEN CensusBlockGroupFipsCode IS NULL
              OR TRIM(CensusBlockGroupFipsCode) = ''
              OR TRIM(CensusBlockGroupFipsCode) IN ('*Unspecified','Unspecified','*Unknown','Unknown','NA')
            THEN 0 ELSE 1
          END AS patient_geography_known_flag,
          {normalize_missing_expr('PatientBirthYearBin')} AS patient_birth_year_bin_missing_class,
          {normalize_missing_expr('MyChartStatus')} AS mychart_status_missing_class,
          {normalize_missing_expr('SmokingStatus')} AS smoking_status_missing_class
        FROM patients;
        CREATE INDEX IF NOT EXISTS idx_dim_patient_DurableKey ON dim_patient(DurableKey);
        """,
        "dim_patient",
    )

    run(conn, "DROP TABLE IF EXISTS dim_diagnosis;")
    run(
        conn,
        """
        CREATE TABLE dim_diagnosis AS
        SELECT DiagnosisKey, GroupName, GroupCode, DiagnosisName, DiagnosisValue
        FROM diagnosis;
        CREATE INDEX IF NOT EXISTS idx_dim_diagnosis_DiagnosisKey ON dim_diagnosis(DiagnosisKey);
        CREATE INDEX IF NOT EXISTS idx_dim_diagnosis_DiagnosisValue ON dim_diagnosis(DiagnosisValue);
        CREATE INDEX IF NOT EXISTS idx_dim_diagnosis_GroupCode ON dim_diagnosis(GroupCode);
        """,
        "dim_diagnosis",
    )

    run(conn, "DROP TABLE IF EXISTS dim_department;")
    run(
        conn,
        """
        CREATE TABLE dim_department AS
        SELECT
          DepartmentKey, Address, City, County, DepartmentName,
          DepartmentSpecialty, DepartmentType, PostalCode, CensusTract
        FROM departments;
        CREATE INDEX IF NOT EXISTS idx_dim_department_DepartmentKey ON dim_department(DepartmentKey);
        """,
        "dim_department",
    )

    run(conn, "DROP TABLE IF EXISTS dim_provider;")
    run(
        conn,
        """
        CREATE TABLE dim_provider AS
        SELECT
          DurableKey, ClinicianTitle, OfficeAddress, OfficeCity,
          OfficePostalCode, PrimaryDepartment, PrimarySpecialty, Type
        FROM providers;
        CREATE INDEX IF NOT EXISTS idx_dim_provider_DurableKey ON dim_provider(DurableKey);
        """,
        "dim_provider",
    )

    run(conn, "DROP TABLE IF EXISTS dim_geography;")
    run(
        conn,
        """
        CREATE TABLE dim_geography AS
        SELECT
          GEOID,
          PopulationValue,
          CENTLAT,
          CENTLON,
          CASE
            WHEN PopulationValue IS NULL OR TRIM(PopulationValue) = '' THEN NULL
            WHEN CAST(PopulationValue AS REAL) < 500 THEN 'POP_LT_500'
            WHEN CAST(PopulationValue AS REAL) < 1500 THEN 'POP_500_1499'
            WHEN CAST(PopulationValue AS REAL) < 3000 THEN 'POP_1500_2999'
            ELSE 'POP_3000_PLUS'
          END AS patient_block_population_bin
        FROM tigercensuscodes;
        CREATE INDEX IF NOT EXISTS idx_dim_geography_GEOID ON dim_geography(GEOID);
        """,
        "dim_geography",
    )

    run(conn, "DROP TABLE IF EXISTS fact_encounter_base;")
    encounter_cols = ",\n          ".join(qident(c) for c in EXPECTED_COLUMNS["encounters"])
    run(
        conn,
        f"""
        CREATE TABLE fact_encounter_base AS
        SELECT
          {encounter_cols},
          {normalize_missing_expr('PrimaryDiagnosisKey')} AS primary_diagnosis_missing_class,
          {normalize_missing_expr('DepartmentKey')} AS department_key_missing_class,
          {normalize_missing_expr('ProviderDurableKey')} AS provider_key_missing_class,
          {normalize_missing_expr('AttendingProviderDurableKey')} AS attending_provider_key_missing_class,
          {normalize_missing_expr('DischargeProviderDurableKey')} AS discharge_provider_key_missing_class
        FROM encounters;
        CREATE INDEX IF NOT EXISTS idx_fact_encounter_EncounterKey ON fact_encounter_base(EncounterKey);
        CREATE INDEX IF NOT EXISTS idx_fact_encounter_patient_time ON fact_encounter_base(PatientDurableKey, Date, AdmissionInstant, EncounterKey);
        """,
        "fact_encounter_base",
    )

    run(conn, "DROP TABLE IF EXISTS fact_sdoh_response;")
    run(
        conn,
        f"""
        CREATE TABLE fact_sdoh_response AS
        SELECT
          DisplayName,
          AnswerText,
          EncounterKey,
          PatientDurableKey,
          Domain,
          {normalize_missing_expr('AnswerText')} AS answer_missing_class,
          'SDOH:' || {safe_token_expr('Domain')} || ':' || {safe_token_expr('DisplayName')} || ':' || {safe_token_expr('AnswerText')}
            AS sdoh_answer_token
        FROM social_determinants;
        CREATE INDEX IF NOT EXISTS idx_fact_sdoh_encounter_patient ON fact_sdoh_response(EncounterKey, PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_fact_sdoh_patient ON fact_sdoh_response(PatientDurableKey);
        """,
        "fact_sdoh_response",
    )


# ----------------------------- level 2 --------------------------------------

def build_encounter_enriched(conn: sqlite3.Connection) -> None:
    print("[BUILD] encounter_enriched")
    run(conn, "DROP TABLE IF EXISTS encounter_enriched;")
    run(
        conn,
        """
        CREATE TABLE encounter_enriched AS
        SELECT
          e.EncounterKey,
          e.PatientDurableKey,
          e.Date,
          e.AdmissionInstant,
          e.AdmitYear,
          e.AdmitMonth,
          e.AdmitDay,
          e.AdmitHour,
          e.AdmitMinute,
          e.AdmissionSource,
          e.AdmissionType,
          e.DischargeInstant,
          e.DischargeYear,
          e.DischargeMonth,
          e.DischargeDay,
          e.DischargeHour,
          e.DischargeMinute,
          e.Type,
          e.VisitType,
          e.VisitTypeDescription,
          e.ProviderDurableKey,
          e.AttendingProviderDurableKey,
          e.DischargeProviderDurableKey,
          e.DepartmentKey,
          e.PrimaryDiagnosisKey,
          e.IsEdVisit,
          e.IsHospitalAdmission,
          e.IsHospitalOutpatientVisit,
          e.IsInpatientAdmission,
          e.IsObservation,
          e.IsOutpatientFaceToFaceVisit,
          e.primary_diagnosis_missing_class,
          e.department_key_missing_class,
          e.provider_key_missing_class,
          e.attending_provider_key_missing_class,
          e.discharge_provider_key_missing_class,

          p.DurableKey AS patient_DurableKey,
          p.CensusBlockGroupFipsCode,
          p.FirstRace,
          p.MaritalStatus,
          p.MyChartStatus,
          p.OmbEthnicity,
          p.OmbRace,
          p.SexAssignedAtBirth,
          p.SexualOrientation,
          p.SmokingStatus,
          p.VitalStatus,
          p.PatientBirthYearBin,
          p.patient_geography_known_flag,
          p.patient_birth_year_bin_missing_class,
          p.mychart_status_missing_class,
          p.smoking_status_missing_class,

          d.DiagnosisKey,
          d.GroupName,
          d.GroupCode,
          d.DiagnosisName,
          d.DiagnosisValue,

          dept.DepartmentName,
          dept.DepartmentSpecialty,
          dept.DepartmentType,
          dept.Address AS department_Address,
          dept.City AS department_City,
          dept.County AS department_County,
          dept.PostalCode AS department_PostalCode,
          dept.CensusTract AS department_CensusTract,

          prov.ClinicianTitle AS provider_ClinicianTitle,
          prov.PrimarySpecialty AS provider_PrimarySpecialty,
          prov.Type AS provider_Type,
          prov.PrimaryDepartment AS provider_PrimaryDepartment,
          prov.OfficeCity AS provider_OfficeCity,
          prov.OfficePostalCode AS provider_OfficePostalCode,

          att.ClinicianTitle AS attending_provider_ClinicianTitle,
          att.PrimarySpecialty AS attending_provider_PrimarySpecialty,
          att.Type AS attending_provider_Type,
          att.PrimaryDepartment AS attending_provider_PrimaryDepartment,
          att.OfficeCity AS attending_provider_OfficeCity,
          att.OfficePostalCode AS attending_provider_OfficePostalCode,

          dis.ClinicianTitle AS discharge_provider_ClinicianTitle,
          dis.PrimarySpecialty AS discharge_provider_PrimarySpecialty,
          dis.Type AS discharge_provider_Type,
          dis.PrimaryDepartment AS discharge_provider_PrimaryDepartment,
          dis.OfficeCity AS discharge_provider_OfficeCity,
          dis.OfficePostalCode AS discharge_provider_OfficePostalCode,

          geo.GEOID,
          geo.PopulationValue,
          geo.CENTLAT,
          geo.CENTLON,
          geo.patient_block_population_bin
        FROM fact_encounter_base e
        LEFT JOIN dim_patient p
          ON e.PatientDurableKey = p.DurableKey
        LEFT JOIN dim_diagnosis d
          ON e.PrimaryDiagnosisKey = d.DiagnosisKey
        LEFT JOIN dim_department dept
          ON e.DepartmentKey = dept.DepartmentKey
        LEFT JOIN dim_provider prov
          ON e.ProviderDurableKey = prov.DurableKey
        LEFT JOIN dim_provider att
          ON e.AttendingProviderDurableKey = att.DurableKey
        LEFT JOIN dim_provider dis
          ON e.DischargeProviderDurableKey = dis.DurableKey
        LEFT JOIN dim_geography geo
          ON p.CensusBlockGroupFipsCode = geo.GEOID;

        CREATE INDEX IF NOT EXISTS idx_enc_enriched_EncounterKey ON encounter_enriched(EncounterKey);
        CREATE INDEX IF NOT EXISTS idx_enc_enriched_patient_time ON encounter_enriched(PatientDurableKey, Date, AdmissionInstant, EncounterKey);
        CREATE INDEX IF NOT EXISTS idx_enc_enriched_diag ON encounter_enriched(PatientDurableKey, DiagnosisValue, Date, AdmissionInstant, EncounterKey);
        """,
    )


def build_sdoh_summary(conn: sqlite3.Connection) -> None:
    print("[BUILD] sdoh_encounter_summary")
    run(conn, "DROP TABLE IF EXISTS sdoh_encounter_summary;")

    domain_cols = []
    for domain in SDOH_DOMAINS:
        alias = "sdoh_" + "".join(ch for ch in domain if ch.isalnum()) + "_observed"
        domain_cols.append(
            f"MAX(CASE WHEN Domain = {repr(domain)} THEN 1 ELSE 0 END) AS {qident(alias)}"
        )
    domain_sql = ",\n          ".join(domain_cols)

    run(
        conn,
        f"""
        CREATE TABLE sdoh_encounter_summary AS
        SELECT
          EncounterKey,
          PatientDurableKey,
          1 AS sdoh_any_observed,
          COUNT(*) AS sdoh_num_questions_answered,
          COUNT(DISTINCT Domain) AS sdoh_num_domains_answered,
          {domain_sql},
          GROUP_CONCAT(sdoh_answer_token, ' || ') AS sdoh_domain_answer_tokens,
          GROUP_CONCAT(DISTINCT Domain) AS sdoh_domains_observed
        FROM fact_sdoh_response
        GROUP BY EncounterKey, PatientDurableKey;

        CREATE INDEX IF NOT EXISTS idx_sdoh_summary_encounter_patient
          ON sdoh_encounter_summary(EncounterKey, PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_sdoh_summary_patient
          ON sdoh_encounter_summary(PatientDurableKey);
        """,
    )


def build_encounter_with_sdoh(conn: sqlite3.Connection) -> None:
    print("[BUILD] encounter_enriched_with_sdoh")
    run(conn, "DROP TABLE IF EXISTS encounter_enriched_with_sdoh;")

    zero_domain_cols = []
    for domain in SDOH_DOMAINS:
        alias = "sdoh_" + "".join(ch for ch in domain if ch.isalnum()) + "_observed"
        zero_domain_cols.append(f"COALESCE(s.{qident(alias)}, 0) AS {qident(alias)}")
    zero_domain_sql = ",\n          ".join(zero_domain_cols)

    run(
        conn,
        f"""
        CREATE TABLE encounter_enriched_with_sdoh AS
        SELECT
          e.*,
          COALESCE(s.sdoh_any_observed, 0) AS sdoh_any_observed,
          COALESCE(s.sdoh_num_questions_answered, 0) AS sdoh_num_questions_answered,
          COALESCE(s.sdoh_num_domains_answered, 0) AS sdoh_num_domains_answered,
          {zero_domain_sql},
          s.sdoh_domain_answer_tokens,
          s.sdoh_domains_observed
        FROM encounter_enriched e
        LEFT JOIN sdoh_encounter_summary s
          ON e.EncounterKey = s.EncounterKey
         AND e.PatientDurableKey = s.PatientDurableKey;

        CREATE INDEX IF NOT EXISTS idx_enc_sdoh_EncounterKey ON encounter_enriched_with_sdoh(EncounterKey);
        CREATE INDEX IF NOT EXISTS idx_enc_sdoh_patient_time ON encounter_enriched_with_sdoh(PatientDurableKey, Date, AdmissionInstant, EncounterKey);
        CREATE INDEX IF NOT EXISTS idx_enc_sdoh_diag ON encounter_enriched_with_sdoh(PatientDurableKey, DiagnosisValue, Date, AdmissionInstant, EncounterKey);
        """,
    )




def build_event_enriched(conn: sqlite3.Connection) -> None:
    """
    Build the long event-level joined dataset used by numeric event tokenization.

    Grain:
      - ENCOUNTER rows: one event row per encounter.
      - SDOH_RESPONSE rows: one event row per answered SDOH question.

    Each event row carries the full joined context from encounter_enriched_with_sdoh:
    patient info, encounter info, diagnosis, department/location, provider roles,
    geography, and SDOH summary. SDOH_RESPONSE rows additionally carry the raw
    SDOH question/answer in event_domain/event_value/event_description.
    """
    print("[BUILD] event_enriched")
    run(conn, "DROP TABLE IF EXISTS event_enriched;")
    run(
        conn,
        """
        CREATE TABLE event_enriched AS
        WITH encounter_events AS (
          SELECT
            'ENC|' || COALESCE(EncounterKey, '') AS event_id,
            'ENCOUNTER' AS event_source,
            'ENCOUNTER' AS event_grain,
            EncounterKey AS event_EncounterKey,
            PatientDurableKey AS event_PatientDurableKey,
            Date AS event_date,
            COALESCE(NULLIF(TRIM(AdmissionInstant), ''), Date) AS event_time,
            0 AS event_index_within_encounter,
            Type AS event_type,
            VisitType AS event_subtype,
            VisitTypeDescription AS event_description,
            NULL AS event_domain,
            NULL AS event_value,
            NULL AS event_display_name,
            NULL AS event_answer_text,
            NULL AS event_sdoh_answer_token,
            e.*,
            e.department_City AS City,
            e.department_County AS County,
            e.department_PostalCode AS PostalCode,
            e.department_CensusTract AS CensusTract
          FROM encounter_enriched_with_sdoh e
        ), sdoh_indexed AS (
          SELECT
            s.*,
            ROW_NUMBER() OVER (
              PARTITION BY s.EncounterKey, s.PatientDurableKey
              ORDER BY s.Domain, s.DisplayName, s.AnswerText, s.sdoh_answer_token
            ) AS sdoh_event_index
          FROM fact_sdoh_response s
        ), sdoh_events AS (
          SELECT
            'SDOH|' || COALESCE(s.EncounterKey, '') || '|' ||
              COALESCE(s.PatientDurableKey, '') || '|' || s.sdoh_event_index AS event_id,
            'SDOH_RESPONSE' AS event_source,
            'SDOH_RESPONSE' AS event_grain,
            s.EncounterKey AS event_EncounterKey,
            s.PatientDurableKey AS event_PatientDurableKey,
            e.Date AS event_date,
            COALESCE(NULLIF(TRIM(e.AdmissionInstant), ''), e.Date) AS event_time,
            s.sdoh_event_index AS event_index_within_encounter,
            'SDOH_RESPONSE' AS event_type,
            s.DisplayName AS event_subtype,
            s.DisplayName AS event_description,
            s.Domain AS event_domain,
            s.AnswerText AS event_value,
            s.DisplayName AS event_display_name,
            s.AnswerText AS event_answer_text,
            s.sdoh_answer_token AS event_sdoh_answer_token,
            e.*,
            e.department_City AS City,
            e.department_County AS County,
            e.department_PostalCode AS PostalCode,
            e.department_CensusTract AS CensusTract
          FROM sdoh_indexed s
          LEFT JOIN encounter_enriched_with_sdoh e
            ON s.EncounterKey = e.EncounterKey
           AND s.PatientDurableKey = e.PatientDurableKey
        )
        SELECT * FROM encounter_events
        UNION ALL
        SELECT * FROM sdoh_events;

        CREATE INDEX IF NOT EXISTS idx_event_enriched_event_id
          ON event_enriched(event_id);
        CREATE INDEX IF NOT EXISTS idx_event_enriched_patient_time
          ON event_enriched(event_PatientDurableKey, event_date, event_time, event_EncounterKey, event_index_within_encounter, event_id);
        CREATE INDEX IF NOT EXISTS idx_event_enriched_encounter_patient
          ON event_enriched(event_EncounterKey, event_PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_event_enriched_source
          ON event_enriched(event_source);
        """,
        "event_enriched",
    )


# ----------------------------- level 3 patient timeline ---------------------

def build_patient_timeline_event(conn: sqlite3.Connection) -> None:
    print("[BUILD] patient_timeline_event")
    run(conn, "DROP TABLE IF EXISTS patient_timeline_event;")

    setting_expr = f"""
    TRIM(
      (CASE WHEN {true_expr('IsEdVisit')} THEN 'ED;' ELSE '' END) ||
      (CASE WHEN {true_expr('IsHospitalAdmission')} THEN 'HOSP_ADMIT;' ELSE '' END) ||
      (CASE WHEN {true_expr('IsHospitalOutpatientVisit')} THEN 'HOSP_OP;' ELSE '' END) ||
      (CASE WHEN {true_expr('IsInpatientAdmission')} THEN 'INPATIENT;' ELSE '' END) ||
      (CASE WHEN {true_expr('IsObservation')} THEN 'OBS;' ELSE '' END) ||
      (CASE WHEN {true_expr('IsOutpatientFaceToFaceVisit')} THEN 'OP_FACE;' ELSE '' END)
    , ';')
    """

    run(
        conn,
        f"""
        CREATE TABLE patient_timeline_event AS
        WITH ordered AS (
          SELECT
            e.*,
            ROW_NUMBER() OVER (
              PARTITION BY PatientDurableKey
              ORDER BY Date, AdmissionInstant, EncounterKey
            ) AS patient_event_index,
            LAG(Date) OVER (
              PARTITION BY PatientDurableKey
              ORDER BY Date, AdmissionInstant, EncounterKey
            ) AS previous_patient_event_date
          FROM encounter_enriched_with_sdoh e
        ), gaps AS (
          SELECT
            *,
            CASE
              WHEN previous_patient_event_date IS NULL THEN NULL
              WHEN julianday(Date) IS NULL OR julianday(previous_patient_event_date) IS NULL THEN NULL
              ELSE CAST(julianday(Date) - julianday(previous_patient_event_date) AS INTEGER)
            END AS days_since_previous_patient_event
          FROM ordered
        ), tokens AS (
          SELECT
            *,
            CASE
              WHEN patient_event_index = 1 THEN 'GAP:START'
              WHEN days_since_previous_patient_event IS NULL THEN 'GAP:UNKNOWN'
              WHEN days_since_previous_patient_event = 0 THEN 'GAP:0D'
              WHEN days_since_previous_patient_event BETWEEN 1 AND 7 THEN 'GAP:1_7D'
              WHEN days_since_previous_patient_event BETWEEN 8 AND 30 THEN 'GAP:8_30D'
              WHEN days_since_previous_patient_event BETWEEN 31 AND 90 THEN 'GAP:31_90D'
              WHEN days_since_previous_patient_event BETWEEN 91 AND 180 THEN 'GAP:91_180D'
              WHEN days_since_previous_patient_event BETWEEN 181 AND 365 THEN 'GAP:181_365D'
              WHEN days_since_previous_patient_event > 365 THEN 'GAP:365PLUS'
              ELSE 'GAP:UNKNOWN'
            END AS gap_bin,
            {setting_expr} AS setting_flags
          FROM gaps
        )
        SELECT
          *,
          gap_bin AS time_gap_token,
          'DXG:' || COALESCE(NULLIF(TRIM(GroupCode), ''), 'MISSING') || '|DX:' || COALESCE(NULLIF(TRIM(DiagnosisValue), ''), 'MISSING')
            AS diagnosis_token,
          'DEPT_TYPE:' || COALESCE(NULLIF(TRIM(DepartmentType), ''), 'MISSING') || '|DEPT_SPEC:' || COALESCE(NULLIF(TRIM(DepartmentSpecialty), ''), 'MISSING')
            AS department_token,
          'SETTING:' || COALESCE(NULLIF(TRIM(setting_flags), ''), 'NONE')
            AS care_setting_token,
          CASE
            WHEN sdoh_any_observed = 1 THEN 'SDOH:' || COALESCE(sdoh_domains_observed, 'OBSERVED')
            ELSE 'SDOH:NONE_OBSERVED'
          END AS sdoh_token,
          gap_bin ||
            '|TYPE:' || COALESCE(NULLIF(TRIM(Type), ''), 'MISSING') ||
            '|VTD:' || COALESCE(NULLIF(TRIM(VisitTypeDescription), ''), 'MISSING') ||
            '|DEPT_TYPE:' || COALESCE(NULLIF(TRIM(DepartmentType), ''), 'MISSING') ||
            '|DXG:' || COALESCE(NULLIF(TRIM(GroupCode), ''), 'MISSING') ||
            '|SETTING:' || COALESCE(NULLIF(TRIM(setting_flags), ''), 'NONE') ||
            '|SDOH:' || CASE WHEN sdoh_any_observed = 1 THEN 'OBSERVED' ELSE 'NONE' END
            AS event_token
        FROM tokens;

        CREATE INDEX IF NOT EXISTS idx_patient_timeline_event_patient_idx
          ON patient_timeline_event(PatientDurableKey, patient_event_index);
        CREATE INDEX IF NOT EXISTS idx_patient_timeline_event_encounter
          ON patient_timeline_event(EncounterKey);
        """,
    )


def build_patient_timeline_sequence(conn: sqlite3.Connection) -> None:
    print("[BUILD] patient_timeline_sequence")
    run(conn, "DROP TABLE IF EXISTS patient_timeline_sequence;")
    run(
        conn,
        """
        CREATE TABLE patient_timeline_sequence AS
        WITH ordered AS (
          SELECT *
          FROM patient_timeline_event
          ORDER BY PatientDurableKey, patient_event_index
        )
        SELECT
          PatientDurableKey,
          MIN(Date) AS first_observed_event_date,
          MAX(Date) AS last_observed_event_date,
          CASE
            WHEN julianday(MAX(Date)) IS NULL OR julianday(MIN(Date)) IS NULL THEN NULL
            ELSE CAST(julianday(MAX(Date)) - julianday(MIN(Date)) AS INTEGER)
          END AS observed_history_duration_days,
          COUNT(*) AS n_events,
          COUNT(DISTINCT DiagnosisValue) AS n_unique_DiagnosisValue,
          COUNT(DISTINCT GroupCode) AS n_unique_GroupCode,
          COUNT(DISTINCT DepartmentType) AS n_unique_DepartmentType,
          COUNT(DISTINCT DepartmentSpecialty) AS n_unique_DepartmentSpecialty,
          COUNT(DISTINCT Type) AS n_unique_Type,
          'PATIENT_START || ' || GROUP_CONCAT(event_token, ' || ') || ' || PATIENT_END' AS full_event_token_sequence,
          GROUP_CONCAT(time_gap_token, ' || ') AS full_gap_token_sequence,
          GROUP_CONCAT(diagnosis_token, ' || ') AS full_diagnosis_token_sequence,
          GROUP_CONCAT(department_token, ' || ') AS full_department_token_sequence,
          GROUP_CONCAT(care_setting_token, ' || ') AS full_setting_token_sequence,
          GROUP_CONCAT(sdoh_token, ' || ') AS full_sdoh_token_sequence
        FROM ordered
        GROUP BY PatientDurableKey;

        CREATE INDEX IF NOT EXISTS idx_patient_timeline_sequence_patient
          ON patient_timeline_sequence(PatientDurableKey);
        """,
    )


def build_patient_timeline_transition(conn: sqlite3.Connection) -> None:
    print("[BUILD] patient_timeline_transition")
    run(conn, "DROP TABLE IF EXISTS patient_timeline_transition;")
    run(
        conn,
        """
        CREATE TABLE patient_timeline_transition AS
        SELECT
          a.PatientDurableKey,
          a.EncounterKey AS from_EncounterKey,
          b.EncounterKey AS to_EncounterKey,
          a.patient_event_index AS from_patient_event_index,
          b.patient_event_index AS to_patient_event_index,
          a.Date AS from_Date,
          b.Date AS to_Date,
          b.days_since_previous_patient_event AS gap_days,
          b.gap_bin AS gap_bin,
          a.Type AS from_Type,
          b.Type AS to_Type,
          a.VisitTypeDescription AS from_VisitTypeDescription,
          b.VisitTypeDescription AS to_VisitTypeDescription,
          a.GroupCode AS from_GroupCode,
          b.GroupCode AS to_GroupCode,
          a.DiagnosisValue AS from_DiagnosisValue,
          b.DiagnosisValue AS to_DiagnosisValue,
          a.DepartmentType AS from_DepartmentType,
          b.DepartmentType AS to_DepartmentType,
          a.DepartmentSpecialty AS from_DepartmentSpecialty,
          b.DepartmentSpecialty AS to_DepartmentSpecialty,
          a.event_token AS from_event_token,
          b.event_token AS to_event_token,
          'TRANSITION:TYPE:' || COALESCE(NULLIF(TRIM(a.Type), ''), 'MISSING') ||
            '_TO_TYPE:' || COALESCE(NULLIF(TRIM(b.Type), ''), 'MISSING') AS transition_token
        FROM patient_timeline_event a
        JOIN patient_timeline_event b
          ON a.PatientDurableKey = b.PatientDurableKey
         AND b.patient_event_index = a.patient_event_index + 1;

        CREATE INDEX IF NOT EXISTS idx_patient_timeline_transition_patient
          ON patient_timeline_transition(PatientDurableKey, from_patient_event_index);
        """,
    )


def build_patient_timeline_summary(conn: sqlite3.Connection) -> None:
    print("[BUILD] patient_timeline_summary")
    run(conn, "DROP TABLE IF EXISTS patient_timeline_summary;")
    run(
        conn,
        """
        CREATE TABLE patient_timeline_summary AS
        WITH event_agg AS (
          SELECT
            PatientDurableKey,
            MIN(Date) AS first_observed_event_date,
            MAX(Date) AS last_observed_event_date,
            CASE
              WHEN julianday(MAX(Date)) IS NULL OR julianday(MIN(Date)) IS NULL THEN NULL
              ELSE CAST(julianday(MAX(Date)) - julianday(MIN(Date)) AS INTEGER)
            END AS observed_history_duration_days,
            COUNT(*) AS n_events,
            COUNT(DISTINCT DiagnosisValue) AS n_unique_DiagnosisValue,
            COUNT(DISTINCT GroupCode) AS n_unique_GroupCode,
            COUNT(DISTINCT DepartmentType) AS n_unique_DepartmentType,
            COUNT(DISTINCT DepartmentSpecialty) AS n_unique_DepartmentSpecialty,
            COUNT(DISTINCT Type) AS n_unique_Type,
            SUM(CASE WHEN LOWER(TRIM(COALESCE(IsEdVisit,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_ed_visits,
            SUM(CASE WHEN LOWER(TRIM(COALESCE(IsHospitalAdmission,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_hospital_admissions,
            SUM(CASE WHEN LOWER(TRIM(COALESCE(IsHospitalOutpatientVisit,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_hospital_outpatient_visits,
            SUM(CASE WHEN LOWER(TRIM(COALESCE(IsInpatientAdmission,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_inpatient_admissions,
            SUM(CASE WHEN LOWER(TRIM(COALESCE(IsObservation,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_observations,
            SUM(CASE WHEN LOWER(TRIM(COALESCE(IsOutpatientFaceToFaceVisit,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_outpatient_face_to_face_visits,
            MAX(sdoh_any_observed) AS sdoh_any_observed,
            MAX(sdoh_num_domains_answered) AS sdoh_num_domains_observed,
            MIN(CASE WHEN sdoh_any_observed = 1 THEN Date ELSE NULL END) AS sdoh_first_observed_date,
            MAX(CASE WHEN sdoh_any_observed = 1 THEN Date ELSE NULL END) AS sdoh_last_observed_date
          FROM patient_timeline_event
          GROUP BY PatientDurableKey
        ), gap_agg AS (
          SELECT
            PatientDurableKey,
            AVG(days_since_previous_patient_event) AS mean_gap_days,
            MAX(days_since_previous_patient_event) AS max_gap_days,
            SUM(CASE WHEN days_since_previous_patient_event >= 30 THEN 1 ELSE 0 END) AS n_long_gaps_30d,
            SUM(CASE WHEN days_since_previous_patient_event >= 90 THEN 1 ELSE 0 END) AS n_long_gaps_90d,
            SUM(CASE WHEN days_since_previous_patient_event >= 180 THEN 1 ELSE 0 END) AS n_long_gaps_180d,
            SUM(CASE WHEN days_since_previous_patient_event >= 365 THEN 1 ELSE 0 END) AS n_long_gaps_365d
          FROM patient_timeline_event
          WHERE patient_event_index > 1
          GROUP BY PatientDurableKey
        ), dominant_group AS (
          SELECT PatientDurableKey, GroupCode AS dominant_GroupCode
          FROM (
            SELECT
              PatientDurableKey, GroupCode, COUNT(*) AS n,
              ROW_NUMBER() OVER (PARTITION BY PatientDurableKey ORDER BY COUNT(*) DESC, GroupCode) AS rn
            FROM patient_timeline_event
            WHERE GroupCode IS NOT NULL AND TRIM(GroupCode) <> ''
            GROUP BY PatientDurableKey, GroupCode
          )
          WHERE rn = 1
        ), dominant_dept AS (
          SELECT PatientDurableKey, DepartmentType AS dominant_DepartmentType
          FROM (
            SELECT
              PatientDurableKey, DepartmentType, COUNT(*) AS n,
              ROW_NUMBER() OVER (PARTITION BY PatientDurableKey ORDER BY COUNT(*) DESC, DepartmentType) AS rn
            FROM patient_timeline_event
            WHERE DepartmentType IS NOT NULL AND TRIM(DepartmentType) <> ''
            GROUP BY PatientDurableKey, DepartmentType
          )
          WHERE rn = 1
        ), dominant_type AS (
          SELECT PatientDurableKey, Type AS dominant_Type
          FROM (
            SELECT
              PatientDurableKey, Type, COUNT(*) AS n,
              ROW_NUMBER() OVER (PARTITION BY PatientDurableKey ORDER BY COUNT(*) DESC, Type) AS rn
            FROM patient_timeline_event
            WHERE Type IS NOT NULL AND TRIM(Type) <> ''
            GROUP BY PatientDurableKey, Type
          )
          WHERE rn = 1
        )
        SELECT
          p.DurableKey AS PatientDurableKey,
          p.PatientBirthYearBin,
          p.SexAssignedAtBirth,
          p.FirstRace,
          p.OmbRace,
          p.OmbEthnicity,
          p.MaritalStatus,
          p.SmokingStatus,
          p.VitalStatus,
          p.MyChartStatus,
          p.SexualOrientation,
          p.CensusBlockGroupFipsCode,
          e.first_observed_event_date,
          e.last_observed_event_date,
          e.observed_history_duration_days,
          COALESCE(e.n_events, 0) AS n_events,
          COALESCE(e.n_unique_DiagnosisValue, 0) AS n_unique_DiagnosisValue,
          COALESCE(e.n_unique_GroupCode, 0) AS n_unique_GroupCode,
          COALESCE(e.n_unique_DepartmentType, 0) AS n_unique_DepartmentType,
          COALESCE(e.n_unique_DepartmentSpecialty, 0) AS n_unique_DepartmentSpecialty,
          COALESCE(e.n_unique_Type, 0) AS n_unique_Type,
          COALESCE(e.n_ed_visits, 0) AS n_ed_visits,
          COALESCE(e.n_hospital_admissions, 0) AS n_hospital_admissions,
          COALESCE(e.n_hospital_outpatient_visits, 0) AS n_hospital_outpatient_visits,
          COALESCE(e.n_inpatient_admissions, 0) AS n_inpatient_admissions,
          COALESCE(e.n_observations, 0) AS n_observations,
          COALESCE(e.n_outpatient_face_to_face_visits, 0) AS n_outpatient_face_to_face_visits,
          g.mean_gap_days,
          g.max_gap_days,
          COALESCE(g.n_long_gaps_30d, 0) AS n_long_gaps_30d,
          COALESCE(g.n_long_gaps_90d, 0) AS n_long_gaps_90d,
          COALESCE(g.n_long_gaps_180d, 0) AS n_long_gaps_180d,
          COALESCE(g.n_long_gaps_365d, 0) AS n_long_gaps_365d,
          COALESCE(e.sdoh_any_observed, 0) AS sdoh_any_observed,
          COALESCE(e.sdoh_num_domains_observed, 0) AS sdoh_num_domains_observed,
          e.sdoh_first_observed_date,
          e.sdoh_last_observed_date,
          dg.dominant_GroupCode,
          dd.dominant_DepartmentType,
          dt.dominant_Type,
          p.patient_geography_known_flag AS geography_known_flag,
          geo.PopulationValue,
          geo.CENTLAT,
          geo.CENTLON,
          CASE
            WHEN COALESCE(e.n_events, 0) = 0 THEN 'NO_ENCOUNTERS'
            WHEN e.n_events = 1 THEN 'ONE_EVENT'
            WHEN e.n_events BETWEEN 2 AND 5 THEN 'TWO_TO_FIVE_EVENTS'
            WHEN e.n_events BETWEEN 6 AND 20 THEN 'SIX_TO_TWENTY_EVENTS'
            ELSE 'TWENTY_PLUS_EVENTS'
          END AS patient_sequence_length_bin,
          (COALESCE(e.n_events, 0)
            + COALESCE(e.n_unique_GroupCode, 0)
            + COALESCE(e.n_unique_DepartmentType, 0)
            + COALESCE(e.n_ed_visits, 0)
            + COALESCE(e.n_hospital_admissions, 0)
            + COALESCE(g.n_long_gaps_90d, 0)
          ) AS patient_complexity_score
        FROM dim_patient p
        LEFT JOIN event_agg e ON p.DurableKey = e.PatientDurableKey
        LEFT JOIN gap_agg g ON p.DurableKey = g.PatientDurableKey
        LEFT JOIN dominant_group dg ON p.DurableKey = dg.PatientDurableKey
        LEFT JOIN dominant_dept dd ON p.DurableKey = dd.PatientDurableKey
        LEFT JOIN dominant_type dt ON p.DurableKey = dt.PatientDurableKey
        LEFT JOIN dim_geography geo ON p.CensusBlockGroupFipsCode = geo.GEOID;

        CREATE INDEX IF NOT EXISTS idx_patient_timeline_summary_patient
          ON patient_timeline_summary(PatientDurableKey);
        """,
    )


# ----------------------------- supporting diagnosis journey tables ----------

def build_diagnosis_episode(conn: sqlite3.Connection, split_gap_days: int = 180) -> None:
    print("[BUILD] diagnosis_episode")
    run(conn, "DROP TABLE IF EXISTS diagnosis_episode;")
    run(
        conn,
        f"""
        CREATE TABLE diagnosis_episode AS
        WITH usable AS (
          SELECT *
          FROM patient_timeline_event
          WHERE PrimaryDiagnosisKey IS NOT NULL
            AND TRIM(PrimaryDiagnosisKey) <> ''
            AND PrimaryDiagnosisKey <> '-1'
            AND DiagnosisValue IS NOT NULL
            AND TRIM(DiagnosisValue) <> ''
        ), ordered AS (
          SELECT
            *,
            LAG(Date) OVER (
              PARTITION BY PatientDurableKey, DiagnosisValue
              ORDER BY Date, AdmissionInstant, EncounterKey
            ) AS prev_diag_date
          FROM usable
        ), flags AS (
          SELECT
            *,
            CASE
              WHEN prev_diag_date IS NULL THEN 1
              WHEN julianday(Date) IS NULL OR julianday(prev_diag_date) IS NULL THEN 0
              WHEN CAST(julianday(Date) - julianday(prev_diag_date) AS INTEGER) > {split_gap_days} THEN 1
              ELSE 0
            END AS new_episode_flag
          FROM ordered
        ), numbered AS (
          SELECT
            *,
            SUM(new_episode_flag) OVER (
              PARTITION BY PatientDurableKey, DiagnosisValue
              ORDER BY Date, AdmissionInstant, EncounterKey
              ROWS UNBOUNDED PRECEDING
            ) AS episode_number
          FROM flags
        )
        SELECT
          PatientDurableKey || '|' || DiagnosisValue || '|' || episode_number AS diagnosis_episode_id,
          PatientDurableKey,
          DiagnosisValue,
          MAX(DiagnosisName) AS DiagnosisName,
          MAX(GroupCode) AS GroupCode,
          MAX(GroupName) AS GroupName,
          episode_number,
          MIN(Date) AS first_observed_date,
          MAX(Date) AS last_observed_date,
          CASE
            WHEN julianday(MAX(Date)) IS NULL OR julianday(MIN(Date)) IS NULL THEN NULL
            ELSE CAST(julianday(MAX(Date)) - julianday(MIN(Date)) AS INTEGER)
          END AS episode_duration_days,
          COUNT(*) AS n_events_with_this_DiagnosisValue,
          MIN(patient_event_index) AS first_patient_event_index,
          MAX(patient_event_index) AS last_patient_event_index,
          MAX(CASE WHEN days_since_previous_patient_event >= 90 THEN 1 ELSE 0 END) AS reappears_after_90d_flag,
          MAX(CASE WHEN days_since_previous_patient_event >= 180 THEN 1 ELSE 0 END) AS reappears_after_180d_flag,
          MAX(CASE WHEN days_since_previous_patient_event >= 365 THEN 1 ELSE 0 END) AS reappears_after_365d_flag
        FROM numbered
        GROUP BY PatientDurableKey, DiagnosisValue, episode_number;

        CREATE INDEX IF NOT EXISTS idx_diagnosis_episode_patient
          ON diagnosis_episode(PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_diagnosis_episode_diag
          ON diagnosis_episode(DiagnosisValue, GroupCode);
        """,
    )


def build_journey_episode(conn: sqlite3.Connection) -> None:
    print("[BUILD] journey_episode")
    run(conn, "DROP TABLE IF EXISTS journey_episode;")
    run(
        conn,
        """
        CREATE TABLE journey_episode AS
        WITH usable AS (
          SELECT *
          FROM encounter_enriched_with_sdoh
          WHERE PrimaryDiagnosisKey IS NOT NULL
            AND TRIM(PrimaryDiagnosisKey) <> ''
            AND PrimaryDiagnosisKey <> '-1'
            AND DiagnosisValue IS NOT NULL
            AND TRIM(DiagnosisValue) <> ''
        ), ordered AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY PatientDurableKey, DiagnosisValue
              ORDER BY Date, AdmissionInstant, EncounterKey
            ) AS event_index,
            LAG(Date) OVER (
              PARTITION BY PatientDurableKey, DiagnosisValue
              ORDER BY Date, AdmissionInstant, EncounterKey
            ) AS previous_journey_event_date
          FROM usable
        ), gaps AS (
          SELECT
            *,
            CASE
              WHEN previous_journey_event_date IS NULL THEN NULL
              WHEN julianday(Date) IS NULL OR julianday(previous_journey_event_date) IS NULL THEN NULL
              ELSE CAST(julianday(Date) - julianday(previous_journey_event_date) AS INTEGER)
            END AS journey_gap_days
          FROM ordered
        ), first_last AS (
          SELECT
            PatientDurableKey,
            DiagnosisValue,
            MAX(CASE WHEN event_index = 1 THEN Type END) AS first_Type,
            MAX(CASE WHEN event_index = 1 THEN VisitTypeDescription END) AS first_VisitTypeDescription,
            MAX(CASE WHEN event_index = 1 THEN DepartmentType END) AS first_DepartmentType
          FROM gaps
          GROUP BY PatientDurableKey, DiagnosisValue
        ), last_rows AS (
          SELECT PatientDurableKey, DiagnosisValue, Type AS last_Type,
                 VisitTypeDescription AS last_VisitTypeDescription,
                 DepartmentType AS last_DepartmentType
          FROM (
            SELECT
              *,
              ROW_NUMBER() OVER (
                PARTITION BY PatientDurableKey, DiagnosisValue
                ORDER BY Date DESC, AdmissionInstant DESC, EncounterKey DESC
              ) AS rn_desc
            FROM gaps
          )
          WHERE rn_desc = 1
        )
        SELECT
          g.PatientDurableKey || '|' || g.DiagnosisValue AS journey_id,
          g.PatientDurableKey,
          g.DiagnosisValue,
          MAX(g.DiagnosisName) AS DiagnosisName,
          MAX(g.GroupCode) AS GroupCode,
          MAX(g.GroupName) AS GroupName,
          MIN(g.Date) AS first_observed_date,
          MAX(g.Date) AS last_observed_date,
          CASE
            WHEN julianday(MAX(g.Date)) IS NULL OR julianday(MIN(g.Date)) IS NULL THEN NULL
            ELSE CAST(julianday(MAX(g.Date)) - julianday(MIN(g.Date)) AS INTEGER)
          END AS journey_observed_duration_days,
          COUNT(*) AS n_encounters,
          COUNT(DISTINCT g.DepartmentKey) AS n_unique_departments,
          COUNT(DISTINCT g.DepartmentType) AS n_unique_department_types,
          COUNT(DISTINCT g.provider_Type) AS n_unique_provider_types,
          SUM(CASE WHEN LOWER(TRIM(COALESCE(g.IsEdVisit,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_ed_visits,
          SUM(CASE WHEN LOWER(TRIM(COALESCE(g.IsHospitalAdmission,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_hospital_admissions,
          SUM(CASE WHEN LOWER(TRIM(COALESCE(g.IsHospitalOutpatientVisit,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_hospital_outpatient_visits,
          SUM(CASE WHEN LOWER(TRIM(COALESCE(g.IsInpatientAdmission,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_inpatient_admissions,
          SUM(CASE WHEN LOWER(TRIM(COALESCE(g.IsObservation,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_observations,
          SUM(CASE WHEN LOWER(TRIM(COALESCE(g.IsOutpatientFaceToFaceVisit,''))) IN ('1','true','t','yes','y') THEN 1 ELSE 0 END) AS n_outpatient_face_to_face_visits,
          fl.first_Type,
          lr.last_Type,
          fl.first_VisitTypeDescription,
          lr.last_VisitTypeDescription,
          fl.first_DepartmentType,
          lr.last_DepartmentType,
          MIN(g.journey_gap_days) AS min_gap_days,
          AVG(g.journey_gap_days) AS mean_gap_days,
          MAX(g.journey_gap_days) AS max_gap_days,
          SUM(CASE WHEN g.journey_gap_days >= 30 THEN 1 ELSE 0 END) AS n_long_gaps_30d,
          SUM(CASE WHEN g.journey_gap_days >= 90 THEN 1 ELSE 0 END) AS n_long_gaps_90d,
          SUM(CASE WHEN g.journey_gap_days >= 180 THEN 1 ELSE 0 END) AS n_long_gaps_180d,
          SUM(CASE WHEN g.journey_gap_days >= 365 THEN 1 ELSE 0 END) AS n_long_gaps_365d,
          MAX(g.sdoh_any_observed) AS sdoh_any_observed_during_journey,
          MAX(g.sdoh_num_domains_answered) AS sdoh_num_domains_observed_during_journey,
          GROUP_CONCAT(DISTINCT g.sdoh_domains_observed) AS sdoh_domain_history,
          CASE WHEN MIN(g.Date) <= (SELECT MIN(Date) FROM encounters) THEN 1 ELSE 0 END AS observed_start_censored_flag,
          CASE WHEN MAX(g.Date) >= (SELECT MAX(Date) FROM encounters) THEN 1 ELSE 0 END AS observed_end_censored_flag
        FROM gaps g
        LEFT JOIN first_last fl
          ON g.PatientDurableKey = fl.PatientDurableKey AND g.DiagnosisValue = fl.DiagnosisValue
        LEFT JOIN last_rows lr
          ON g.PatientDurableKey = lr.PatientDurableKey AND g.DiagnosisValue = lr.DiagnosisValue
        GROUP BY g.PatientDurableKey, g.DiagnosisValue;

        CREATE INDEX IF NOT EXISTS idx_journey_episode_journey ON journey_episode(journey_id);
        CREATE INDEX IF NOT EXISTS idx_journey_episode_patient ON journey_episode(PatientDurableKey);
        CREATE INDEX IF NOT EXISTS idx_journey_episode_diag ON journey_episode(DiagnosisValue, GroupCode);
        """,
    )


def build_journey_event_sequence(conn: sqlite3.Connection) -> None:
    print("[BUILD] journey_event_sequence")
    run(conn, "DROP TABLE IF EXISTS journey_event_sequence;")
    run(
        conn,
        """
        CREATE TABLE journey_event_sequence AS
        SELECT
          pte.PatientDurableKey || '|' || pte.DiagnosisValue AS journey_id,
          pte.PatientDurableKey,
          pte.EncounterKey,
          ROW_NUMBER() OVER (
            PARTITION BY pte.PatientDurableKey, pte.DiagnosisValue
            ORDER BY pte.Date, pte.AdmissionInstant, pte.EncounterKey
          ) AS event_index,
          pte.Date,
          pte.AdmissionInstant,
          pte.DischargeInstant,
          pte.days_since_previous_patient_event AS days_since_previous_event,
          pte.gap_bin,
          pte.PrimaryDiagnosisKey,
          pte.DiagnosisValue,
          pte.DiagnosisName,
          pte.GroupCode,
          pte.GroupName,
          pte.Type,
          pte.VisitType,
          pte.VisitTypeDescription,
          pte.DepartmentKey,
          pte.DepartmentName,
          pte.DepartmentSpecialty,
          pte.DepartmentType,
          pte.IsEdVisit,
          pte.IsHospitalAdmission,
          pte.IsHospitalOutpatientVisit,
          pte.IsInpatientAdmission,
          pte.IsObservation,
          pte.IsOutpatientFaceToFaceVisit,
          pte.sdoh_any_observed,
          pte.sdoh_num_questions_answered,
          pte.sdoh_num_domains_answered,
          pte.sdoh_domain_answer_tokens,
          pte.event_token,
          pte.time_gap_token,
          pte.care_setting_token,
          pte.diagnosis_token,
          pte.department_token,
          pte.sdoh_token
        FROM patient_timeline_event pte
        WHERE pte.PrimaryDiagnosisKey IS NOT NULL
          AND TRIM(pte.PrimaryDiagnosisKey) <> ''
          AND pte.PrimaryDiagnosisKey <> '-1'
          AND pte.DiagnosisValue IS NOT NULL
          AND TRIM(pte.DiagnosisValue) <> '';

        CREATE INDEX IF NOT EXISTS idx_journey_event_sequence_journey_idx
          ON journey_event_sequence(journey_id, event_index);
        """,
    )


def build_journey_transition(conn: sqlite3.Connection) -> None:
    print("[BUILD] journey_transition")
    run(conn, "DROP TABLE IF EXISTS journey_transition;")
    run(
        conn,
        """
        CREATE TABLE journey_transition AS
        SELECT
          a.journey_id,
          a.PatientDurableKey,
          a.EncounterKey AS from_EncounterKey,
          b.EncounterKey AS to_EncounterKey,
          a.event_index AS from_event_index,
          b.event_index AS to_event_index,
          a.Date AS from_Date,
          b.Date AS to_Date,
          CASE
            WHEN julianday(b.Date) IS NULL OR julianday(a.Date) IS NULL THEN NULL
            ELSE CAST(julianday(b.Date) - julianday(a.Date) AS INTEGER)
          END AS gap_days,
          b.gap_bin AS gap_bin,
          a.Type AS from_Type,
          b.Type AS to_Type,
          a.VisitTypeDescription AS from_VisitTypeDescription,
          b.VisitTypeDescription AS to_VisitTypeDescription,
          a.DepartmentType AS from_DepartmentType,
          b.DepartmentType AS to_DepartmentType,
          a.DepartmentSpecialty AS from_DepartmentSpecialty,
          b.DepartmentSpecialty AS to_DepartmentSpecialty,
          a.event_token AS from_event_token,
          b.event_token AS to_event_token,
          'TRANSITION:TYPE:' || COALESCE(NULLIF(TRIM(a.Type), ''), 'MISSING') ||
            '_TO_TYPE:' || COALESCE(NULLIF(TRIM(b.Type), ''), 'MISSING') AS transition_token
        FROM journey_event_sequence a
        JOIN journey_event_sequence b
          ON a.journey_id = b.journey_id
         AND b.event_index = a.event_index + 1;

        CREATE INDEX IF NOT EXISTS idx_journey_transition_journey
          ON journey_transition(journey_id, from_event_index);
        """,
    )


def build_diagnosis_group_summary(conn: sqlite3.Connection) -> None:
    print("[BUILD] diagnosis_group_summary")
    run(conn, "DROP TABLE IF EXISTS diagnosis_group_summary;")
    run(
        conn,
        """
        CREATE TABLE diagnosis_group_summary AS
        SELECT
          GroupCode,
          GroupName,
          NULL AS DiagnosisValue,
          NULL AS DiagnosisName,
          COUNT(DISTINCT PatientDurableKey) AS n_patients,
          COUNT(*) AS n_journeys,
          SUM(n_encounters) AS n_encounters,
          AVG(journey_observed_duration_days) AS mean_journey_observed_duration_days,
          AVG(n_encounters) AS mean_n_encounters_per_journey,
          AVG(max_gap_days) AS mean_max_gap_days,
          AVG(CASE WHEN n_ed_visits > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_ed_visit,
          AVG(CASE WHEN n_hospital_admissions > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_hospital_admission,
          AVG(CASE WHEN n_inpatient_admissions > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_inpatient_admission,
          AVG(CASE WHEN n_outpatient_face_to_face_visits > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_outpatient_face_to_face_visit,
          AVG(CASE WHEN sdoh_any_observed_during_journey > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_sdoh_observed
        FROM journey_episode
        GROUP BY GroupCode, GroupName
        UNION ALL
        SELECT
          GroupCode,
          GroupName,
          DiagnosisValue,
          DiagnosisName,
          COUNT(DISTINCT PatientDurableKey) AS n_patients,
          COUNT(*) AS n_journeys,
          SUM(n_encounters) AS n_encounters,
          AVG(journey_observed_duration_days) AS mean_journey_observed_duration_days,
          AVG(n_encounters) AS mean_n_encounters_per_journey,
          AVG(max_gap_days) AS mean_max_gap_days,
          AVG(CASE WHEN n_ed_visits > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_ed_visit,
          AVG(CASE WHEN n_hospital_admissions > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_hospital_admission,
          AVG(CASE WHEN n_inpatient_admissions > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_inpatient_admission,
          AVG(CASE WHEN n_outpatient_face_to_face_visits > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_outpatient_face_to_face_visit,
          AVG(CASE WHEN sdoh_any_observed_during_journey > 0 THEN 1.0 ELSE 0.0 END) AS pct_with_sdoh_observed
        FROM journey_episode
        GROUP BY GroupCode, GroupName, DiagnosisValue, DiagnosisName;

        CREATE INDEX IF NOT EXISTS idx_diagnosis_group_summary_group
          ON diagnosis_group_summary(GroupCode, DiagnosisValue);
        """,
    )


def build_model_feature_matrix_patient(conn: sqlite3.Connection) -> None:
    print("[BUILD] model_feature_matrix_patient")
    run(conn, "DROP TABLE IF EXISTS model_feature_matrix_patient;")
    run(
        conn,
        """
        CREATE TABLE model_feature_matrix_patient AS
        WITH trans AS (
          SELECT
            PatientDurableKey,
            COUNT(*) AS n_transitions,
            COUNT(DISTINCT transition_token) AS n_unique_transition_tokens,
            SUM(CASE WHEN gap_days >= 30 THEN 1 ELSE 0 END) AS transition_gaps_30d,
            SUM(CASE WHEN gap_days >= 90 THEN 1 ELSE 0 END) AS transition_gaps_90d,
            SUM(CASE WHEN gap_days >= 180 THEN 1 ELSE 0 END) AS transition_gaps_180d,
            SUM(CASE WHEN gap_days >= 365 THEN 1 ELSE 0 END) AS transition_gaps_365d
          FROM patient_timeline_transition
          GROUP BY PatientDurableKey
        ), diag AS (
          SELECT
            PatientDurableKey,
            COUNT(*) AS n_diagnosis_episodes,
            COUNT(DISTINCT DiagnosisValue) AS n_diagnosis_values_in_episodes,
            COUNT(DISTINCT GroupCode) AS n_group_codes_in_episodes,
            SUM(CASE WHEN reappears_after_90d_flag = 1 THEN 1 ELSE 0 END) AS n_diag_reappears_after_90d,
            SUM(CASE WHEN reappears_after_180d_flag = 1 THEN 1 ELSE 0 END) AS n_diag_reappears_after_180d,
            SUM(CASE WHEN reappears_after_365d_flag = 1 THEN 1 ELSE 0 END) AS n_diag_reappears_after_365d
          FROM diagnosis_episode
          GROUP BY PatientDurableKey
        )
        SELECT
          s.*,
          seq.full_event_token_sequence,
          seq.full_gap_token_sequence,
          seq.full_diagnosis_token_sequence,
          seq.full_department_token_sequence,
          seq.full_setting_token_sequence,
          seq.full_sdoh_token_sequence,
          COALESCE(t.n_transitions, 0) AS n_transitions,
          COALESCE(t.n_unique_transition_tokens, 0) AS n_unique_transition_tokens,
          COALESCE(t.transition_gaps_30d, 0) AS transition_gaps_30d,
          COALESCE(t.transition_gaps_90d, 0) AS transition_gaps_90d,
          COALESCE(t.transition_gaps_180d, 0) AS transition_gaps_180d,
          COALESCE(t.transition_gaps_365d, 0) AS transition_gaps_365d,
          COALESCE(d.n_diagnosis_episodes, 0) AS n_diagnosis_episodes,
          COALESCE(d.n_diagnosis_values_in_episodes, 0) AS n_diagnosis_values_in_episodes,
          COALESCE(d.n_group_codes_in_episodes, 0) AS n_group_codes_in_episodes,
          COALESCE(d.n_diag_reappears_after_90d, 0) AS n_diag_reappears_after_90d,
          COALESCE(d.n_diag_reappears_after_180d, 0) AS n_diag_reappears_after_180d,
          COALESCE(d.n_diag_reappears_after_365d, 0) AS n_diag_reappears_after_365d,
          CASE WHEN s.n_long_gaps_90d > 0 THEN 1 ELSE 0 END AS long_gap_90d,
          CASE WHEN s.n_ed_visits > 0 THEN 1 ELSE 0 END AS has_ed_visit,
          CASE WHEN s.n_hospital_admissions > 0 THEN 1 ELSE 0 END AS has_hospital_admission,
          CASE WHEN s.patient_complexity_score >= 20 THEN 1 ELSE 0 END AS high_complexity_patient_flag
        FROM patient_timeline_summary s
        LEFT JOIN patient_timeline_sequence seq
          ON s.PatientDurableKey = seq.PatientDurableKey
        LEFT JOIN trans t
          ON s.PatientDurableKey = t.PatientDurableKey
        LEFT JOIN diag d
          ON s.PatientDurableKey = d.PatientDurableKey;

        CREATE INDEX IF NOT EXISTS idx_model_feature_matrix_patient
          ON model_feature_matrix_patient(PatientDurableKey);
        """,
    )


# ----------------------------- QC + export ---------------------------------

def write_qc_report(conn: sqlite3.Connection, processed_dir: Path) -> None:
    print("[QC] Writing row count and join coverage report")
    report = processed_dir / "join_validation_report.md"

    def scalar(sql: str) -> int | float | str | None:
        row = conn.execute(sql).fetchone()
        return row[0] if row else None

    raw_counts = {t: scalar(f"SELECT COUNT(*) FROM {qident(t)}") for t in RAW_FILES}
    final_counts = {
        t: scalar(f"SELECT COUNT(*) FROM {qident(t)}") for t in FINAL_TABLES if table_exists(conn, t)
    }

    join_checks = {
        "encounters.PatientDurableKey -> patients.DurableKey": """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN patients p ON e.PatientDurableKey = p.DurableKey
            WHERE p.DurableKey IS NULL
        """,
        "encounters.PrimaryDiagnosisKey -> diagnosis.DiagnosisKey": """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN diagnosis d ON e.PrimaryDiagnosisKey = d.DiagnosisKey
            WHERE e.PrimaryDiagnosisKey <> '-1' AND d.DiagnosisKey IS NULL
        """,
        "encounters.DepartmentKey -> departments.DepartmentKey": """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN departments d ON e.DepartmentKey = d.DepartmentKey
            WHERE d.DepartmentKey IS NULL
        """,
        "encounters.ProviderDurableKey -> providers.DurableKey": """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN providers p ON e.ProviderDurableKey = p.DurableKey
            WHERE e.ProviderDurableKey IS NOT NULL AND TRIM(e.ProviderDurableKey) <> '' AND p.DurableKey IS NULL
        """,
        "encounters.AttendingProviderDurableKey -> providers.DurableKey": """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN providers p ON e.AttendingProviderDurableKey = p.DurableKey
            WHERE e.AttendingProviderDurableKey IS NOT NULL AND TRIM(e.AttendingProviderDurableKey) <> '' AND p.DurableKey IS NULL
        """,
        "encounters.DischargeProviderDurableKey -> providers.DurableKey": """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN providers p ON e.DischargeProviderDurableKey = p.DurableKey
            WHERE e.DischargeProviderDurableKey IS NOT NULL AND TRIM(e.DischargeProviderDurableKey) <> '' AND p.DurableKey IS NULL
        """,
        "patients.CensusBlockGroupFipsCode -> tigercensuscodes.GEOID": """
            SELECT COUNT(*) FROM patients p
            LEFT JOIN tigercensuscodes t ON p.CensusBlockGroupFipsCode = t.GEOID
            WHERE p.CensusBlockGroupFipsCode IS NOT NULL
              AND TRIM(p.CensusBlockGroupFipsCode) <> ''
              AND p.CensusBlockGroupFipsCode NOT IN ('*Unspecified','Unspecified','*Unknown','Unknown','NA')
              AND t.GEOID IS NULL
        """,
        "social_determinants.EncounterKey -> encounters.EncounterKey": """
            SELECT COUNT(*) FROM social_determinants s
            LEFT JOIN encounters e ON s.EncounterKey = e.EncounterKey
            WHERE e.EncounterKey IS NULL
        """,
        "social_determinants.PatientDurableKey -> patients.DurableKey": """
            SELECT COUNT(*) FROM social_determinants s
            LEFT JOIN patients p ON s.PatientDurableKey = p.DurableKey
            WHERE p.DurableKey IS NULL
        """,
    }

    with report.open("w", encoding="utf-8") as f:
        f.write("# Join Validation Report\n\n")
        f.write("## Raw row counts\n\n")
        f.write("| Table | Rows |\n|---|---:|\n")
        for t, n in raw_counts.items():
            f.write(f"| `{t}` | {n:,} |\n")
        f.write("\n## Derived row counts\n\n")
        f.write("| Table | Rows |\n|---|---:|\n")
        for t, n in final_counts.items():
            f.write(f"| `{t}` | {n:,} |\n")
        f.write("\n## Unmatched join counts\n\n")
        f.write("| Join | Unmatched rows |\n|---|---:|\n")
        for label, sql in join_checks.items():
            n = scalar(sql)
            f.write(f"| `{label}` | {n:,} |\n")
        f.write("\nNote: unmatched provider-role keys can be expected when an encounter role does not correspond to a person in `providers.csv`.\n")
    print(f"[QC] {report}")


def export_table(conn: sqlite3.Connection, table: str, processed_dir: Path, chunksize: int) -> None:
    if not table_exists(conn, table):
        print(f"[EXPORT] Skipping missing table {table}")
        return
    out = processed_dir / f"{table}.csv.gz"
    print(f"[EXPORT] {table} -> {out}")
    first = True
    with gzip_open_text(out, "wt") as f:
        for chunk in pd.read_sql_query(f"SELECT * FROM {qident(table)}", conn, chunksize=chunksize):
            chunk.to_csv(f, index=False, header=first)
            first = False


def gzip_open_text(path: Path, mode: str):
    import gzip

    return gzip.open(path, mode, encoding="utf-8", newline="")


def export_all(conn: sqlite3.Connection, processed_dir: Path, chunksize: int) -> None:
    print("[EXPORT] Exporting final tables as CSV.GZ")
    for table in FINAL_TABLES:
        export_table(conn, table, processed_dir, chunksize)


# ----------------------------- orchestration --------------------------------

def build_all(conn: sqlite3.Connection, split_gap_days: int) -> None:
    build_level1(conn)
    build_encounter_enriched(conn)
    build_sdoh_summary(conn)
    build_encounter_with_sdoh(conn)
    build_event_enriched(conn)
    build_patient_timeline_event(conn)
    build_patient_timeline_sequence(conn)
    build_patient_timeline_transition(conn)
    build_patient_timeline_summary(conn)
    build_diagnosis_episode(conn, split_gap_days=split_gap_days)
    build_journey_episode(conn)
    build_journey_event_sequence(conn)
    build_journey_transition(conn)
    build_diagnosis_group_summary(conn)
    build_model_feature_matrix_patient(conn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DataFest patient timeline and journey tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python src/build_datafest_tables.py --raw-dir data/raw --export

            Expected raw files:
              data/raw/departments.csv
              data/raw/diagnosis.csv
              data/raw/encounters.csv
              data/raw/patients.csv
              data/raw/providers.csv
              data/raw/social_determinants.csv
              data/raw/tigercensuscodes.csv
            """
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--replace-raw", action="store_true", help="Reload raw CSV tables even if they already exist in SQLite.")
    parser.add_argument("--skip-load", action="store_true", help="Skip raw CSV loading and use existing SQLite raw tables.")
    parser.add_argument("--skip-build", action="store_true", help="Skip derived table building.")
    parser.add_argument("--export", action="store_true", help="Export final tables to data/processed/*.csv.gz.")
    parser.add_argument("--split-gap-days", type=int, default=180, help="Gap threshold for optional diagnosis_episode splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.raw_dir, args.interim_dir, args.processed_dir)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_load:
        validate_raw_schema(args.raw_dir)

    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")

    try:
        if not args.skip_load:
            load_all_raw(conn, args.raw_dir, args.chunksize, replace=args.replace_raw)
            create_indexes(conn)
        if not args.skip_build:
            build_all(conn, split_gap_days=args.split_gap_days)
            write_qc_report(conn, args.processed_dir)
        if args.export:
            export_all(conn, args.processed_dir, args.chunksize)
    finally:
        conn.close()

    print("[DONE] DataFest pipeline complete")


if __name__ == "__main__":
    main()
