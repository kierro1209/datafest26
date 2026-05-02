"""
DataFest table-build configuration.

This file is the single source of truth for exact raw CSV column names and
expected raw file names. Do not edit source column names unless the team updates
its schema documentation.
"""
from __future__ import annotations

from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
DB_PATH = INTERIM_DIR / "datafest_pipeline.sqlite"

RAW_FILES = {
    "departments": "departments.csv",
    "diagnosis": "diagnosis.csv",
    "encounters": "encounters.csv",
    "patients": "patients.csv",
    "providers": "providers.csv",
    "social_determinants": "social_determinants.csv",
    "tigercensuscodes": "tigercensuscodes.csv",
}

EXPECTED_COLUMNS = {
    "departments": [
        "DepartmentKey", "Address", "City", "County", "DepartmentName",
        "DepartmentSpecialty", "DepartmentType", "PostalCode", "CensusTract",
    ],
    "diagnosis": [
        "DiagnosisKey", "GroupName", "GroupCode", "DiagnosisName", "DiagnosisValue",
    ],
    "encounters": [
        "Date", "AdmissionInstant", "AdmitYear", "AdmitMonth", "AdmitDay",
        "AdmitHour", "AdmitMinute", "AdmissionSource", "AdmissionType",
        "DischargeInstant", "DischargeYear", "DischargeMonth", "DischargeDay",
        "DischargeHour", "DischargeMinute", "EncounterKey", "PatientDurableKey",
        "Type", "VisitType", "VisitTypeDescription", "ProviderDurableKey",
        "AttendingProviderDurableKey", "DischargeProviderDurableKey", "DepartmentKey",
        "PrimaryDiagnosisKey", "IsEdVisit", "IsHospitalAdmission",
        "IsHospitalOutpatientVisit", "IsInpatientAdmission", "IsObservation",
        "IsOutpatientFaceToFaceVisit",
    ],
    "patients": [
        "CensusBlockGroupFipsCode", "DurableKey", "FirstRace", "MaritalStatus",
        "MyChartStatus", "OmbEthnicity", "OmbRace", "SexAssignedAtBirth",
        "SexualOrientation", "SmokingStatus", "VitalStatus", "PatientBirthYearBin",
    ],
    "providers": [
        "DurableKey", "ClinicianTitle", "OfficeAddress", "OfficeCity",
        "OfficePostalCode", "PrimaryDepartment", "PrimarySpecialty", "Type",
    ],
    "social_determinants": [
        "DisplayName", "AnswerText", "EncounterKey", "PatientDurableKey", "Domain",
    ],
    "tigercensuscodes": [
        "GEOID", "PopulationValue", "CENTLAT", "CENTLON",
    ],
}

SDOH_DOMAINS = [
    "Alcohol Use",
    "Depression",
    "Financial Resource Strain",
    "Food Insecurity",
    "Housing Stability",
    "Intimate Partner Violence",
    "Physical Activity",
    "Social Connections",
    "Stress",
    "Transportation Needs",
    "Utilities",
]

FINAL_TABLES = [
    # level 1
    "dim_patient",
    "dim_diagnosis",
    "dim_department",
    "dim_provider",
    "dim_geography",
    "fact_encounter_base",
    "fact_sdoh_response",
    # level 2
    "encounter_enriched",
    "sdoh_encounter_summary",
    "encounter_enriched_with_sdoh",
    "event_enriched",
    # level 3: patient timeline
    "patient_timeline_event",
    "patient_timeline_sequence",
    "patient_timeline_transition",
    "patient_timeline_summary",
    "model_feature_matrix_patient",
    # supporting diagnosis-centered tables
    "diagnosis_episode",
    "journey_episode",
    "journey_event_sequence",
    "journey_transition",
    "diagnosis_group_summary",
]
