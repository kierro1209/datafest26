#!/usr/bin/env python3
"""
Probe raw DataFest CSVs for fields that may describe the purpose of a visit/event.

Goal:
  Find values in raw columns that may indicate tests, imaging, labs, medication,
  procedure, surgery, phone calls, office visits, etc.

Inputs default to data/raw/*.csv. Outputs are written to data/processed/qa/visit_purpose_probe/.

Run:
  python probe_visit_purpose_fields.py --raw-dir data/raw --output-dir data/processed/qa/visit_purpose_probe

This script only summarizes values. It does not print patient-level rows.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

CSV_ENCODING_CANDIDATES = ["utf-8-sig", "utf-16", "cp1252", "latin1"]

RAW_FILES = {
    "encounters": "encounters.csv",
    "departments": "departments.csv",
    "providers": "providers.csv",
    "diagnosis": "diagnosis.csv",
    "social_determinants": "social_determinants.csv",
}

# Raw columns that may answer “what did the patient come for / what happened?”
PROBE_COLUMNS = {
    "encounters": [
        "Type",
        "VisitType",
        "VisitTypeDescription",
        "AdmissionSource",
        "AdmissionType",
        "IsEdVisit",
        "IsHospitalAdmission",
        "IsHospitalOutpatientVisit",
        "IsInpatientAdmission",
        "IsObservation",
        "IsOutpatientFaceToFaceVisit",
        "PrimaryDiagnosisKey",
    ],
    "departments": [
        "DepartmentName",
        "DepartmentSpecialty",
        "DepartmentType",
    ],
    "providers": [
        "ClinicianTitle",
        "PrimaryDepartment",
        "PrimarySpecialty",
        "Type",
    ],
    "diagnosis": [
        "GroupName",
        "GroupCode",
        "DiagnosisName",
        "DiagnosisValue",
    ],
    "social_determinants": [
        "Domain",
        "DisplayName",
        "AnswerText",
    ],
}

# Keywords for coarse “purpose” buckets. These are intentionally broad and should
# be inspected by the team before being used as labels.
PURPOSE_PATTERNS = {
    "LAB_BLOOD_URINE": [
        r"\bLAB\b", r"LABORATORY", r"BLOOD", r"URINE", r"SPECIMEN", r"PHLEBOT", r"VENIPUNCT",
        r"CBC", r"CMP", r"A1C", r"HEMOGLOBIN", r"GLUCOSE", r"CULTURE", r"PATHOLOGY",
    ],
    "IMAGING_RADIOLOGY": [
        r"\bXR\b", r"X\s*RAY", r"XRAY", r"RADIOLOGY", r"IMAGING", r"CT\b", r"MRI\b",
        r"ULTRASOUND", r"US\b", r"MAMMO", r"MAMMOGRAPHY", r"PET\b", r"DEXA", r"ECHO",
    ],
    "PROCEDURE_SURGERY": [
        r"PROCEDURE", r"SURGER", r"OPERAT", r"OR\b", r"ENDOSCOPY", r"COLONOSCOPY",
        r"BIOPSY", r"INJECTION", r"INFUSION", r"DIALYSIS", r"CATH", r"WOUND",
    ],
    "MEDICATION_PHARMACY": [
        r"MEDICATION", r"MEDICINE", r"PHARM", r"RX\b", r"PRESCRIPTION", r"REFILL",
        r"IMMUNIZATION", r"VACCIN", r"CHEMO", r"ANTICOAG",
    ],
    "THERAPY_REHAB": [
        r"THERAPY", r"REHAB", r"PT\b", r"OT\b", r"SPEECH", r"PHYSICAL THERAPY",
        r"OCCUPATIONAL", r"RESPIRATORY THERAPY",
    ],
    "EMERGENCY_URGENT": [
        r"EMERGENCY", r"\bED\b", r"ER\b", r"URGENT", r"TRAUMA", r"AMBULANCE",
    ],
    "INPATIENT_HOSPITAL": [
        r"HOSPITAL", r"INPATIENT", r"ADMISSION", r"ADMIT", r"DISCHARGE", r"OBSERVATION",
    ],
    "OUTPATIENT_OFFICE": [
        r"OFFICE", r"CLINIC", r"OUTPATIENT", r"FOLLOW", r"VISIT", r"APPOINTMENT", r"CONSULT",
    ],
    "VIRTUAL_PHONE_PORTAL": [
        r"PHONE", r"TELE", r"VIRTUAL", r"VIDEO", r"MYCHART", r"PORTAL", r"MESSAGE", r"E\s*VISIT",
    ],
    "PREVENTIVE_SCREENING": [
        r"PREVENT", r"SCREEN", r"WELLNESS", r"ANNUAL", r"PHYSICAL", r"ROUTINE",
    ],
    "BEHAVIORAL_HEALTH": [
        r"BEHAVIOR", r"PSYCH", r"MENTAL", r"DEPRESSION", r"ANXIETY", r"SUBSTANCE", r"COUNSEL",
    ],
}


def choose_encoding(path: Path) -> str:
    for enc in CSV_ENCODING_CANDIDATES:
        try:
            pd.read_csv(path, nrows=0, encoding=enc)
            return enc
        except Exception:
            continue
    return "latin1"


def norm_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def purpose_hits(value: str) -> list[str]:
    text = value.upper()
    hits = []
    for bucket, patterns in PURPOSE_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            hits.append(bucket)
    return hits


def top_counter_rows(counter: Counter, table: str, column: str, total: int, max_rows: int) -> list[dict]:
    rows = []
    for value, count in counter.most_common(max_rows):
        rows.append({
            "table": table,
            "column": column,
            "value": value,
            "count": count,
            "pct_nonmissing": count / total if total else None,
            "purpose_buckets": ";".join(purpose_hits(value)),
        })
    return rows


def probe_table(path: Path, table: str, columns: list[str], chunksize: int, max_values_per_col: int):
    enc = choose_encoding(path)
    header = list(pd.read_csv(path, nrows=0, encoding=enc).columns)
    usecols = [c for c in columns if c in header]
    missing_cols = [c for c in columns if c not in header]

    counters = {c: Counter() for c in usecols}
    nonmissing_counts = Counter()
    missing_counts = Counter()
    bucket_counts = defaultdict(Counter)  # column -> bucket -> count across row values
    n_rows = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunksize,
        encoding=enc,
        on_bad_lines="skip",
    ):
        n_rows += len(chunk)
        for col in usecols:
            vals = chunk[col].map(norm_text)
            missing = vals.eq("")
            missing_counts[col] += int(missing.sum())
            nonmissing_counts[col] += int((~missing).sum())
            counters[col].update(vals[~missing])
            for v in vals[~missing]:
                for bucket in purpose_hits(v):
                    bucket_counts[col][bucket] += 1

    top_rows = []
    summary_rows = []
    bucket_rows = []
    for col in usecols:
        total_nonmissing = nonmissing_counts[col]
        top_rows.extend(top_counter_rows(counters[col], table, col, total_nonmissing, max_values_per_col))
        summary_rows.append({
            "table": table,
            "column": col,
            "rows_scanned": n_rows,
            "nonmissing_rows": total_nonmissing,
            "missing_rows": missing_counts[col],
            "unique_values": len(counters[col]),
            "top_value": counters[col].most_common(1)[0][0] if counters[col] else "",
            "top_value_count": counters[col].most_common(1)[0][1] if counters[col] else 0,
        })
        for bucket, count in bucket_counts[col].most_common():
            bucket_rows.append({
                "table": table,
                "column": col,
                "purpose_bucket": bucket,
                "matching_value_occurrences": count,
                "pct_nonmissing_value_occurrences": count / total_nonmissing if total_nonmissing else None,
            })

    return {
        "table": table,
        "path": str(path),
        "encoding": enc,
        "rows_scanned": n_rows,
        "used_columns": usecols,
        "missing_probe_columns": missing_cols,
        "top_rows": top_rows,
        "summary_rows": summary_rows,
        "bucket_rows": bucket_rows,
    }


def write_markdown_report(results: list[dict], output_dir: Path) -> None:
    report = output_dir / "visit_purpose_probe_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Visit Purpose Probe Report\n\n")
        f.write("This report probes raw fields that may describe what the patient came for or what happened during a visit. It summarizes value frequencies and keyword-based purpose buckets.\n\n")
        f.write("## Files scanned\n\n")
        f.write("| Table | Rows scanned | Encoding | Columns probed | Missing probe columns |\n")
        f.write("|---|---:|---|---|---|\n")
        for r in results:
            f.write(f"| `{r['table']}` | {r['rows_scanned']:,} | `{r['encoding']}` | {len(r['used_columns'])} | {', '.join(r['missing_probe_columns']) or 'None'} |\n")
        f.write("\n## Most relevant raw fields\n\n")
        f.write("Start by inspecting `encounters.Type`, `encounters.VisitType`, `encounters.VisitTypeDescription`, and department fields. These are the most likely to contain visit purpose/service clues.\n\n")
        f.write("Detailed CSV outputs are written next to this report.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe raw fields for visit purpose/service values.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/qa/visit_purpose_probe"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-values-per-column", type=int, default=5000)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_top = []
    all_summary = []
    all_buckets = []
    results = []

    for table, filename in RAW_FILES.items():
        path = args.raw_dir / filename
        if not path.exists():
            print(f"[SKIP] missing {path}")
            continue
        print(f"[PROBE] {table}: {path}")
        res = probe_table(path, table, PROBE_COLUMNS[table], args.chunksize, args.max_values_per_column)
        results.append(res)
        all_top.extend(res["top_rows"])
        all_summary.extend(res["summary_rows"])
        all_buckets.extend(res["bucket_rows"])

    pd.DataFrame(all_summary).to_csv(args.output_dir / "visit_purpose_field_summary.csv", index=False)
    pd.DataFrame(all_top).to_csv(args.output_dir / "visit_purpose_top_values.csv", index=False)
    pd.DataFrame(all_buckets).to_csv(args.output_dir / "visit_purpose_keyword_bucket_counts.csv", index=False)
    write_markdown_report(results, args.output_dir)

    print(f"[DONE] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
