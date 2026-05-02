"""
Canonical schema-to-RL semantics for planner inputs and routing hints.

GPT-side outputs are treated as a 3-vector per planned event; see
schema_to_rl_mapping.md for full join notes to encounters/departments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List, Tuple

MAPPING_VERSION: Final[str] = "0.1.0"

# Ordered buckets: collapse departments.DepartmentType (+ unknown) to small action space.
# Index in this list == value for dim2 (routing_hint_id).
ROUTING_HINT_BUCKETS: Final[List[str]] = [
    "ED",
    "ICU",
    "INPATIENT_FLOOR",  # HOD and generic inpatient-style types
    "L_AND_D",
    "OBSERVATION",
    "OUTPATIENT_FACE_TO_FACE",
    "HOSPITAL_OUTPATIENT",
    "OTHER_OR_UNKNOWN",
]

# Placeholder vocabulary size for dim0 when not tied to real diagnosis.csv yet.
DEFAULT_SYNTHETIC_DIAG_VOCAB_SIZE: Final[int] = 2048


@dataclass(frozen=True)
class GPTPlannerVectorSpec:
    """Semantic names for the three dimensions (order matters)."""

    dim0_name: str = "token_diagnosis_id"
    dim1_name: str = "arrival_fraction_day"
    dim2_name: str = "routing_hint_id"


GPT_VECTOR_SPEC = GPTPlannerVectorSpec()


def routing_hint_label(hint_id: int) -> str:
    if hint_id < 0 or hint_id >= len(ROUTING_HINT_BUCKETS):
        return "INVALID"
    return ROUTING_HINT_BUCKETS[hint_id]


def department_type_to_routing_hint_id(dept_type: str | None) -> int:
    """
    Map raw departments.DepartmentType (or encounter-enriched column) to dim2 index.
    Unknown / missing values map to OTHER_OR_UNKNOWN.
    """
    if dept_type is None or (isinstance(dept_type, str) and not dept_type.strip()):
        return len(ROUTING_HINT_BUCKETS) - 1
    t = dept_type.strip().upper()
    if t == "ED":
        return 0
    if t == "ICU":
        return 1
    if t in {"HOD", "*UNKNOWN", "*UNSPECIFIED"} or "INPATIENT" in t or "FLOOR" in t:
        return 2
    if t in {"L&D", "L AND D"}:
        return 3
    if t == "OBSERVATION" or "OBS" in t:
        return 4
    if "OUTPATIENT" in t and "FACE" in t.replace(" ", "").upper():
        return 5
    if "HOSPITAL" in t and "OUTPATIENT" in t:
        return 6
    return len(ROUTING_HINT_BUCKETS) - 1


def validate_gpt_vector(row: Tuple[float, float, float]) -> Tuple[int, float, int]:
    """
    Coerce a raw triple to (int, float in [0,1), int) with clamping.
    Used when ingesting noisy model outputs.
    """
    d0, d1, d2 = row
    i0 = max(0, int(round(d0)))
    f1 = float(d1)
    if f1 < 0.0:
        f1 = 0.0
    elif f1 >= 1.0:
        f1 = 0.999999  # keep in [0, 1)
    i2 = max(0, min(len(ROUTING_HINT_BUCKETS) - 1, int(round(d2))))
    return i0, f1, i2
