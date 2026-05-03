"""
Gap-bin labels aligned with token vocabulary ``gap_to_id`` in final_token_format.json.

Inter-encounter gaps (EDA linked pairs) omit START (reserved for first timestep in sequences).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Display order for stacked plots (excludes START; UNKNOWN last).
ORDERED_GAP_LABELS_INTER_ENCOUNTER: tuple[str, ...] = (
    "0D",
    "1_7D",
    "8_30D",
    "31_90D",
    "91_180D",
    "181_365D",
    "365PLUS",
    "UNKNOWN",
)


def load_gap_vocab_keys(vocab_path: Path) -> set[str]:
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    g = data.get("gap_to_id")
    if not isinstance(g, dict):
        raise ValueError(f"No gap_to_id in {vocab_path}")
    return set(g.keys())


def days_to_gap_label(days: float | int | None) -> str:
    """
    Map nonnegative integer days between encounters to vocabulary gap labels.

    Same bin edges as tokenization (see tokenization/patch_encounter_only_with_sdoh_status.gap_bin).
    Does not emit START (sequence start only).
    """
    if days is None:
        return "UNKNOWN"
    try:
        fd = float(days)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if math.isnan(fd):
        return "UNKNOWN"
    d = int(np.floor(fd))
    if d < 0:
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
