#!/usr/bin/env python3
"""
Generate synthetic 3-D GPT-style planner inputs for RL training pipelines.

No dependency on raw DataFest CSVs: uses mapping.ROUTING_HINT_BUCKETS and
DEFAULT_SYNTHETIC_DIAG_VOCAB_SIZE for plausible ranges.

Example:
  python generate_synthetic_gpt_vectors.py -n 10000 -o synthetic/gpt_planner_train.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Repo root must be on path before importing `rl.*` (script is not run as -m only).
_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.mapping import (  # noqa: E402
    DEFAULT_SYNTHETIC_DIAG_VOCAB_SIZE,
    GPT_VECTOR_SPEC,
    MAPPING_VERSION,
    ROUTING_HINT_BUCKETS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--num-rows", type=int, default=10_000, help="Number of synthetic rows")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "synthetic" / "gpt_planner_train.csv",
        help="Output CSV path",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--diag-vocab",
        type=int,
        default=DEFAULT_SYNTHETIC_DIAG_VOCAB_SIZE,
        help="Exclusive upper bound for token_diagnosis_id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    spec = GPT_VECTOR_SPEC
    fieldnames = [
        spec.dim0_name,
        spec.dim1_name,
        spec.dim2_name,
        "mapping_version",
    ]
    n_hint = len(ROUTING_HINT_BUCKETS)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _ in range(args.num_rows):
            w.writerow(
                {
                    spec.dim0_name: rng.randrange(0, max(1, args.diag_vocab)),
                    spec.dim1_name: rng.random(),  # [0.0, 1.0); Python upper bound exclusive on rand
                    spec.dim2_name: rng.randrange(0, n_hint),
                    "mapping_version": MAPPING_VERSION,
                }
            )

    print(f"Wrote {args.num_rows} rows to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
