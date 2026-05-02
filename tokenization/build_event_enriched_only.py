#!/usr/bin/env python3
"""
Build only the event-level joined dataset needed for numeric event tokenization.

Use this when raw/Level 1 tables are already loaded/built in the SQLite DB.
It will rebuild the Level 2 dependencies if needed:
  encounter_enriched
  sdoh_encounter_summary
  encounter_enriched_with_sdoh
then build/export:
  event_enriched

Example from project root:
  python src/build_event_enriched_only.py --db-path data/interim/datafest_pipeline.sqlite --processed-dir data/processed --export
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Allow running from project root or directly from src/
sys.path.append(str(Path(__file__).resolve().parent))

from datafest_config import DB_PATH, PROCESSED_DIR  # noqa: E402
from build_datafest_tables import (  # noqa: E402
    build_encounter_enriched,
    build_sdoh_summary,
    build_encounter_with_sdoh,
    build_event_enriched,
    export_table,
    table_exists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event_enriched only.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--export", action="store_true", help="Export data/processed/event_enriched.csv.gz")
    parser.add_argument("--rebuild-level2", action="store_true", help="Force rebuild encounter_enriched, sdoh summary, and encounter_enriched_with_sdoh before event_enriched.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")
    try:
        if args.rebuild_level2 or not table_exists(conn, "encounter_enriched"):
            build_encounter_enriched(conn)
        if args.rebuild_level2 or not table_exists(conn, "sdoh_encounter_summary"):
            build_sdoh_summary(conn)
        if args.rebuild_level2 or not table_exists(conn, "encounter_enriched_with_sdoh"):
            build_encounter_with_sdoh(conn)

        build_event_enriched(conn)

        if args.export:
            export_table(conn, "event_enriched", args.processed_dir, args.chunksize)
    finally:
        conn.close()
    print("[DONE] event_enriched build complete")


if __name__ == "__main__":
    main()
