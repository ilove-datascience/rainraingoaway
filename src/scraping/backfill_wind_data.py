"""Backfill wind_dir/wind_speed into existing weather CSVs that predate the tolerance-match fix.

Finds weather_*.csv files missing wind_dir/wind_speed, re-queries the GovSG API for that
historical timestamp, and merges the wind columns (plus offset diagnostics) back in by station_id.
Safe to re-run: files that already have wind data are skipped.
"""
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gov_api import fetch_once, OUTPUT_DIR

WIND_COLS = ["wind_dir", "wind_speed", "wind_dir_offset_seconds", "wind_speed_offset_seconds"]

def file_timestamp(csv_path: Path) -> datetime:
    return datetime.strptime(csv_path.stem.replace("weather_", ""), "%Y%m%d%H%M")


def needs_backfill(csv_path: Path) -> bool:
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()
    return "wind_dir" not in header or "wind_speed" not in header


def backfill_file(csv_path: Path) -> bool:
    timestamp_str = csv_path.stem.replace("weather_", "")
    query_date = datetime.strptime(timestamp_str, "%Y%m%d%H%M")

    existing_df = pd.read_csv(csv_path)
    fresh_df = fetch_once(query_date)

    if fresh_df.empty or "wind_dir" not in fresh_df.columns:
        print(f"  no wind data available for {timestamp_str}, skipping")
        return False

    wind_lookup = fresh_df.set_index("station_id")[
        [col for col in WIND_COLS if col in fresh_df.columns]
    ]

    merged_df = existing_df.merge(
        wind_lookup, left_on="station_id", right_index=True, how="left"
    )
    merged_df.to_csv(csv_path, index=False)

    filled = merged_df["wind_dir"].notna().sum() if "wind_dir" in merged_df.columns else 0
    print(f"  backfilled {timestamp_str}: {filled}/{len(merged_df)} rows got wind data")
    return True


def main():
    csv_files = sorted(OUTPUT_DIR.glob("weather_*.csv"))
    to_fix = [f for f in csv_files if needs_backfill(f)]
    print(f"Found {len(to_fix)}/{len(csv_files)} files needing wind backfill "
          f"({len(csv_files)} total files)\n")

    fixed = 0
    failed = 0
    for i, csv_path in enumerate(to_fix, start=1):
        print(f"[{i}/{len(to_fix)}] {csv_path.name}")
        try:
            if backfill_file(csv_path):
                fixed += 1  
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")

        time.sleep(1 + random.random())

    print(f"\nDone. Fixed {fixed}, failed {failed}, out of {len(to_fix)} candidates.")


if __name__ == "__main__":
    main()
