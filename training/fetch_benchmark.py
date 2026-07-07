"""Incrementally sync the local benchmark corpus with the public poker44 API.

Fetches any source dates missing from the local dataset file (bootstrapping
from FIRST_SOURCE_DATE when the file does not exist yet) and appends them, so
the daily retrain always trains on the freshest released benchmark data.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

API_BASE = "https://api.poker44.net/api/v1/benchmark"
FIRST_SOURCE_DATE = date(2026, 5, 26)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "hands_generator" / "evaluation_datas" / "training_benchmark_full.txt"


def load_existing(path: Path) -> tuple[str, list[dict]]:
    if not path.exists():
        return "", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload.get("data", payload)
    return str(root.get("releaseVersion", "")), list(root.get("chunks") or [])


def main() -> int:
    status = requests.get(API_BASE, timeout=30).json()["data"]
    latest = date.fromisoformat(status["latestSourceDate"])
    release = str(status.get("releaseVersion", ""))
    print(f"API status: latest_source_date={latest} release={release}", flush=True)

    _, records = load_existing(DATASET_PATH)
    have_dates = {str(rec.get("sourceDate", "")) for rec in records}
    print(f"Local dataset: {len(records)} records, {len(have_dates)} dates", flush=True)

    missing = []
    day = FIRST_SOURCE_DATE
    while day <= latest:
        if day.isoformat() not in have_dates:
            missing.append(day.isoformat())
        day += timedelta(days=1)

    if not missing:
        print("No new source dates; dataset already current.", flush=True)
        return 0

    print(f"Fetching {len(missing)} missing dates: {missing}", flush=True)
    added = 0
    for date_str in missing:
        try:
            resp = requests.get(
                f"{API_BASE}/chunks", params={"sourceDate": date_str}, timeout=120
            ).json()
            chunks = resp.get("data", {}).get("chunks", []) or []
            for chunk in chunks:
                chunk.setdefault("sourceDate", date_str)
            records.extend(chunks)
            added += len(chunks)
            print(f"  {date_str}: +{len(chunks)} records", flush=True)
        except Exception as err:  # noqa: BLE001 - keep syncing remaining dates
            print(f"  {date_str}: ERROR {err}", flush=True)
        time.sleep(0.3)

    if not added:
        print("No records fetched; leaving dataset untouched.", flush=True)
        return 0

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATASET_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps({"data": {"releaseVersion": release, "chunks": records}}),
        encoding="utf-8",
    )
    tmp_path.replace(DATASET_PATH)
    print(f"Saved {len(records)} records (+{added}) to {DATASET_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
