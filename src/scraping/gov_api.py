from datetime import datetime, timedelta
import os
import time
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
import requests

URL_humidity = "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity"
URL_TEMP = "https://api-open.data.gov.sg/v2/real-time/api/air-temperature"
URL_WIND_DIR = "https://api-open.data.gov.sg/v2/real-time/api/wind-direction"
URL_WIND_SPEED = "https://api-open.data.gov.sg/v2/real-time/api/wind-speed"
url_list = [URL_humidity, URL_TEMP, URL_WIND_DIR, URL_WIND_SPEED]
SINGAPORE_TZ = ZoneInfo("Asia/Singapore")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "environment"


def _get_api_key() -> str:
    api_key = os.getenv("gov_api_key") or os.getenv("GOV_API_KEY")
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                if key.strip() in {"gov_api_key", "GOV_API_KEY"}:
                    api_key = value.strip().strip('"').strip("'")
                    break

    if not api_key:
        raise RuntimeError("Missing GovSG API key. Set gov_api_key in the environment or repo root .env file.")
    return api_key


def _get_json_with_retry(url: str, params: Optional[dict] = None, timeout_seconds: int = 60, max_wait_seconds: int = 60) -> dict:
    headers = {"api-key": _get_api_key()}
    started_at = time.monotonic()
    while True:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout_seconds)
        except requests.Timeout:
            elapsed = time.monotonic() - started_at
            if elapsed >= max_wait_seconds:
                raise

            time.sleep(1)
            continue

        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        elapsed = time.monotonic() - started_at
        if elapsed >= max_wait_seconds:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        sleep_for = 5
        if retry_after:
            try:
                sleep_for = max(1, min(int(retry_after), max_wait_seconds))
            except ValueError:
                sleep_for = 5

        remaining = max_wait_seconds - elapsed
        time.sleep(max(1, min(sleep_for, remaining)))


def fetch_once(query_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
    params = None
    if query_date is not None:
        if isinstance(query_date, datetime):
            if query_date.tzinfo is None:
                query_date = query_date.replace(tzinfo=SINGAPORE_TZ)
            else:
                query_date = query_date.astimezone(SINGAPORE_TZ)
            query_date = query_date.strftime("%Y-%m-%dT%H:%M:%S")
        params = {"date": query_date}

    data_list = []
    for url in url_list:
        payload = _get_json_with_retry(url, params=params)
        data_list.append(payload["data"])

    stations = {
        station["id"]: station
        for station in data_list[1]["stations"]
    }

    rows = []
    for i, data in enumerate(data_list):
        for reading_batch in data["readings"]:
            timestamp = pd.to_datetime(reading_batch["timestamp"])
            timestamp = timestamp.tz_convert("Asia/Singapore")
            conv_timestamp = timestamp.strftime("%Y%m%d%H%M")

            for reading in reading_batch["data"]:
                station_id = reading["stationId"]
                station = stations.get(station_id, {})

                location = station.get("location") or station.get("labelLocation") or {}
                if i == 0:
                    rows.append({
                        "timestamp": timestamp,
                        "conv_timesamp": conv_timestamp,
                        "station_id": station_id,
                        "station_name": station.get("name"),
                        "humidity": reading["value"],
                        "latitude": location.get("latitude"),
                        "longitude": location.get("longitude"),
                    })
                elif i == 1:
                    for row in rows:
                        if row["timestamp"] == timestamp and row["station_id"] == station_id and row["station_name"] == station.get("name"):
                            row["temperature"] = reading["value"]
                elif i == 2:
                    for row in rows:
                        if row["timestamp"] == timestamp and row["station_id"] == station_id and row["station_name"] == station.get("name"):
                            row["wind_dir"] = reading["value"]
                elif i == 3:
                    for row in rows:
                        if row["timestamp"] == timestamp and row["station_id"] == station_id and row["station_name"] == station.get("name"):
                            row["wind_speed"] = reading["value"]

    return pd.DataFrame(rows)


def sleep_until_next_five_minute_boundary() -> None:
    now = datetime.now(SINGAPORE_TZ)
    # If we're exactly on a 5-minute boundary (e.g. :00, :05, :10) with zero seconds,
    # don't sleep — allow an immediate fetch.
    if now.second == 0 and now.microsecond == 0 and (now.minute % 5) == 0:
        return

    # Otherwise compute the next 5-minute boundary and sleep until then.
    next_minute = ((now.minute // 5) + 1) * 5
    if next_minute >= 60:
        next_run = (now.replace(second=0, microsecond=0) + timedelta(hours=1)).replace(minute=0)
    else:
        next_run = now.replace(minute=next_minute, second=0, microsecond=0)

    time.sleep(max(0.0, (next_run - now).total_seconds()))


def save_to_csv(data_df: pd.DataFrame, timestamp: Optional[str] = None) -> Path:
    if timestamp is None:
        timestamp = datetime.now(SINGAPORE_TZ).strftime("%Y%m%d%H%M")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"weather_{timestamp}.csv"
    print(f"saved at {output_path}")
    data_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    while True:
        # Ensure we run on an exact 5-minute boundary (or immediately if already aligned).
        sleep_until_next_five_minute_boundary()
        data_df = fetch_once()
        output_path = save_to_csv(data_df)
        print(f"saved {len(data_df)} rows to {output_path}")


if __name__ == "__main__":
    main()