import os
import time
import uuid
from datetime import datetime, timedelta
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

    # Wind sensors publish on a different cadence than temperature/humidity, so their
    # "same" reading can land 1-2 minutes off. Match within this tolerance instead of
    # requiring an exact timestamp match, or wind rows get silently dropped.
    MATCH_TOLERANCE = timedelta(minutes=2)

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
                    match = _find_closest_row(rows, timestamp, station_id, station.get("name"), "wind_dir", MATCH_TOLERANCE)
                    if match is not None:
                        match["wind_dir"] = reading["value"]
                        match["wind_dir_offset_seconds"] = (timestamp - match["timestamp"]).total_seconds()
                elif i == 3:
                    match = _find_closest_row(rows, timestamp, station_id, station.get("name"), "wind_speed", MATCH_TOLERANCE)
                    if match is not None:
                        match["wind_speed"] = reading["value"]
                        match["wind_speed_offset_seconds"] = (timestamp - match["timestamp"]).total_seconds()

    return pd.DataFrame(rows)


def _find_closest_row(rows, timestamp, station_id, station_name, field_name, tolerance):
    """Find the row for this station closest in time to `timestamp` that doesn't already have `field_name`, within `tolerance`."""
    best_row = None
    best_diff = None
    for row in rows:
        if row["station_id"] != station_id or row["station_name"] != station_name:
            continue
        if field_name in row:
            continue

        diff = abs(row["timestamp"] - timestamp)
        if diff > tolerance:
            continue
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_row = row

    return best_row


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
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        data_df.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


WEATHER_RETRY_MAX_AGE = timedelta(hours=1)


def collect_pending_weather(pending, file_ready_queue=None, now=None):
    """Prioritise the newest tick, then repair one older gap per pass."""
    current = now or datetime.now(SINGAPORE_TZ)
    expired = {tick for tick in pending if current - tick >= WEATHER_RETRY_MAX_AGE}
    for tick in sorted(expired):
        print(f"weather retry expired for {tick:%Y%m%d%H%M}; use the backlog tool for older gaps")
    pending.difference_update(expired)
    if not pending:
        return
    ordered = sorted(pending)
    selected = [ordered[-1]]
    if len(ordered) > 1:
        selected.append(ordered[0])
    for tick in selected:
        # A slow previous request must not start another already-expired retry.
        current = now or datetime.now(SINGAPORE_TZ)
        if current - tick >= WEATHER_RETRY_MAX_AGE:
            pending.discard(tick)
            print(f"weather retry expired for {tick:%Y%m%d%H%M}")
            continue
        key = tick.strftime("%Y%m%d%H%M")
        path = OUTPUT_DIR / f"weather_{key}.csv"
        if path.is_file() and path.stat().st_size > 0:
            pending.discard(tick)
            continue
        try:
            # Retry the original timestamp, never relabel current observations.
            data_df = fetch_once(tick)
            if data_df.empty:
                raise ValueError("API returned no weather rows")
            output_path = save_to_csv(data_df, timestamp=key)
        except Exception as exc:
            print(f"weather fetch failed for {key}; retained for retry: {exc}")
            continue
        pending.discard(tick)
        if file_ready_queue is not None:
            file_ready_queue.put(int(key))
        print(f"saved {len(data_df)} rows to {output_path}")


def main(file_ready_queue=None) -> None:
    now = datetime.now(SINGAPORE_TZ)
    current = now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)
    # Recover recent gaps after a restart; the backlog tool handles older history.
    next_tick = current - timedelta(hours=1)
    pending = set()
    while True:
        now = datetime.now(SINGAPORE_TZ)
        current = now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)
        # Include every elapsed boundary even if an API call took several minutes.
        while next_tick <= current:
            pending.add(next_tick)
            next_tick += timedelta(minutes=5)
        collect_pending_weather(pending, file_ready_queue)
        time.sleep(15)


if __name__ == "__main__":
    main()
