import json
import time
import random # For jittering sleep intervals
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests




BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SG_OFFSET_HOURS = 8
FETCH_INTERVAL_SECONDS = 5 * 60


def datetime_now_str(custom_minutes: int | None = None, offset_hours: int = 0) -> int:
    """Return YYYYMMDDHHMM as int (floored to nearest multiple of 5 minutes)
    The result is adjusted by `offset_hours` and optional `custom_minutes`.
    This combines the previous wrapper behaviour by always flooring the
    numeric timestamp to a multiple of 5 (mirrors Math.floor(d/5)*5).
    """
    # Start from timezone-aware UTC, then apply hour offset
    d = datetime.now(timezone.utc) + timedelta(hours=offset_hours)

    if custom_minutes:
        d = d + timedelta(minutes=custom_minutes)

    raw = int(d.strftime("%Y%m%d%H%M"))
    return (raw // 5) * 5


def seconds_until_next_five_minute_tick(offset_hours: int = 0) -> float:
    now = datetime.now(timezone.utc) + timedelta(hours=offset_hours)
    floored = now.replace(second=0, microsecond=0)
    next_minute = ((now.minute // 5) + 1) * 5

    if next_minute >= 60:
        next_run = floored.replace(minute=0) + timedelta(hours=1)
    else:
        next_run = floored.replace(minute=next_minute)

    return max((next_run - now).total_seconds(), 0.0)


def fetch_radar_snapshot(img_name: str) -> tuple[int, Path, Path]:
    dt = datetime_now_str(offset_hours=SG_OFFSET_HOURS)
    base_url_name = "50km" if img_name == "70km" else img_name
    # 70km maps to the 50km endpoint (which uses /v2/), while 240km does not.
    if img_name == "70km":
        base_url = f"https://www.weather.gov.sg/files/rainarea/{base_url_name}/v2/"
    else:
        base_url = f"https://www.weather.gov.sg/files/rainarea/{base_url_name}/"
    headers = {"User-Agent": "Mozilla/5.0"}

    # Try the computed timestamp first; if the server doesn't yet have
    # that image (404), fall back to earlier 5-minute ticks.
    max_retries = 3
    dt_dt = datetime.strptime(str(dt), "%Y%m%d%H%M")
    response = None
    last_exc = None

    for attempt in range(max_retries + 1):
        attempt_dt = dt_dt - timedelta(minutes=5 * attempt)
        attempt_str = int(attempt_dt.strftime("%Y%m%d%H%M"))
        url = f"{base_url}dpsri_{img_name}_{attempt_str}0000dBR.dpsri.png"

        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            if attempt > 0:
                print(f"Fell back to earlier tick: {attempt_str} (attempt {attempt})")
            dt = attempt_str
            break
        except requests.exceptions.HTTPError as e:
            last_exc = e
            if response is not None and response.status_code == 404:
                # image not available yet; try previous tick
                time.sleep(1)
                continue
            raise

    if response is None or response.status_code != 200:
        # Re-raise the last HTTP error to surface the failure
        if last_exc:
            raise last_exc
        raise RuntimeError("Failed to download radar image")

    output_png_dir = DATA_DIR / img_name / "png"
  #  output_json_dir = DATA_DIR / img_name / "json"
    output_png_dir.mkdir(parents=True, exist_ok=True)
   # output_json_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_png_dir / f"{dt}.png"
   # json_path = output_json_dir / f"{dt}.json"

    png_path.write_bytes(response.content)

   # points = png_to_xy_intensity(png_path)
   # json_points = [
   #     {"x": point["x"], "y": point["y"], "value": point["intensity"]}
   #     for point in points
   # ]

    #json_path.write_text(
    #    json.dumps(json_points, indent=2),
    #    encoding="utf-8",
   # )

    print(f"Saved {png_path}")
   # print(f"Saved {json_path} ({len(json_points)} points)")

    return dt, png_path #, json_path


def main() -> None:
    while True:
        for img_name in ("70km", "240km"):
            dt, _ = fetch_radar_snapshot(img_name)
            print(f"Captured {img_name} {dt}")

        sleep_seconds = FETCH_INTERVAL_SECONDS
        print(f"Sleeping {sleep_seconds:.1f}s until next 5-minute tick")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()