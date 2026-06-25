import json
import time
import random # For jittering sleep intervals
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta, timezone

import requests




BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SG_OFFSET_HOURS = 8
FETCH_INTERVAL_SECONDS = 5 * 60
FALLBACK_SLEEP_RATIO = 0.75


def datetime_now_str(custom_minutes: Optional[int] = None, offset_hours: int = 0) -> int:
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


def fallback_sleep_seconds(offset_hours: int = SG_OFFSET_HOURS) -> float:
    """Return a shorter retry wait that stays before the next 5-minute tick."""
    remaining = seconds_until_next_five_minute_tick(offset_hours=offset_hours)
    return max(5.0, remaining * FALLBACK_SLEEP_RATIO)


def scrape_once(img_names: tuple[str, ...] = ("70km", "240km")) -> bool:
    """Fetch one pass of radar images.

    Returns True if any image had to fall back to an earlier tick.
    """
    fell_back_any = False
    for img_name in img_names:
        try:
            dt, _, fell_back = fetch_radar_snapshot(img_name)
            print(f"Captured {img_name} {dt}")
            fell_back_any = fell_back_any or fell_back
        except Exception as exc:
            print(f"Failed to capture {img_name}: {exc}")
    return fell_back_any


def run_scraper_forever(
    img_names: tuple[str, ...] = (["70km"]),
    interval_seconds: int = FETCH_INTERVAL_SECONDS,
) -> None:
    """Continuously fetch radar images and retry sooner after a fallback."""
    while True:
        fell_back_any = scrape_once(img_names)
        sleep_seconds = fallback_sleep_seconds() if fell_back_any else interval_seconds
        print(f"Sleeping {sleep_seconds:.1f}s until next 5-minute tick")
        time.sleep(sleep_seconds)

def floor_to_5min(dt: datetime) -> datetime:
    return dt.replace(
        minute=dt.minute - (dt.minute % 5),
        second=0,
        microsecond=0
    )

def attempt_get_most_recent(img_name = "70km", dt_now:datetime | None= None) -> bool:
    if dt_now is None:
        dt_now = datetime_now_str(offset_hours=SG_OFFSET_HOURS)

    dt_now = datetime.strptime(str(dt_now), "%Y%m%d%H%M")
    dt_rounded_down = int(floor_to_5min(dt_now).strftime("%Y%m%d%H%M"))
    _,_,fellback= fetch_radar_snapshot(img_name=img_name, dt=dt_rounded_down)
    if fellback:
        print(f"Radar for {dt_now}, not available")
    
    return not fellback 
    
    
def get_previous_ticks(dt: int, count: int = 4,most_recent_success=False) -> list[int]:
    """Return a list of the previous `count` 5-m
    inute ticks as YYYYMMDDHHMM ints.

    Example: if dt == 202606241310 and count == 3, returns
    [202606241305, 202606241300, 202606241255]
    """
    dt_dt = datetime.strptime(str(dt), "%Y%m%d%H%M")
    prev_ticks: list[int] = []
    if most_recent_success:
        prev_ticks.append(int(dt_dt.strftime("%Y%m%d%H%M")))
        print(f"Using timestamp: {dt_dt}")
        count = max(count - 1, 0)
    
        
    
    for i in range(1, count + 1):
        prev = dt_dt - timedelta(minutes=5 * i)
        print(f"Using timestamp: {prev}")
        prev_ticks.append(int(prev.strftime("%Y%m%d%H%M")))
    return prev_ticks


def check_previous_pngs_exist(img_name: str, dt: int | None = None, count: int = 3) -> tuple[bool, list[int]]:
    """Check whether the previous `count` PNG files exist for `img_name`.

    Returns (all_exist, missing_list) where `missing_list` contains the
    timestamps that were not found.
    If `dt` is None, uses the current floored timestamp via `datetime_now_str`.
    """
    if dt is None:
        dt = datetime_now_str(offset_hours=SG_OFFSET_HOURS)

    prev_ticks = get_previous_ticks(dt, count)
    missing: list[int] = []
    for t in prev_ticks:
        png_path = DATA_DIR / img_name / "png" / f"{t}.png"
        if not png_path.exists():
            missing.append(t)

    return (len(missing) == 0, missing)


def check_history(img_name: str, dt: int | None = None, count: int = 3) -> bool:
    """Return True only if all previous `count` PNGs exist, otherwise False."""
    all_exist, _missing = check_previous_pngs_exist(img_name, dt=dt, count=count)
    return all_exist


def fetch_radar_snapshot(img_name: str, dt: int | None = None) -> tuple[int, Path, bool]:
    if dt is None:
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
    request_retries = 3
    retry_delay_seconds = 2
    dt_dt = datetime.strptime(str(dt), "%Y%m%d%H%M")
    response = None
    last_exc = None
    fell_back = False

    for attempt in range(max_retries + 1):
        attempt_dt = dt_dt - timedelta(minutes=5 * attempt)
        attempt_str = int(attempt_dt.strftime("%Y%m%d%H%M"))
        url = f"{base_url}dpsri_{img_name}_{attempt_str}0000dBR.dpsri.png"

        for request_attempt in range(request_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                if attempt > 0:
                    print(f"Fell back to earlier tick: {attempt_str} (attempt {attempt})")
                    fell_back = True
                dt = attempt_str
                break
            #timeout handling
            except requests.exceptions.Timeout as e:
                last_exc = e
                if request_attempt < request_retries:
                    print(
                        f"Request timed out for {img_name} {attempt_str} "
                        f"(retry {request_attempt + 1}/{request_retries})"
                    )
                    time.sleep(retry_delay_seconds)
                    continue
                break
            except requests.exceptions.HTTPError as e:
                last_exc = e
                if response is not None and response.status_code == 404:
                    # image not available yet; try previous tick
                    time.sleep(1)
                    break
                raise
        else:
            continue

        if response is not None and response.status_code == 200:
            break

        if response is not None and response.status_code == 404:
            continue

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

    return dt, png_path, fell_back #, json_path


def main() -> None:
    run_scraper_forever()


if __name__ == "__main__":
    main()