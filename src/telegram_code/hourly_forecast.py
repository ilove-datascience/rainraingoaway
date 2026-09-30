"""Hourly rain estimates from Open-Meteo, separate from official NEA outlooks."""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta
import logging
import math

import requests

from telegram_code.daily_forecast import SGT, _sg_time, _clock_label


URL = 'https://api.open-meteo.com/v1/forecast'
LOGGER = logging.getLogger(__name__)
REFERENCE = {'label': 'Singapore reference point', 'latitude': 1.35, 'longitude': 103.82}


def _locations(chat_id):
    # Keep optional location lookup bounded, including when MySQL is unavailable.
    import mysql.connector
    from telegram_code.database import _load_db_config
    with closing(mysql.connector.connect(**_load_db_config(), connection_timeout=3,
                                        read_timeout=5, write_timeout=5)) as conn:
        with closing(conn.cursor(dictionary=True)) as cur:
            cur.execute('SELECT label,latitude,longitude FROM user_location WHERE userid=%s ORDER BY location_id LIMIT 6',
                        (chat_id,))
            return cur.fetchall()


def _number(value, maximum):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= maximum)


def parse_hourly(payload, target_date):
    """Open-Meteo precipitation at 14:00 describes 13:00–14:00, not 14:00–15:00."""
    try:
        units, hourly = payload['hourly_units'], payload['hourly']
        if payload['utc_offset_seconds'] != 8*3600 or units != {
                'time': 'unixtime', 'precipitation_probability': '%', 'precipitation': 'mm'}:
            return None
        times, probabilities, amounts = (hourly[k] for k in ('time', 'precipitation_probability', 'precipitation'))
        if not all(isinstance(values, list) for values in (times, probabilities, amounts)):
            return None
        if not len(times) == len(probabilities) == len(amounts) or len(times) > 96:
            return None
        rows, seen = [], set()
        for stamp, chance, amount in zip(times, probabilities, amounts):
            if not _number(stamp, 10**11) or stamp in seen:
                return None
            seen.add(stamp)
            end = datetime.fromtimestamp(stamp, SGT)
            start = end - timedelta(hours=1)
            if start.date() != target_date or end.minute or end.second or end.microsecond:
                continue
            chance = chance if _number(chance, 100) else None
            amount = amount if _number(amount, 500) else None
            if chance is not None or amount is not None:
                rows.append({'start': start, 'end': end, 'chance': chance, 'amount': amount})
        return sorted(rows, key=lambda row: row['start']) or None
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None


def fetch_hourly(latitude, longitude, target_date):
    # Only approximate coordinates go to the provider; labels/chat IDs stay local.
    params = {'latitude': round(latitude, 2), 'longitude': round(longitude, 2),
              'hourly': 'precipitation_probability,precipitation', 'timezone': 'Asia/Singapore',
              'timeformat': 'unixtime', 'precipitation_unit': 'mm',
              'start_date': target_date.isoformat(),
              # Include next midnight to cover the target day's 23:00–00:00 rain.
              'end_date': (target_date + timedelta(days=1)).isoformat()}
    with requests.get(URL, params=params, timeout=(3, 5)) as response:
        response.raise_for_status()
        return parse_hourly(response.json(), target_date)


async def _cached_hourly(bot_data, latitude, longitude, target_date, now):
    cache = bot_data.setdefault('hourly_forecast_cache', {})
    for old_key, value in list(cache.items()):
        if not value['lock'].locked() and value.get('until', now) < now:
            cache.pop(old_key, None)
    key = (round(latitude, 2), round(longitude, 2), target_date)
    entry = cache.setdefault(key, {'lock': asyncio.Lock()})
    async with entry['lock']:
        if now < entry.get('until', now):
            return entry['rows']
        try:
            rows = await asyncio.to_thread(fetch_hourly, *key)
        except (requests.RequestException, ValueError, TypeError) as exc:
            LOGGER.warning('Hourly forecast unavailable: %s', type(exc).__name__)
            rows = None
        entry.update(rows=rows, until=now+timedelta(minutes=15 if rows else 5))
        return rows


def _time_range(start, end):
    return f'{_clock_label(start)}–{_clock_label(end)}'


def format_hourly_location(label, rows, target_date, now, max_windows=3):
    now = _sg_time(now)
    lines = [f'Around {label}']
    rows = [row for row in (rows or []) if row['end'] > now]
    if not rows:
        return '\n'.join(lines + ['Hourly timing unavailable right now.'])
    windows = []
    for row in rows:
        if (row['chance'] or 0) >= 40 or (row['amount'] or 0) >= .1:
            if windows and windows[-1][-1]['end'] == row['start']:
                windows[-1].append(row)
            else:
                windows.append([row])
    def rank(window):
        return (max((r['chance'] if r['chance'] is not None else -1) for r in window),
                max((r['amount'] or 0) for r in window))
    if windows:
        selected = sorted(sorted(windows, key=rank, reverse=True)[:max_windows], key=lambda w: w[0]['start'])
        for window in selected:
            chances = [row['chance'] for row in window if row['chance'] is not None]
            chance = f'hourly chance up to {max(chances):g}%' if chances else 'chance unavailable'
            lines.append(f'• {_time_range(window[0]["start"], window[-1]["end"])}: rain possible; {chance}.')
        if len(windows) > max_windows:
            lines.append(f'Showing the {max_windows} strongest of {len(windows)} rain windows.')
        peak = max((row for window in windows for row in window), key=lambda row: rank([row]))
        detail = []
        if peak['chance'] is not None:
            detail.append(f'{peak["chance"]:g}% chance')
        if peak['amount'] is not None:
            detail.append(f'{peak["amount"]:g} mm forecast in that hour')
        peak_label = 'Highest chance' if peak['chance'] is not None else 'Largest forecast amount'
        lines.append(f'{peak_label}: {_time_range(peak["start"], peak["end"])} ({"; ".join(detail)}).')
    elif all(row['chance'] is not None for row in rows):
        lines.append(f'Low rain signal in available hours; highest hourly chance {max(r["chance"] for r in rows):g}%.')
    else:
        lines.append('No strong rain amount forecast in available hours; some rain chances are unavailable.')
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=SGT)
    hours_expected = math.ceil(((day_start+timedelta(days=1))-max(day_start, now)).total_seconds()/3600)
    if len(rows) < hours_expected or any(row['chance'] is None for row in rows):
        lines.append('Some hourly data is unavailable.')
    return '\n'.join(lines)


async def hourly_forecast_text(bot_data, chat_id, target_date, now=None):
    now = _sg_time(now)
    try:
        locations = await asyncio.to_thread(_locations, chat_id)
    except Exception as exc:
        LOGGER.warning('Hourly forecast location lookup unavailable: %s', type(exc).__name__)
        # A reference point must not silently stand in for a saved location.
        return None
    locations = locations or [REFERENCE]
    async def section(location):
        label = ' '.join(str(location.get('label', 'Saved location')).split())[:80]
        try:
            lat, lon = float(location['latitude']), float(location['longitude'])
            if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError('Invalid location')
            rows = await _cached_hourly(bot_data, lat, lon, target_date, now)
        except (KeyError, ValueError, TypeError):
            rows = None
        return rows, format_hourly_location(label, rows, target_date, now,
                                           max_windows=2 if len(locations) > 3 else 3)
    sections = await asyncio.gather(*(section(location) for location in locations[:6]))
    if not any(rows for rows, _ in sections):
        return None
    return ('Hourly rain estimates (SGT)\n\n' + '\n\n'.join(text for _, text in sections)
            + '\n\nTimes may shift; these are area forecasts, not exact rain-start times.'
              '\nHourly source: Open-Meteo · https://open-meteo.com/')
