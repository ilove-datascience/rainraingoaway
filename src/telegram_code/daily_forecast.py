"""Cached Singapore outlook from the official NEA/MSS 24-hour forecast.

The forecast's validity often crosses midnight. Always show its actual window,
rather than presenting it as a midnight-to-midnight or location-specific forecast.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import math

import requests


FORECAST_URL = 'https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast'
OUTLOOK_URL = 'https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook'
SOURCE_URL = 'https://data.gov.sg/datasets/d_ce2eb1e307bda31993c533285834ef2b/view'
SGT = timezone(timedelta(hours=8))
CACHE_TTL = timedelta(minutes=15)
RETRY_DELAY = timedelta(minutes=5)
MAX_ISSUE_AGE = timedelta(hours=12)
_LOGGER = logging.getLogger(__name__)


def _sg_time(value=None):
    if value is None:
        return datetime.now(SGT)
    # The bot's existing clock uses naive Singapore-local datetimes.
    if value.tzinfo is None:
        return value.replace(tzinfo=SGT)
    return value.astimezone(SGT)


def _api_time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Forecast timestamps must include a timezone')
    return result.astimezone(SGT)


def _current(forecast, now):
    return bool(forecast and forecast['valid_from'] <= now < forecast['valid_until']
                and timedelta(0) <= now - forecast['issued_at'] <= MAX_ISSUE_AGE)


def _weather_values(text, temperature):
    if not isinstance(text, str):
        raise ValueError('Invalid forecast text')
    text = ' '.join(text.split())
    if not text or len(text) > 180:
        raise ValueError('Invalid forecast text length')
    low, high = temperature['low'], temperature['high']
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (low, high)):
        raise ValueError('Invalid temperature')
    if temperature.get('unit') != 'Degrees Celsius' or not -20 <= low <= high <= 60:
        raise ValueError('Invalid temperature range or unit')
    return {'outlook': text, 'low': low, 'high': high}


REGIONS = ('north', 'south', 'east', 'west', 'central')
WIND_DIRECTIONS = {
    'N': 'N', 'NNE': 'NNE', 'NE': 'NE', 'ENE': 'ENE', 'E': 'E', 'ESE': 'ESE',
    'SE': 'SE', 'SSE': 'SSE', 'S': 'S', 'SSW': 'SSW', 'SW': 'SW', 'WSW': 'WSW',
    'W': 'W', 'WNW': 'WNW', 'NW': 'NW', 'NNW': 'NNW', 'VRB': 'Variable', 'VARIABLE': 'Variable',
}


def _range(values, maximum):
    low, high = values['low'], values['high']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (low, high)) or not 0 <= low <= high <= maximum:
        raise ValueError('Invalid weather range')
    return low, high


def _weather_extras(values):
    """Optional detail must not invalidate an otherwise usable daily forecast."""
    extras = {}
    try:
        humidity = values['relativeHumidity']
        if humidity.get('unit') == 'Percentage':
            extras['humidity'] = _range(humidity, 100)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    try:
        wind = values['wind']
        if wind.get('direction') in WIND_DIRECTIONS and wind['speed'].get('unit', 'km/h') == 'km/h':
            extras['wind'] = (WIND_DIRECTIONS[wind['direction']], *_range(wind['speed'], 200))
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return extras


def _parse_periods(records, forecast):
    """Use ISO timestamps: upstream display labels can contain incorrect dates."""
    if not isinstance(records, list) or len(records) > 8:
        return []
    periods = []
    for record in records:
        try:
            start = _api_time(record['timePeriod']['start'])
            end = _api_time(record['timePeriod']['end'])
            if not forecast['valid_from'] <= start < end <= forecast['valid_until']:
                continue
            regions = {}
            for name in REGIONS:
                value = record['regions'].get(name)
                if not isinstance(value, dict) or not isinstance(value.get('text'), str):
                    continue
                text = ' '.join(value['text'].split())
                if text and len(text) <= 80:
                    regions[name] = text
            if regions:
                periods.append({'start': start, 'end': end, 'regions': regions})
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            continue
    periods.sort(key=lambda period: period['start'])
    if any(first['end'] > second['start'] for first, second in zip(periods, periods[1:])):
        return []  # Contradictory windows do not support a reliable timeline.
    return periods


def parse_daily_forecast(payload, now=None, *, scheduled=False):
    """Return the newest valid outlook, or None for missing/stale/invalid data."""
    now = _sg_time(now)
    try:
        if payload.get('code') != 0:
            return None
        records = payload['data']['records']
        if not isinstance(records, list):
            return None
    except (KeyError, TypeError, AttributeError):
        return None
    forecasts = []
    for record in records:
        try:
            general = record['general']
            forecast = {
                **_weather_values(general['forecast']['text'], general['temperature']),
                'issued_at': _api_time(record['timestamp']),
                'valid_from': _api_time(general['validPeriod']['start']),
                'valid_until': _api_time(general['validPeriod']['end']),
                **_weather_extras(general),
            }
            duration = forecast['valid_until'] - forecast['valid_from']
            current = _current(forecast, now)
            if scheduled:
                # At 05:00 a new outlook can begin at 06:00. Conversely, last
                # night's outlook ending at 06:00 is not a forecast for today.
                current = (timedelta(0) <= now - forecast['issued_at'] <= MAX_ISSUE_AGE
                           and forecast['valid_from'] <= now + timedelta(hours=2)
                           and forecast['valid_until'] >= now + timedelta(hours=6))
            if timedelta(0) < duration <= timedelta(hours=24) and current:
                forecast['periods'] = _parse_periods(record.get('periods'), forecast)
                forecasts.append(forecast)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            continue
    return max(forecasts, key=lambda row: row['issued_at'], default=None)


def parse_dated_outlook(payload, target_date, now=None):
    """Select an exact calendar day from NEA's four-day outlook."""
    now = _sg_time(now)
    forecasts = []
    try:
        if payload.get('code') != 0 or not isinstance(payload['data']['records'], list):
            return None
        for record in payload['data']['records']:
            try:
                issued = _api_time(record['timestamp'])
                if not timedelta(0) <= now - issued <= timedelta(hours=36):
                    continue
                for day in record['forecasts']:
                    try:
                        start = _api_time(day['timestamp'])
                        if start.date() != target_date or start.hour or start.minute or start.second:
                            continue
                        forecasts.append({
                            **_weather_values(day['forecast'].get('summary') or day['forecast']['text'], day['temperature']),
                            'issued_at': issued, 'valid_from': start,
                            'valid_until': start + timedelta(days=1),
                            **_weather_extras(day),
                        })
                    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                        continue
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                continue
    except (KeyError, TypeError, AttributeError):
        return None
    return max(forecasts, key=lambda row: row['issued_at'], default=None)


def fetch_scheduled_forecast(slot, now):
    """Fetch a national forecast covering today, or tomorrow for the night slot."""
    now = _sg_time(now)
    target = now.date() + timedelta(days=slot == 'night')
    if slot != 'night':
        with requests.get(FORECAST_URL, timeout=(3, 5)) as response:
            response.raise_for_status()
            forecast = parse_daily_forecast(response.json(), now, scheduled=True)
        if forecast:
            return forecast
    # Before the morning update, yesterday's four-day issue includes today.
    # This avoids labelling a nearly expired overnight outlook as today's weather.
    params = {'date': (now.date() - timedelta(days=1)).isoformat()} if slot != 'night' else {}
    with requests.get(OUTLOOK_URL, params=params, timeout=(3, 5)) as response:
        response.raise_for_status()
        return parse_dated_outlook(response.json(), target, now)


def fetch_daily_forecast(now=None):
    """Bounded network request; callers should run this outside the event loop."""
    with requests.get(FORECAST_URL, timeout=(3, 5)) as response:
        response.raise_for_status()
        return parse_daily_forecast(response.json(), now)


async def get_daily_forecast(bot_data, now=None):
    """Share one request across chats; failures never prevent rain alerts.

    A still-valid cached outlook survives a transient API failure. Invalid or
    expired forecasts are never returned. No user/location data is sent upstream.
    """
    now = _sg_time(now)
    cache = bot_data.setdefault('_daily_forecast_cache', {'lock': asyncio.Lock()})
    async with cache['lock']:
        forecast = cache.get('forecast')
        if now < cache.get('retry_at', datetime.min.replace(tzinfo=SGT)):
            return forecast if _current(forecast, now) else None
        try:
            updated = await asyncio.to_thread(fetch_daily_forecast, now)
        except (requests.RequestException, ValueError, TypeError) as error:
            # Log only exception type: responses/URLs may include service details.
            _LOGGER.warning('Daily forecast unavailable: %s', type(error).__name__)
            updated = None
        if updated is not None:
            cache['forecast'] = updated
            cache['retry_at'] = now + CACHE_TTL
            return updated
        cache['retry_at'] = now + RETRY_DELAY
        return forecast if _current(forecast, now) else None


def _clock_label(value):
    if value.hour == 0 and value.minute == 0:
        return 'midnight'
    if value.hour == 12 and value.minute == 0:
        return '12 noon'
    minutes = f':{value.minute:02d}' if value.minute else ''
    return f'{value.hour % 12 or 12}{minutes} {"am" if value.hour < 12 else "pm"}'


def _regional_conditions(regions):
    groups = {}
    for name in REGIONS:
        if name in regions:
            groups.setdefault(regions[name], []).append(name)
    if len(groups) == 1 and len(regions) == len(REGIONS):
        return f'{next(iter(groups))} across Singapore'
    return '; '.join(f'{text} ({", ".join(names)})' for text, names in groups.items())


def forecast_detail_lines(forecast, now=None, target_date=None, include_timeline=True):
    """Render supplied windows, never inventing hourly chances or rain duration."""
    now = _sg_time(now)
    lines = []
    if target_date is None:
        start_limit, end_limit = forecast['valid_from'], forecast['valid_until']
    else:
        start_limit = datetime.combine(target_date, datetime.min.time(), tzinfo=SGT)
        end_limit = start_limit + timedelta(days=1)
    timeline = []
    for period in forecast.get('periods', []):
        start, end = max(period['start'], start_limit), min(period['end'], end_limit)
        if end <= start or end <= now:
            continue  # Noon updates omit elapsed morning windows.
        start_label, end_label = _clock_label(start), _clock_label(end)
        if target_date is None and start.date() != now.date():
            start_label = f'{start:%d %b} {start_label}'
        if target_date is None and end.date() != start.date():
            end_label = f'{end:%d %b} {end_label}'
        timeline.append(f'• {start_label}–{end_label}: {_regional_conditions(period["regions"])}')
    if timeline and include_timeline:
        lines.extend(['Approximate timing (SGT)', *timeline,
                      'Showers may be brief or local within these windows.'])
    elif include_timeline:
        lines.append('Detailed time windows are not available for this outlook yet.')
    extras = []
    if 'wind' in forecast:
        direction, low, high = forecast['wind']
        extras.append(f'Wind: {direction} {low:g}–{high:g} km/h')
    if 'humidity' in forecast:
        low, high = forecast['humidity']
        extras.append(f'Humidity: {low:g}–{high:g}%')
    if extras:
        lines.extend(['', *extras])
    return lines


def format_daily_forecast(forecast, compact=True, now=None, hourly=None):
    """A national outlook with explicit source and its true validity window."""
    start, end = forecast['valid_from'], forecast['valid_until']
    lines = [
        'Singapore daily outlook · NEA/MSS',
        f"{forecast['outlook']} · {forecast['low']:g}–{forecast['high']:g}°C",
    ]
    if not compact:
        if hourly:
            lines.extend(['', hourly])
        lines.extend(['', *forecast_detail_lines(forecast, now, include_timeline=not hourly), ''])
    lines.append(f'Valid {start:%d %b %H:%M}–{end:%d %b %H:%M} SGT')
    if not compact:
        lines.append(f"Issued {forecast['issued_at']:%d %b %H:%M} SGT · data.gov.sg")
    return '\n'.join(lines)


async def weather_command(update, context):
    """Show the national outlook without requiring a saved location or alerts."""
    message = update.effective_message
    if message is None:
        return
    forecast = await get_daily_forecast(context.application.bot_data)
    if forecast is None:
        text = 'Daily outlook unavailable right now. Please try /weather again shortly.'
    else:
        from telegram_code.hourly_forecast import hourly_forecast_text
        now = _sg_time()
        try:
            hourly = await hourly_forecast_text(context.application.bot_data, update.effective_chat.id, now.date(), now)
        except Exception as exc:
            _LOGGER.warning('Optional hourly forecast unavailable: %s', type(exc).__name__)
            hourly = None
        text = f'{format_daily_forecast(forecast, compact=False, now=now, hourly=hourly)}\nNEA source: {SOURCE_URL}'
    await message.reply_text(text, disable_web_page_preview=True)
