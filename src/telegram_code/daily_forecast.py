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


def parse_daily_forecast(payload, now=None):
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
            outlook = general['forecast']['text']
            if not isinstance(outlook, str):
                continue
            outlook = ' '.join(outlook.split())
            # Unexpected long content is omitted, never silently truncated.
            if not outlook or len(outlook) > 180:
                continue
            temperature = general['temperature']
            low, high = temperature['low'], temperature['high']
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in (low, high)):
                continue
            if temperature.get('unit') != 'Degrees Celsius' or not -20 <= low <= high <= 60:
                continue
            forecast = {
                'outlook': outlook,
                'low': low,
                'high': high,
                'issued_at': _api_time(record['timestamp']),
                'valid_from': _api_time(general['validPeriod']['start']),
                'valid_until': _api_time(general['validPeriod']['end']),
            }
            duration = forecast['valid_until'] - forecast['valid_from']
            if timedelta(0) < duration <= timedelta(hours=24) and _current(forecast, now):
                forecasts.append(forecast)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            continue
    return max(forecasts, key=lambda row: row['issued_at'], default=None)


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


def format_daily_forecast(forecast, compact=True):
    """A national outlook with explicit source and its true validity window."""
    start, end = forecast['valid_from'], forecast['valid_until']
    lines = [
        'Singapore daily outlook · NEA/MSS',
        f"{forecast['outlook']} · {forecast['low']:g}–{forecast['high']:g}°C",
        f'Valid {start:%d %b %H:%M}–{end:%d %b %H:%M} SGT',
    ]
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
        text = f'{format_daily_forecast(forecast, compact=False)}\nSource: {SOURCE_URL}'
    await message.reply_text(text, disable_web_page_preview=True)
