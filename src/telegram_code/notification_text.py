"""Short, explicit forecast versus observation messages."""
from telegram_code.rain_state import START, ENDING, ENDED, CANCELLED, OBSERVED
from data_processing.radar_codec import SOURCE_CATEGORIES
from datetime import datetime, timedelta, timezone
import json
import math
import re


def intensity_label(value, actual=False):
    """Relative thirds of the normalized radar-colour scale, not mm/hour."""
    if not math.isfinite(value):
        raise ValueError('Invalid rain intensity')
    dry = value <= 0.01 if actual else value < 0.003
    if dry:
        return 'No rain'
    if value < 1 / 3:
        return 'Light'
    if value < 2 / 3:
        return 'Moderate'
    return 'Heavy'


def rain_notice(title, observed, forecast=None, radius=None, intensity=None, observed_category=None):
    """Intent first, then intensity, timestamp and area in the same order."""
    actual = forecast is None
    source = 'ACTUAL RADAR' if actual else 'FORECAST'
    lines = [f'{title} | {source}', '']
    if observed_category is not None and not actual:
        raise ValueError("Source categories are only valid for actual radar")
    if observed_category is not None:
        lines.append(f"Observed intensity: {observed_category} (source radar scale)")
    elif intensity is not None:
        kind = 'Observed' if actual else 'Expected'
        lines.append(f'{kind} intensity: {intensity_label(intensity, actual=actual)} (radar scale)')
    if not actual:
        lines.append(f'Forecast for: {forecast:%d %b, %H:%M} SGT')
    lines.append(f'Radar observed: {observed:%d %b, %H:%M} SGT')
    if radius is not None:
        lines.append(f'Area: ~{radius} m around your saved location')
    return '\n'.join(lines)


def alert_text(reason, observed, forecast, radius, intensity=None):
    titles = {OBSERVED: 'RAIN DETECTED', START: 'RAIN EXPECTED', ENDING: 'RAIN EXPECTED TO CLEAR',
              ENDED: 'RAIN CLEARED', CANCELLED: 'RAIN NO LONGER EXPECTED'}
    return rain_notice(titles[reason], observed, None if reason in (ENDED, OBSERVED) else forecast, radius, intensity)


def _location_label(label):
    """Keep a saved label on one line, matching the database's 80-character limit."""
    return ' '.join(str(label or '').split())[:80] or 'Saved location'


def _intensity_categories(reason):
    return ('No rain', *SOURCE_CATEGORIES) if reason == OBSERVED else ('No rain', 'Light', 'Moderate', 'Heavy')


def encode_alert(label, reason, intensity=None, photo_locations=None, feedback=None):
    """Persist facts so several location events can share one delivery caption."""
    if intensity is not None and intensity not in _intensity_categories(reason):
        raise ValueError('Unknown radar intensity category')
    payload = {'rain_alert': 1, 'label': _location_label(label),
               'reason': reason, 'intensity': intensity}
    if feedback is not None:
        payload['feedback'] = feedback
    if photo_locations is not None:
        scope = []
        for pair in photo_locations:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(value) is not int or value < 0 for value in pair)):
                raise ValueError('Photo locations must contain location ID and settings version pairs')
            scope.append(list(pair))
        payload['photo_locations'] = scope
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def _alert_details(item):
    """Read structured events and notices queued before the format was upgraded."""
    message = item.get('message') or ''
    try:
        payload = json.loads(message)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and payload.get('rain_alert') == 1:
        reason = item.get('reason') or payload.get('reason')
        intensity = payload.get('intensity')
        if intensity == 'No rain' or intensity not in _intensity_categories(reason):
            intensity = None
        return _location_label(payload.get('label') or item.get('label')), reason, intensity

    lines = str(message).splitlines()
    # Previously each notice began with a saved label followed by its alert title.
    label = lines[0] if len(lines) > 1 and lines[1].startswith('RAIN ') else item.get('label')
    match = re.search(r'(?:Observed|Expected) intensity: '
                      r'(Light to Moderate|Moderate to Heavy|Light|Moderate|Heavy)\b', str(message))
    intensity = match.group(1) if match else None
    reason = item.get('reason')
    if intensity not in _intensity_categories(reason):
        intensity = None
    return _location_label(label), reason, intensity


def _notice_time(value):
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone(timedelta(hours=8)))
    return f'{value:%d %b, %H:%M} SGT'


def compose_notice(items, outlook=None):
    """Describe a batch without implying every saved location has the same weather.

    Do not truncate: delivery can use a text message when a full caption is too long.
    Times in the database are Singapore local time; aware inputs are converted.
    """
    items = list(items)
    if not items:
        return ''
    details = [_alert_details(item) for item in items]
    observed_times = [_notice_time(item.get('observed_at')) for item in items]
    common_observed = observed_times[0] if len(set(observed_times)) == 1 else None
    forecasts = [_notice_time(item.get('forecast_at')) for item, (_, reason, _) in zip(items, details)
                 if reason == START]
    common_forecast = forecasts[0] if forecasts and len(set(forecasts)) == 1 else None
    lines = ['Rain update', '']
    uses_intensity = False
    for item, (label, reason, intensity), observed in zip(items, details, observed_times):
        if reason == OBSERVED:
            status = f'{intensity} rain detected' if intensity else 'Rain detected'
        elif reason == START:
            status = f'{intensity} rain expected' if intensity else 'Rain expected'
        elif reason == ENDING:
            status = 'Rain may clear soon'
        elif reason == ENDED:
            status = 'Rain cleared (radar dry for 15 minutes)'
        elif reason == CANCELLED:
            status = 'Earlier rain forecast has passed; no rain detected'
        elif reason == 'Predicted rain has strengthened':
            status = 'Rain forecast has strengthened'
        else:
            status = 'Weather conditions changed'
        uses_intensity = uses_intensity or (intensity is not None and reason in (START, OBSERVED))
        lines.append(f'• {label}: {status}.')
        times = []
        if observed and common_observed is None:
            times.append(f'Radar: {observed}')
        if reason == START and common_forecast is None:
            forecast = _notice_time(item.get('forecast_at'))
            if forecast:
                times.append(f'Forecast: {forecast}')
        if times:
            lines.append('  ' + ' · '.join(times))
    context = []
    if common_observed:
        context.append(f'Radar: {common_observed}')
    if common_forecast:
        context.append(f'Forecast: {common_forecast}')
    if uses_intensity:
        context.append('Intensity is relative to the radar colour scale.')
    if context:
        lines.extend(['', *context])
    if outlook and str(outlook).strip():
        lines.extend(['', str(outlook).strip()])
    return '\n'.join(lines)
