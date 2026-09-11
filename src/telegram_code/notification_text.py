"""Short, explicit forecast versus observation messages."""
from telegram_code.rain_state import START, ENDING, ENDED, CANCELLED
import math


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


def rain_notice(title, observed, forecast=None, radius=None, intensity=None):
    """Intent first, then intensity, timestamp and area in the same order."""
    actual = forecast is None
    source = 'ACTUAL RADAR' if actual else 'FORECAST'
    lines = [f'{title} | {source}', '']
    if intensity is not None:
        kind = 'Observed' if actual else 'Expected'
        lines.append(f'{kind} intensity: {intensity_label(intensity, actual=actual)} (radar scale)')
    if not actual:
        lines.append(f'Forecast for: {forecast:%d %b, %H:%M} SGT')
    lines.append(f'Radar observed: {observed:%d %b, %H:%M} SGT')
    if radius is not None:
        lines.append(f'Area: ~{radius} m around your saved location')
    return '\n'.join(lines)


def alert_text(reason, observed, forecast, radius, intensity=None):
    titles = {START: 'RAIN EXPECTED', ENDING: 'RAIN EXPECTED TO CLEAR',
              ENDED: 'RAIN CLEARED', CANCELLED: 'RAIN NO LONGER EXPECTED'}
    return rain_notice(titles[reason], observed, None if reason == ENDED else forecast, radius, intensity)
