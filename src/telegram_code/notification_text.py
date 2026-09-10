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


def alert_text(reason, observed, forecast, radius, intensity=None):
    nearby = f'Near your saved location (~{radius} m).'
    if reason == ENDED:
        strength = f'\nObserved intensity: {intensity_label(intensity, actual=True)} (radar scale)' if intensity is not None else ''
        return f'Actual radar\nRain has cleared nearby.{strength}\n\nObserved: {observed:%d %b, %H:%M} SGT\n{nearby}'
    headlines = {START: 'Rain is predicted nearby.', ENDING: 'Rain is predicted to clear nearby.',
                 CANCELLED: 'Rain is no longer predicted nearby.'}
    strength = f'\nEstimated intensity: {intensity_label(intensity)} (radar scale)' if intensity is not None else ''
    return (f'Forecast\n{headlines[reason]}{strength}\n\nFor: {forecast:%d %b, %H:%M} SGT\n'
            f'Based on radar at {observed:%H:%M} SGT.\n{nearby}')
