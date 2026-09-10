"""Forecast wording, freshness and automatic-alert decisions."""
from datetime import datetime, timedelta, timezone

SG_TZ = timezone(timedelta(hours=8))
RAIN_THRESHOLD = 0.003


def sg_now():
    return datetime.now(SG_TZ).replace(tzinfo=None)


def is_fresh(prediction, now=None):
    if not prediction:
        return False
    now = now or sg_now()
    observed = prediction.get('observed_at', prediction['next_tick'] - timedelta(minutes=5))
    return observed <= now < prediction['next_tick']


def forecast_text(value, prediction):
    observed = prediction.get('observed_at', prediction['next_tick'] - timedelta(minutes=5))
    outlook = 'Rain is predicted nearby.' if value >= RAIN_THRESHOLD else 'No rain is predicted nearby.'
    return (f"Forecast\n{outlook}\n\nFor: {prediction['next_tick']:%d %b, %H:%M} SGT\n"
            f"Based on radar at {observed:%H:%M} SGT.")

