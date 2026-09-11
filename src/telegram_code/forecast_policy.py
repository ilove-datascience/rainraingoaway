"""Forecast wording, freshness and automatic-alert decisions."""
from datetime import datetime, timedelta, timezone
from telegram_code.notification_text import rain_notice

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


def forecast_text(value, prediction, radius=None):
    observed = prediction.get('observed_at', prediction['next_tick'] - timedelta(minutes=5))
    title = 'RAIN EXPECTED' if value >= RAIN_THRESHOLD else 'NO RAIN EXPECTED'
    return rain_notice(title, observed, prediction['next_tick'], radius, value)
