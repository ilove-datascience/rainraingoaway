"""Persistent location rain-state decisions (times are Singapore local time)."""

START = 'Rain is predicted at your location'
ENDING = 'Rain is predicted to end'
ENDED = 'Radar no longer shows rain at your location'
CANCELLED = 'Rain is no longer predicted at your location'
ACTIVE_ALERTS = {START, 'Radar confirms rain at your location', 'Predicted rain has strengthened'}


def next_rain_state(previous, forecast_value, actual_value, observed_at, forecast_at):
    if previous.get('rain_observed_at') and observed_at <= previous['rain_observed_at']:
        return None
    # Radar uses the dataset's rain threshold; forecast uses the bot's gated threshold.
    wet = actual_value > 0.01
    predicted = forecast_value >= 0.003
    state = 'confirmed' if wet else 'predicted'
    if not predicted:
        state = 'predicted norain'
    reason = None
    last_alert = previous.get('rain_episode_reason') or previous.get('rain_alert_reason')
    active = last_alert in ACTIVE_ALERTS
    if not wet and previous.get('radar_raining') and (active or last_alert == ENDING):
        reason = ENDED
    elif not predicted and active:
        reason = ENDING if wet else CANCELLED
    elif predicted and not active:
        # A forecast changing back to wet before rain stops is the same episode.
        if last_alert != ENDING or not (wet or previous.get('radar_raining')):
            reason = START
    # Persist the observation separately: a dry forecast must not erase actual rain.
    return dict(state=state, rain_observed_at=observed_at, rain_forecast_at=forecast_at,
                radar_raining=wet, reason=reason, rain_forecast_value=forecast_value)


def should_notify(previous, result, now=None):
    """Episode transitions only; no time-based cooldown."""
    previous_reason = previous.get('rain_episode_reason') or previous.get('rain_alert_reason')
    return result['reason'] is not None and result['reason'] != previous_reason
