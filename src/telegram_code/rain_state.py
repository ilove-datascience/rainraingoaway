"""Persistent location rain-state decisions (times are Singapore local time)."""

from datetime import timedelta

DRY_CONFIRMATION = timedelta(minutes=15)

START = 'Rain is predicted at your location'
OBSERVED = 'Radar confirms rain at your location'
ENDING = 'Rain is predicted to end'
ENDED = 'Radar no longer shows rain at your location'
CANCELLED = 'Rain is no longer predicted at your location'
ACTIVE_ALERTS = {START, 'Radar confirms rain at your location', 'Predicted rain has strengthened'}


def next_rain_state(previous, forecast_value, actual_value, observed_at, forecast_at):
    old_time = previous.get('rain_observed_at')
    upgrade = (observed_at == old_time and previous.get('rain_forecast_value') is None
               and forecast_value is not None)
    if old_time and (observed_at < old_time or (observed_at == old_time and not upgrade)):
        return None
    # Radar uses the dataset's rain threshold; forecast uses the bot's gated threshold.
    wet = actual_value > 0.01
    predicted = None if forecast_value is None else forecast_value >= 0.003
    state = 'confirmed' if wet else 'predicted'
    if predicted is False or (predicted is None and not wet):
        state = 'predicted norain'
    reason = None
    last_alert = previous.get('rain_episode_reason') or previous.get('rain_alert_reason')
    active = last_alert in ACTIVE_ALERTS
    # Four distinct dry frames spanning 15 minutes end an episode. Missing
    # observations break continuity; reprocessing a tick cannot advance time.
    contiguous = old_time is not None and observed_at - old_time <= timedelta(minutes=5)
    dry_since = previous.get('rain_dry_since') if contiguous else None
    dry_since = None if wet else (dry_since or observed_at)
    dry_confirmed = dry_since is not None and observed_at - dry_since >= DRY_CONFIRMATION
    episode_wet = bool(previous.get('rain_episode_wet') or previous.get('radar_raining')
                       or last_alert in (OBSERVED, ENDING))
    if wet:
        episode_wet = True
    if not wet and dry_confirmed and predicted is not True and (active or last_alert == ENDING):
        if episode_wet:
            reason = ENDED
        elif predicted is False:
            reason = CANCELLED
        if reason:
            episode_wet = False
    elif wet and not active and last_alert != ENDING:
        reason = OBSERVED
    elif predicted is False and active and wet and last_alert != OBSERVED and not upgrade:
        reason = ENDING
    elif predicted and not active:
        # Until clearance is confirmed, this remains the same rain episode.
        if last_alert != ENDING:
            reason = START
            dry_since = None
            episode_wet = wet
    # Persist the observation separately: a dry forecast must not erase actual rain.
    return dict(state=state, rain_observed_at=observed_at, rain_forecast_at=forecast_at,
                radar_raining=wet, reason=reason, rain_forecast_value=forecast_value,
                rain_dry_since=dry_since, rain_episode_wet=episode_wet)


def should_notify(previous, result, now=None):
    """Episode transitions only; no time-based cooldown."""
    previous_reason = previous.get('rain_episode_reason') or previous.get('rain_alert_reason')
    return result['reason'] is not None and result['reason'] != previous_reason
