"""Persistent location rain-state decisions (times are Singapore local time)."""

from datetime import timedelta

DRY_CONFIRMATION = timedelta(minutes=15)
FORECAST_REARM = timedelta(minutes=15)

START = 'Rain is predicted at your location'
OBSERVED = 'Radar confirms rain at your location'
ENDING = 'Rain is predicted to end'
ENDED = 'Radar no longer shows rain at your location'
CANCELLED = 'Rain is no longer predicted at your location'
ACTIVE_ALERTS = {START, 'Radar confirms rain at your location', 'Predicted rain has strengthened'}
CLOSED_ALERTS = {ENDED, CANCELLED}


def next_rain_state(previous, forecast_value, actual_value, observed_at, forecast_at):
    old_time = previous.get('rain_observed_at')
    upgrade = (observed_at == old_time and previous.get('rain_forecast_value') is None
               and forecast_value is not None)
    if old_time and (observed_at < old_time or (observed_at == old_time and not upgrade)):
        return None
    # Radar uses the dataset's rain threshold; forecast uses the bot's gated threshold.
    wet = actual_value > 0.01
    predicted = None if forecast_value is None else forecast_value >= 0.003
    state = 'confirmed' if wet else ('predicted' if predicted else 'predicted norain')
    reason = None
    last_alert = previous.get('rain_episode_reason') or previous.get('rain_alert_reason')
    # ENDING remains readable for episodes saved by an older bot version.
    active = last_alert in ACTIVE_ALERTS or last_alert == ENDING
    # Four distinct dry frames spanning 15 minutes end an episode. Missing
    # observations break continuity; reprocessing a tick cannot advance time.
    contiguous = old_time is not None and observed_at - old_time <= timedelta(minutes=5)
    # After closure, preserve the timer across gaps so its forecast-only rearm
    # deadline cannot slide forward every time another observation arrives.
    dry_since = previous.get('rain_dry_since') if contiguous or last_alert in CLOSED_ALERTS else None
    dry_since = None if wet else (dry_since or observed_at)
    dry_confirmed = dry_since is not None and observed_at - dry_since >= DRY_CONFIRMATION
    episode_wet = bool(previous.get('rain_episode_wet') or previous.get('radar_raining')
                       or last_alert in (OBSERVED, ENDING))
    if wet:
        episode_wet = True
    if not wet and dry_confirmed and active:
        # Clearance describes observed rain, not the next forecast. This also
        # retires forecasts that never materialised when the model is offline.
        # Radar-first and model-first processing must reach the same decision.
        reason = ENDED if episode_wet else CANCELLED
        episode_wet = False
        # Older versions could leave an episode active for a much longer dry
        # interval. Keep the last confirmed interval so rearm starts at this
        # closure, not at an earlier time when no clearance was actually sent.
        dry_since = observed_at - DRY_CONFIRMATION
    elif wet and not active:
        reason = OBSERVED
    elif predicted and not active:
        # Do not immediately reverse a confirmed closure because a model tick
        # flickers wet. Actual radar rain above bypasses this short rearm window.
        # The existing dry timer persists the deadline without another column.
        rearmed = (last_alert not in CLOSED_ALERTS
                   or observed_at >= dry_since + DRY_CONFIRMATION + FORECAST_REARM)
        if rearmed:
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
