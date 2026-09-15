"""Replay the real episode policy on supplied local observations and forecasts."""
from datetime import timedelta
from telegram_code.rain_state import next_rain_state, should_notify, START, ENDING, ENDED


def replay_location(records, lead_minutes=5):
    """Assume automatic mode and successful immediate delivery; gaps reset episodes.

    Records have time, actual and forecast local maxima. No external side effects.
    Truth is later radar at the forecast time, not independent gauge observations.
    """
    by_time = {r['time']: r for r in records}
    previous, prior_tick, last_start = {}, None, None
    alerts, onsets = [], []
    for row in sorted(records, key=lambda r:r['time']):
        now = row['time']
        contiguous = prior_tick is not None and now-prior_tick == timedelta(minutes=5)
        if not contiguous:
            previous, last_start = {}, None
        was_wet = previous.get('radar_raining',False)
        wet = row['actual'] > .01
        result = next_rain_state(previous, row['forecast'], row['actual'], now,
                                 now+timedelta(minutes=lead_minutes))
        if should_notify(previous,result,now):
            reason = result['reason']
            future = by_time.get(result['rain_forecast_at'])
            truth = None if future is None else future['actual'] > .01
            alerts.append(dict(time=now, forecast_at=result['rain_forecast_at'], reason=reason,
                               future_rain=truth, actual_rain=wet,
                               premature_clear=None if reason!=ENDING or truth is None else bool(truth)))
            result['rain_episode_reason'] = reason
            result['rain_alert_reason'] = reason
            if reason == START:
                last_start = now
        if contiguous and wet and not was_wet:
            minutes = None if last_start is None else (now-last_start).total_seconds()/60
            onsets.append(dict(time=now, warning_lead_minutes=minutes,
                               warned_in_horizon=minutes is not None and 0<minutes<=lead_minutes,
                               alerted_at_onset=minutes==0))
        previous.update(result)
        prior_tick = now
    return alerts, onsets
