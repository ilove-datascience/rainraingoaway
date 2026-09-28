import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
import unittest

spec = importlib.util.spec_from_file_location('rain_state', Path(__file__).resolve().parents[1] / 'src/telegram_code/rain_state.py')
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


class RainStateTests(unittest.TestCase):
    def setUp(self):
        self.tick = datetime(2026, 9, 10, 12)

    def advance(self, previous, forecast, actual, minutes=0):
        observed = self.tick + timedelta(minutes=minutes)
        return state.next_rain_state(previous, forecast, actual, observed, observed + timedelta(minutes=5))

    def test_prediction_confirmation_does_not_send_predicted_clear(self):
        predicted = self.advance({}, .04, 0)
        predicted['rain_alert_reason'] = predicted['reason']
        self.assertEqual(predicted['state'], 'predicted')
        confirmed = self.advance(predicted, .04, .03, 5)
        confirmed['rain_alert_reason'] = predicted['rain_alert_reason']
        self.assertEqual(confirmed['state'], 'confirmed')
        ending = self.advance(confirmed, 0, .03, 10)
        self.assertEqual(ending['state'], 'confirmed')
        self.assertTrue(ending['radar_raining'])
        self.assertIsNone(ending['reason'])

    def test_forecast_alone_never_confirms(self):
        result = self.advance({}, .8, 0)
        self.assertEqual(result['state'], 'predicted')

    def test_old_and_duplicate_radar_ignored(self):
        result = self.advance({}, .03, .03, 5)
        self.assertIsNone(self.advance(result, 0, 0, 5))
        self.assertIsNone(self.advance(result, 0, 0, 0))

    def test_initial_dry_silent(self):
        result = self.advance({}, 0, 0)
        self.assertEqual(result['state'], 'predicted norain')
        self.assertFalse(state.should_notify({}, result, self.tick))

    def test_no_cooldown_and_repeat_suppression(self):
        result = self.advance({}, .02, 0)
        previous = {'rain_alert_at': self.tick, 'rain_alert_reason': 'different'}
        self.assertTrue(state.should_notify(previous, result, self.tick))
        self.assertTrue(state.should_notify(previous, result, self.tick + timedelta(minutes=15)))
        previous['rain_alert_reason'] = result['reason']
        self.assertFalse(state.should_notify(previous, result, self.tick + timedelta(minutes=20)))

    def test_strengthening_and_confirmation_are_silent(self):
        result = self.advance({'rain_alert_reason': state.START, 'rain_alert_value': .02}, .5, .03)
        self.assertIsNone(result['reason'])

    def test_complete_episode_and_new_episode(self):
        previous = {}
        sent = []
        for i, (forecast, actual) in enumerate([
            (.04, 0), (.04, 0), (.1, .03), (0, .03),
            (.04, .03), (0, .03), (0, 0), (0, 0), (0, 0), (0, 0),
            (.04, 0), (.04, 0), (.04, 0),
        ]):
            result = self.advance(previous, forecast, actual, i * 5)
            if state.should_notify(previous, result, self.tick):
                sent.append(result['reason'])
                previous['rain_alert_reason'] = result['reason']
            previous.update(result)
        self.assertEqual(sent, [state.START, state.ENDED, state.START])

    def test_unconfirmed_prediction_cancelled_once(self):
        previous = {'rain_alert_reason': state.START, 'state': 'predicted'}
        for minute in [0,5,10]:
            result = self.advance(previous, 0, 0, minute)
            self.assertIsNone(result['reason'])
            previous.update(result)
        result = self.advance(previous, 0, 0, 15)
        self.assertEqual(result['reason'], state.CANCELLED)
        previous.update(result, rain_alert_reason=result['reason'])
        self.assertIsNone(self.advance(previous, 0, 0, 20)['reason'])

    def test_observed_rain_alerts_when_forecast_misses(self):
        result = self.advance({}, 0, .2)
        self.assertEqual(result['reason'], state.OBSERVED)
        previous = dict(result, rain_episode_reason=state.OBSERVED)
        self.assertIsNone(self.advance(previous, None, .2, 5)['reason'])
        for minute in [5,10,15]:
            dry = self.advance(previous, None, 0, minute)
            self.assertIsNone(dry['reason'])
            previous.update(dry)
        self.assertEqual(self.advance(previous, None, 0, 20)['reason'], state.ENDED)

    def test_missing_forecast_does_not_cancel_predicted_rain(self):
        previous = {'rain_episode_reason':state.START}
        self.assertIsNone(self.advance(previous, None, 0)['reason'])

    def test_forecast_can_upgrade_same_tick_observation(self):
        observed = self.advance({}, None, 0)
        forecast = self.advance(observed, .2, 0)
        self.assertEqual(forecast['reason'], state.START)
        self.assertIsNone(self.advance(forecast, .2, 0))

    def test_radar_start_is_not_repeated_after_predicted_start(self):
        self.assertIsNone(self.advance({'rain_episode_reason':state.START}, None, .2)['reason'])

    def test_overnight_flicker_is_one_episode(self):
        previous = {}
        sent = []
        for minute in range(0,240,5):
            result = self.advance(previous, None, .1 if minute%10==0 else 0, minute)
            if state.should_notify(previous,result):
                sent.append(result['reason'])
                previous['rain_episode_reason'] = result['reason']
            previous.update(result)
            # Model returns dry after the radar job for this same timestamp.
            upgrade = self.advance(previous, 0, .1 if minute%10==0 else 0, minute)
            if state.should_notify(previous,upgrade):
                sent.append(upgrade['reason'])
                previous['rain_episode_reason'] = upgrade['reason']
            previous.update(upgrade)
        self.assertEqual(sent,[state.OBSERVED])
        for minute in [240,245,250]:
            result = self.advance(previous,None,0,minute)
            previous.update(result)
            if result['reason']: sent.append(result['reason'])
        self.assertEqual(sent,[state.OBSERVED,state.ENDED])

    def test_gap_does_not_count_as_continuous_dry(self):
        previous = {'rain_episode_reason':state.OBSERVED,'radar_raining':True}
        previous.update(self.advance(previous,None,0,0))
        result = self.advance(previous,None,0,60)
        self.assertIsNone(result['reason'])
        self.assertEqual(result['rain_dry_since'],self.tick+timedelta(minutes=60))

    def test_observed_start_is_not_immediately_contradicted(self):
        result = self.advance({},None,.2)
        previous = dict(result,rain_episode_reason=state.OBSERVED)
        self.assertIsNone(self.advance(previous,0,.2)['reason'])
        self.assertIsNone(self.advance(previous,0,.2,5)['reason'])

    def test_distinct_locations_have_independent_dry_timers(self):
        first = {'rain_episode_reason':state.OBSERVED,'rain_episode_wet':True}
        second = first.copy()
        for minute in [0,5,10,15]:
            a=self.advance(first,None,0,minute)
            b=self.advance(second,None,.2,minute)
            first.update(a);second.update(b)
        self.assertEqual(a['reason'],state.ENDED)
        self.assertIsNone(b['reason'])

    def test_missing_model_retires_unfulfilled_prediction_and_actual_rain_alerts(self):
        initial = self.advance({}, .2, 0)
        previous = dict(initial, rain_episode_reason=state.START)
        for minute in [5, 10, 15]:
            result = self.advance(previous, None, 0, minute)
            self.assertIsNone(result['reason'])
            previous.update(result)
        result = self.advance(previous, None, 0, 20)
        self.assertEqual(result['reason'], state.CANCELLED)
        previous.update(result, rain_episode_reason=result['reason'])
        self.assertEqual(self.advance(previous, None, .2, 25)['reason'], state.OBSERVED)

    def test_forecast_cannot_restart_on_same_observation_as_clearance(self):
        for closed, was_wet in [(state.ENDED, True), (state.CANCELLED, False)]:
            with self.subTest(closed=closed):
                previous = dict(rain_episode_reason=state.OBSERVED if was_wet else state.START,
                                rain_episode_wet=was_wet)
                for minute in [0, 5, 10, 15]:
                    result = self.advance(previous, None, 0, minute)
                    previous.update(result)
                self.assertEqual(result['reason'], closed)
                previous['rain_episode_reason'] = closed
                upgrade = self.advance(previous, .2, 0, 15)
                self.assertIsNone(upgrade['reason'])
                self.assertEqual(upgrade['rain_dry_since'], self.tick)

    def test_forecast_rearm_is_bounded_and_does_not_suppress_observed_rain(self):
        previous = dict(rain_episode_reason=state.ENDED, rain_episode_wet=False,
                        rain_dry_since=self.tick, rain_observed_at=self.tick+timedelta(minutes=15),
                        rain_forecast_value=None)
        self.assertEqual(self.advance(previous, None, .2, 20)['reason'], state.OBSERVED)
        for minute in [20, 25]:
            result = self.advance(previous, .2, 0, minute)
            self.assertIsNone(result['reason'])
            self.assertEqual(result['rain_dry_since'], self.tick)
            previous.update(result)
        self.assertEqual(self.advance(previous, .2, 0, 30)['reason'], state.START)
        self.assertEqual(self.advance(previous, None, .2, 30)['reason'], state.OBSERVED)

    def test_restart_during_forecast_rearm_preserves_deadline(self):
        previous = dict(rain_episode_reason=state.CANCELLED, rain_episode_wet=False,
                        rain_dry_since=self.tick, rain_observed_at=self.tick+timedelta(minutes=15),
                        rain_forecast_value=0)
        # A missing frame/restart must not slide the deadline from 12:30 to 12:55.
        result = self.advance(previous, .2, 0, 25)
        self.assertIsNone(result['reason'])
        self.assertEqual(result['rain_dry_since'], self.tick)
        previous.update(result)
        self.assertEqual(self.advance(previous, .2, 0, 30)['reason'], state.START)

    def test_long_dry_episode_from_old_version_rearms_from_actual_closure(self):
        previous = dict(rain_episode_reason=state.START, rain_episode_wet=False,
                        rain_dry_since=self.tick, rain_observed_at=self.tick+timedelta(minutes=55),
                        rain_forecast_value=.2)
        closed = self.advance(previous, None, 0, 60)
        self.assertEqual(closed['reason'], state.CANCELLED)
        previous.update(closed, rain_episode_reason=closed['reason'])
        self.assertIsNone(self.advance(previous, .2, 0, 60)['reason'])
        self.assertEqual(closed['rain_dry_since'], self.tick+timedelta(minutes=45))

    def test_radar_and_model_order_have_identical_episode_notifications(self):
        def replay(radar_first):
            previous, sent = {}, []
            # The model predicts rain at clearance and during the rearm window.
            values = [(.2, 0), (.2, .2), (0, .2), (0, 0), (0, 0),
                      (0, .2), (0, .2), (0, .2), (0, .2), (0, .2)]
            for index, (actual, forecast) in enumerate(values):
                for value in ([None, forecast] if radar_first else [forecast, None]):
                    result = self.advance(previous, value, actual, index*5)
                    if result is None:
                        continue
                    if state.should_notify(previous, result):
                        sent.append((index*5, result['reason']))
                        previous['rain_episode_reason'] = result['reason']
                    previous.update(result)
            return sent
        expected = [(0, state.OBSERVED), (25, state.ENDED), (40, state.START)]
        self.assertEqual(replay(True), expected)
        self.assertEqual(replay(False), expected)

    def test_old_predicted_clear_episode_waits_for_confirmed_clearance(self):
        previous = dict(rain_episode_reason=state.ENDING, rain_episode_wet=True)
        self.assertIsNone(self.advance(previous, .2, .2)['reason'])
        for minute in [5, 10, 15, 20]:
            result = self.advance(previous, 0, 0, minute)
            previous.update(result)
        self.assertEqual(result['reason'], state.ENDED)


if __name__ == '__main__':
    unittest.main()
