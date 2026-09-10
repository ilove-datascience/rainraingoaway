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

    def test_prediction_confirmation_and_end_forecast(self):
        predicted = self.advance({}, .04, 0)
        predicted['rain_alert_reason'] = predicted['reason']
        self.assertEqual(predicted['state'], 'predicted')
        confirmed = self.advance(predicted, .04, .03, 5)
        confirmed['rain_alert_reason'] = predicted['rain_alert_reason']
        self.assertEqual(confirmed['state'], 'confirmed')
        ending = self.advance(confirmed, 0, .03, 10)
        self.assertEqual(ending['state'], 'predicted norain')
        self.assertTrue(ending['radar_raining'])
        self.assertEqual(ending['reason'], 'Rain is predicted to end')

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
            (.04, .03), (0, .03), (0, 0), (0, 0), (.04, 0),
        ]):
            result = self.advance(previous, forecast, actual, i * 5)
            if state.should_notify(previous, result, self.tick):
                sent.append(result['reason'])
                previous['rain_alert_reason'] = result['reason']
            previous.update(result)
        self.assertEqual(sent, [state.START, state.ENDING, state.ENDED, state.START])

    def test_unconfirmed_prediction_cancelled_once(self):
        previous = {'rain_alert_reason': state.START, 'state': 'predicted'}
        result = self.advance(previous, 0, 0)
        self.assertEqual(result['reason'], state.CANCELLED)
        previous.update(result, rain_alert_reason=result['reason'])
        self.assertIsNone(self.advance(previous, 0, 0, 5)['reason'])


if __name__ == '__main__':
    unittest.main()
