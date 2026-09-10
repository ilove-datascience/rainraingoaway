import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
import unittest

spec = importlib.util.spec_from_file_location('policy', Path(__file__).resolve().parents[1] / 'src/telegram_code/forecast_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class ForecastPolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 10, 12, 1)
        self.prediction = {'observed_at': self.now - timedelta(minutes=1), 'next_tick': self.now + timedelta(minutes=4)}
        self.previous = {'forecast_time': self.now - timedelta(minutes=10), 'sent_at': self.now - timedelta(minutes=16), 'value': .02, 'rainy': True}

    def test_freshness_boundaries(self):
        self.assertTrue(policy.is_fresh(self.prediction, self.now))
        self.assertFalse(policy.is_fresh(self.prediction, self.prediction['next_tick']))
        self.assertFalse(policy.is_fresh(self.prediction, self.now - timedelta(minutes=2)))
        self.assertFalse(policy.is_fresh(None, self.now))

    def test_wording_uses_valid_time_and_observation(self):
        text = policy.forecast_text(.02, self.prediction)
        self.assertIn('12:05 SGT', text)
        self.assertIn('12:00', text)
        self.assertNotIn('mm', text)


if __name__ == '__main__':
    unittest.main()
