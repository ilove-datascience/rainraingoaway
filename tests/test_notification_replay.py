import sys
import unittest
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation.notification_replay import replay_location
from evaluation.rain_diagnostics import region_diagnostics
from telegram_code.rain_state import START,ENDING,ENDED
from telegram_code.notification_text import rain_notice


class ReplayTests(unittest.TestCase):
    def records(self,values):
        return [dict(time=datetime(2026,8,10)+timedelta(minutes=5*i),actual=a,forecast=f)
                for i,(a,f) in enumerate(values)]

    def test_episode_clear_and_onset_scoring(self):
        alerts,onsets=replay_location(self.records([(0,.2),(.2,0),(.2,.2),(0,0)]))
        self.assertEqual([a['reason'] for a in alerts],[START,ENDING,ENDED])
        self.assertTrue(alerts[1]['premature_clear'])
        self.assertTrue(onsets[0]['warned_in_horizon'])
        self.assertEqual(onsets[0]['warning_lead_minutes'],5)

    def test_gap_does_not_invent_onset(self):
        rows=self.records([(0,.2),(.2,0)])
        rows[1]['time']+=timedelta(minutes=5)
        alerts,onsets=replay_location(rows)
        self.assertEqual(onsets,[])
        self.assertIsNone(alerts[0]['future_rain'])

    def test_true_predicted_clear(self):
        alerts,_=replay_location(self.records([(0,.2),(.2,0),(0,0)]))
        self.assertFalse(alerts[1]['premature_clear'])

    def test_stratification_preserves_targets_and_false_alarms(self):
        target=np.zeros((10,10));target[0,0]=.2;target[5:8,5:8]=.2
        prediction=target.copy();prediction[0,0]=0;prediction[0,8]=.2
        before=target.copy();result=region_diagnostics(target,prediction)
        np.testing.assert_array_equal(target,before)
        self.assertEqual(result['tiny_1_5']['hits'],0)
        self.assertEqual(result['small_6_20']['hits'],9)
        self.assertEqual(result['all_pixels'],dict(tp=9,fp=1,fn=1))

    def test_source_caption_is_only_for_observations(self):
        now=datetime(2026,8,10)
        self.assertIn('Light to Moderate',rain_notice('RAIN DETECTED',now,observed_category='Light to Moderate'))
        with self.assertRaises(ValueError):rain_notice('RAIN',now,forecast=now,observed_category='Light')
