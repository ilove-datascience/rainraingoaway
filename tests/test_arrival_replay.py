import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_arrival_notifications import score_run

class ArrivalReplayTests(unittest.TestCase):
    def test_confirmation_waits_and_resets(self):
        r=score_run(list(range(9)),[False]*4+[True]*5,[.9,.1,.9,.9,0,0,0,0,0],15,.5,2)
        self.assertEqual(r['alerts'],1)
        self.assertEqual(r['alert_records'][0]['index'],3)
        self.assertEqual(r['lead_minutes'],[5])
        r=score_run(list(range(8)),[False]*8,[.9,0,.9,0,.9,0,0,0],15,.5,2)
        self.assertEqual(r['alerts'],0)

    def test_warning_suppressed_until_onset(self):
        r=score_run(list(range(9)),[False]*3+[True]*6,[.9]*9,15,.5)
        self.assertEqual(r['alerts'],1)
        self.assertEqual(r['true_alerts'],1)
        self.assertEqual(r['warned_onsets'],1)
        self.assertEqual(r['lead_minutes'],[15])

    def test_expiry_and_right_censoring(self):
        r=score_run(list(range(8)),[False]*8,[.9]*8,15,.5)
        self.assertEqual(r['alerts'],2) # indexes 0,3; index 6 lacks verification
        self.assertEqual(r['false_alerts'],2)

    def test_no_warning_and_late_onset(self):
        r=score_run(list(range(8)),[False]*4+[True]*4,[.9]+[0]*7,15,.5)
        self.assertEqual(r['false_alerts'],1)
        self.assertEqual(r['onsets'],1)
        self.assertEqual(r['warned_onsets'],0)

if __name__=='__main__':unittest.main()
