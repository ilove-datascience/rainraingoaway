import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code.local_rain import local_rain, in_coverage
from telegram_code.notification_text import alert_text, intensity_label
from telegram_code.rain_state import ENDING, ENDED
from datetime import datetime, timedelta


class Grid:
    shape = (120,217)
    def __init__(self):
        self.values = {}
    def __getitem__(self, key):
        return self.values.get(key,0.)


class LocalRainTests(unittest.TestCase):
    def test_neighbour_included_diagonal_excluded(self):
        grid = Grid()
        _, (x,y), radius = local_rain(grid,1.3,103.8)
        self.assertEqual(radius,300)
        grid.values[y,x+1] = .1
        grid.values[y+1,x+1] = .9
        self.assertEqual(local_rain(grid,1.3,103.8)[0],.1)

    def test_edges_and_outside(self):
        local_rain(Grid(),1.156,103.565)
        self.assertFalse(in_coverage(50,100))
        with self.assertRaises(ValueError):
            local_rain(Grid(),50,100)

    def test_sources_are_clear_and_not_repetitive(self):
        now=datetime(2026,9,10,20,30)
        text=alert_text(ENDING,now,now+timedelta(minutes=5),300)
        self.assertTrue(text.startswith('RAIN EXPECTED TO CLEAR | FORECAST\n'))
        actual=alert_text(ENDED,now,now+timedelta(minutes=5),300)
        self.assertTrue(actual.startswith('RAIN CLEARED | ACTUAL RADAR\n'))
        self.assertNotIn('Forecast for:',actual)

    def test_relative_intensity_bands(self):
        for value, label in [(0,'No rain'),(.003,'Light'),(.2,'Light'),(1/3,'Moderate'),(.5,'Moderate'),(2/3,'Heavy'),(1.2,'Heavy')]:
            self.assertEqual(intensity_label(value),label)
        self.assertEqual(intensity_label(.01,actual=True),'No rain')
        with self.assertRaises(ValueError):
            intensity_label(float('nan'))

    def test_caption_contains_relative_intensity(self):
        now=datetime(2026,9,10,20,30)
        text=alert_text(ENDING,now,now+timedelta(minutes=5),300,0)
        self.assertIn('Expected intensity: No rain (radar scale)',text)
        self.assertLess(len(text),1024)


if __name__ == '__main__':
    unittest.main()
