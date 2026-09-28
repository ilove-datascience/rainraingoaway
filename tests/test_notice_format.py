import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code.notification_text import compose_notice, encode_alert
from telegram_code.rain_state import START, ENDING, ENDED, CANCELLED, OBSERVED
from data_processing.radar_codec import SOURCE_CATEGORIES


class NoticeFormatTests(unittest.TestCase):
    def event(self, label='Main', reason=OBSERVED, intensity=None, minute=0):
        observed = datetime(2026, 9, 28, 14, minute)
        return dict(message=encode_alert(label, reason, intensity), reason=reason,
                    observed_at=observed, forecast_at=observed + timedelta(minutes=5))

    def test_single_location_detected_has_one_source_and_no_forecast(self):
        text = compose_notice([self.event(intensity='Light')])
        self.assertEqual(text, 'Rain update\n\n• Main: Light rain detected.\n\n'
                         'Radar: 28 Sep, 14:00 SGT\n'
                         'Intensity is relative to the radar colour scale.')
        self.assertNotIn('Forecast:', text)

    def test_two_locations_share_header_and_radar_timestamp(self):
        text = compose_notice([self.event('Home'), self.event('Office', ENDED)])
        self.assertIn('• Home: Rain detected.', text)
        self.assertIn('• Office: Rain cleared (radar dry for 15 minutes).', text)
        self.assertEqual(text.count('Rain update'), 1)
        self.assertEqual(text.count('Radar:'), 1)

    def test_forecast_is_explicit_for_expected_rain_only(self):
        text = compose_notice([self.event('Home', START, 'Moderate'),
                               self.event('Office', OBSERVED)])
        self.assertIn('• Home: Moderate rain expected.', text)
        self.assertIn('• Office: Rain detected.', text)
        self.assertIn('Forecast: 28 Sep, 14:05 SGT', text)
        self.assertNotIn('Forecast:', compose_notice([self.event(reason=ENDING)]))

    def test_mixed_radar_times_stay_attached_to_locations(self):
        text = compose_notice([self.event('Home'), self.event('Office', ENDED, minute=5)])
        self.assertIn('• Home: Rain detected.\n  Radar: 28 Sep, 14:00 SGT', text)
        self.assertIn('• Office: Rain cleared (radar dry for 15 minutes).\n'
                      '  Radar: 28 Sep, 14:05 SGT', text)

    def test_mixed_forecast_times_stay_attached_to_locations(self):
        text = compose_notice([self.event('Home', START), self.event('Office', START, minute=5)])
        self.assertIn('• Home: Rain expected.\n  Radar: 28 Sep, 14:00 SGT'
                      ' · Forecast: 28 Sep, 14:05 SGT', text)
        self.assertIn('• Office: Rain expected.\n  Radar: 28 Sep, 14:05 SGT'
                      ' · Forecast: 28 Sep, 14:10 SGT', text)

    def test_cancellation_describes_expired_forecast_without_claiming_it_rained(self):
        text = compose_notice([self.event(reason=CANCELLED)])
        self.assertIn('Earlier rain forecast has passed; no rain detected.', text)
        self.assertNotIn('cleared', text)
        self.assertNotIn('Forecast:', text)

    def test_legacy_prefixed_photo_caption_keeps_location_and_intensity(self):
        item = self.event('ignored', START)
        item['message'] = ('kaixin\nRAIN EXPECTED | FORECAST\n\n'
                           'Expected intensity: Heavy (radar scale)\n'
                           'Forecast for: 28 Sep, 14:05 SGT')
        self.assertIn('• kaixin: Heavy rain expected.', compose_notice([item]))

    def test_legacy_caption_without_location_does_not_use_title_as_label(self):
        item = self.event()
        item['message'] = 'RAIN DETECTED | ACTUAL RADAR\n\nObserved intensity: Light (radar scale)'
        self.assertIn('• Saved location: Light rain detected.', compose_notice([item]))

    def test_label_is_normalized_and_bounded_without_clipping_the_notice(self):
        value = json.loads(encode_alert('  Home\n  upstairs\t balcony  ', OBSERVED))
        self.assertEqual(value['label'], 'Home upstairs balcony')
        label = '🌧️' * 80
        value = json.loads(encode_alert(label, OBSERVED))
        self.assertEqual(len(value['label']), 80)
        items = [self.event(str(i) + label, ENDED) for i in range(6)]
        text = compose_notice(items)
        self.assertGreater(len(text.encode('utf-16-le')) // 2, 1024)
        for i in range(6):
            self.assertIn('• ' + json.loads(items[i]['message'])['label'] + ':', text)
        self.assertEqual(text.count('radar dry for 15 minutes'), 6)

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_alert('Home', OBSERVED, 'Torrential\nInjected status')

    def test_observed_rain_keeps_every_source_radar_category(self):
        for category in set(SOURCE_CATEGORIES):
            with self.subTest(category=category):
                self.assertIn(f'• Main: {category} rain detected.',
                              compose_notice([self.event(intensity=category)]))
        with self.assertRaises(ValueError):
            encode_alert('Home', START, 'Light to Moderate')

    def test_legacy_actual_caption_uses_current_label_and_source_category(self):
        item = self.event()
        item['label'] = 'Office'
        item['message'] = ('RAIN DETECTED | ACTUAL RADAR\n\n'
                           'Observed intensity: Moderate to Heavy (source radar scale)')
        self.assertIn('• Office: Moderate to Heavy rain detected.', compose_notice([item]))

    def test_photo_scope_is_preserved_as_metadata_only(self):
        item = self.event()
        ordinary = compose_notice([item])
        item['message'] = encode_alert('Main', OBSERVED, photo_locations=[[0, 3], [12, 4]])
        self.assertEqual(json.loads(item['message'])['photo_locations'], [[0, 3], [12, 4]])
        self.assertEqual(compose_notice([item]), ordinary)
        self.assertNotIn('photo_locations', json.loads(encode_alert('Main', OBSERVED)))
        for invalid in ([[0]], [[0, '3']], [[0, -1]], [[True, 3]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encode_alert('Main', OBSERVED, photo_locations=invalid)

    def test_optional_outlook_is_supplied_verbatim_once(self):
        outlook = 'Today · NEA: Thundery showers in the afternoon.\n24–32°C · Issued 05:00 SGT'
        text = compose_notice([self.event('Home'), self.event('Office')], outlook)
        self.assertTrue(text.endswith('\n\n' + outlook))
        self.assertEqual(text.count('Today · NEA:'), 1)
        self.assertNotIn('Today', compose_notice([self.event()]))

    def test_aware_times_are_converted_to_singapore(self):
        item = self.event()
        item['observed_at'] = datetime(2026, 9, 28, 6, tzinfo=timezone.utc)
        self.assertIn('Radar: 28 Sep, 14:00 SGT', compose_notice([item]))

    def test_empty_batch_is_empty(self):
        self.assertEqual(compose_notice([], 'Outlook alone'), '')


if __name__ == '__main__':
    unittest.main()
