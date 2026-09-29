"""Exercise automatic alert production without importing model/Telegram runtimes."""
import ast
import asyncio
import copy
from datetime import datetime, timedelta
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_processing.radar_codec import source_category
from telegram_code.forecast_policy import is_fresh
from telegram_code.feedback_context import snapshot
from telegram_code.notification_text import encode_alert, intensity_label
from telegram_code.rain_state import CANCELLED, ENDED, OBSERVED, START, next_rain_state, should_notify


class AutoNoticeBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observed = datetime(2026, 9, 28, 12)
        self.now = self.observed + timedelta(seconds=30)
        self.rows = [
            dict(userid=-1001, location_id=0, rain_settings_version=2,
                 label='Main', latitude=1.31, longitude=103.8, mode='automatic'),
            dict(userid=-1001, location_id=4, rain_settings_version=3,
                 label='Kaixin', latitude=1.33, longitude=103.9, mode='automatic'),
        ]
        self.actual = {1.31: .22, 1.33: .37}
        self.forecast = {1.31: .2, 1.33: .4}
        self.prediction = dict(actual_radar=self.actual, prediction=self.forecast,
                               observed_at=self.observed,
                               next_tick=self.observed + timedelta(minutes=5))
        self.context = SimpleNamespace(application=SimpleNamespace(bot_data={}))
        self.save = Mock(return_value=True)
        self.map_images = []

        def image_for_rows(rows, grid, timestamp, actual=False):
            image = BytesIO(b'shared-map')
            self.map_images.append(image)
            return image, 'Unused manual-map caption'

        self.map = Mock(side_effect=image_for_rows)
        self.clock = Mock(side_effect=lambda: self.now)
        self.locations = Mock(side_effect=lambda: copy.deepcopy(self.rows))
        self.env = dict(
            asyncio=asyncio, timedelta=timedelta,
            get_rain_locations=self.locations,
            in_coverage=lambda latitude, longitude: True,
            local_rain=lambda grid, latitude, longitude: (grid[latitude], (2, 3), 250),
            next_rain_state=next_rain_state, should_notify=should_notify,
            sg_now=self.clock, is_fresh=lambda item: is_fresh(item, self.now),
            START=START, OBSERVED=OBSERVED, ENDED=ENDED,
            source_category=source_category, intensity_label=intensity_label,
            encode_alert=encode_alert, snapshot=snapshot, save_rain_state=self.save,
            build_group_map=self.map,
        )
        source = Path(__file__).resolve().parents[1] / 'src/telegram_code/methods.py'
        names = {'send_auto_update', '_send_auto_update'}
        functions = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
                     if isinstance(node, ast.AsyncFunctionDef) and node.name in names]
        self.assertEqual(len(functions), 2)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'auto-notice-producer', 'exec'), self.env)

    async def produce(self):
        return await self.env['send_auto_update'](self.context, self.prediction)

    def messages(self):
        return [json.loads(call.args[2]) for call in self.save.call_args_list if call.args[2]]

    async def test_two_locations_share_one_actual_map_and_one_available_time(self):
        # Clocks cross a second boundary while the two locations are evaluated.
        # A separate available_at per row can make the delivery loop split a chat.
        self.clock.side_effect = [self.now + timedelta(microseconds=900000),
                                  self.now + timedelta(seconds=1, microseconds=100000),
                                  self.now + timedelta(seconds=2)]
        self.assertTrue(await self.produce())
        self.map.assert_called_once()
        args, kwargs = self.map.call_args
        self.assertEqual([row['location_id'] for row in args[0]], [0, 4])
        self.assertIs(args[1], self.actual)
        self.assertEqual(args[2], self.observed)
        self.assertEqual(kwargs, {'actual': True})
        self.assertEqual(self.save.call_count, 2)
        calls = self.save.call_args_list
        self.assertEqual(calls[0].args[3], calls[1].args[3])
        self.assertEqual(calls[0].args[3], self.now + timedelta(seconds=17))
        self.assertEqual([call.args[4] for call in calls], [b'shared-map'] * 2)
        self.assertTrue(self.map_images[0].closed)
        payloads = self.messages()
        self.assertEqual([item['label'] for item in payloads], ['Main', 'Kaixin'])
        self.assertEqual([item['intensity'] for item in payloads], ['Light', 'Light to Moderate'])
        self.assertTrue(all(item['reason'] == OBSERVED for item in payloads))
        self.assertTrue(all(item['photo_locations'] == [[0, 2], [4, 3]] for item in payloads))
        self.assertEqual([item['feedback']['location_id'] for item in payloads], [0, 4])
        self.assertEqual([item['feedback']['radar_value'] for item in payloads], [.22, .37])
        self.assertTrue(all(item['feedback']['kind'] == 'radar' for item in payloads))
        self.assertTrue(all(item['feedback']['radius_m'] == 250 for item in payloads))

    async def test_distinct_chats_do_not_share_maps_or_photo_scope(self):
        self.rows[1]['userid'] = -1002
        self.assertTrue(await self.produce())
        self.assertEqual(self.map.call_count, 2)
        self.assertEqual([len(call.args[0]) for call in self.map.call_args_list], [1, 1])
        self.assertEqual([item['photo_locations'] for item in self.messages()], [[[0, 2]], [[4, 3]]])

    async def test_observation_only_closes_never_wet_forecast_without_numeric_intensity(self):
        self.rows = self.rows[:1]
        self.rows[0].update(rain_episode_reason=START, rain_episode_wet=False,
                            radar_raining=False,
                            rain_observed_at=self.observed - timedelta(minutes=5),
                            rain_dry_since=self.observed - timedelta(minutes=15))
        self.actual[1.31] = 0
        self.prediction.pop('prediction')
        self.assertTrue(await self.produce())
        saved = self.save.call_args.args
        self.assertEqual(saved[1]['reason'], CANCELLED)
        self.assertIsNone(saved[1]['rain_forecast_value'])
        self.assertEqual(self.messages()[0], {
            'rain_alert': 1, 'label': 'Main', 'reason': CANCELLED, 'intensity': None,
            'photo_locations': [[0, 2]],
            'feedback': {
                'kind': 'radar', 'latitude': 1.31, 'longitude': 103.8, 'label': 'Main',
                'location_id': 0, 'settings_version': 2,
                'observed_at': self.observed.isoformat(), 'target_at': self.observed.isoformat(),
                'radar_value': 0., 'forecast_value': None, 'radius_m': 250,
                'source': 'automatic_notice',
            },
        })
        self.map.assert_called_once()
        self.assertTrue(self.map.call_args.kwargs['actual'])

    async def test_mixed_forecast_and_observed_claims_keep_distinct_feedback_times(self):
        self.actual[1.31] = 0.
        self.assertTrue(await self.produce())
        forecast, observed = self.messages()
        self.assertEqual(forecast['reason'], START)
        self.assertEqual(observed['reason'], OBSERVED)
        self.assertEqual(forecast['feedback']['kind'], 'forecast')
        self.assertEqual(forecast['feedback']['target_at'], self.prediction['next_tick'].isoformat())
        self.assertEqual(forecast['feedback']['radar_value'], 0.)
        self.assertEqual(forecast['feedback']['forecast_value'], .2)
        self.assertEqual(observed['feedback']['kind'], 'radar')
        self.assertEqual(observed['feedback']['target_at'], self.observed.isoformat())
        self.assertEqual(observed['feedback']['radar_value'], .37)
        self.assertEqual(observed['feedback']['forecast_value'], .4)
        self.assertEqual([forecast['feedback']['radius_m'], observed['feedback']['radius_m']], [250, 250])
        # The shared picture is actual radar, but must not relabel the future claim.
        self.assertIs(self.map.call_args.args[1], self.actual)
        self.assertTrue(self.map.call_args.kwargs['actual'])
        self.rows[0].update(latitude=1.4, longitude=104., label='Moved')
        self.assertEqual(self.messages()[0]['feedback']['latitude'], 1.31)
        self.assertEqual(self.messages()[0]['feedback']['label'], 'Main')

    async def test_map_failure_does_not_advance_affected_locations_and_retry_succeeds(self):
        initial = copy.deepcopy(self.rows)
        self.map.side_effect = ValueError('Map unavailable')
        with self.assertRaisesRegex(RuntimeError, 'keeping timestamp for retry'):
            await self.produce()
        self.save.assert_not_called()
        self.assertEqual(self.rows, initial)
        self.map.side_effect = lambda *args, **kwargs: (BytesIO(b'retry-map'), 'caption')
        self.assertTrue(await self.produce())
        self.assertEqual(self.save.call_count, 2)
        self.assertTrue(all(item['reason'] == OBSERVED for item in self.messages()))
        self.assertTrue(all(call.args[4] == b'retry-map' for call in self.save.call_args_list))

    async def test_stale_or_future_observation_updates_state_without_notification(self):
        self.rows = self.rows[:1]
        for now in (self.observed - timedelta(seconds=1), self.observed + timedelta(minutes=10)):
            with self.subTest(now=now):
                self.now = now
                self.save.reset_mock()
                self.assertTrue(await self.produce())
                self.assertEqual(self.save.call_count, 1)
                self.assertIsNone(self.save.call_args.args[2])
                self.assertIsNone(self.save.call_args.args[4])
        self.map.assert_not_called()

    async def test_forecast_expiry_suppresses_start_but_observed_rain_remains_fresh(self):
        self.rows = self.rows[:1]
        self.actual[1.31] = 0
        self.now = self.prediction['next_tick']
        self.assertTrue(await self.produce())
        self.assertEqual(self.save.call_args.args[1]['reason'], START)
        self.assertIsNone(self.save.call_args.args[2])
        self.map.assert_not_called()
        self.actual[1.31] = .22
        self.save.reset_mock()
        self.assertTrue(await self.produce())
        self.assertEqual(self.messages()[0]['reason'], OBSERVED)
        self.map.assert_called_once()

    async def test_old_clearance_is_not_queued_as_a_fresh_notice(self):
        self.rows = self.rows[:1]
        self.rows[0].update(rain_episode_reason=OBSERVED, rain_episode_wet=True,
                            rain_observed_at=self.observed - timedelta(minutes=5),
                            rain_dry_since=self.observed - timedelta(minutes=15))
        self.actual[1.31] = 0
        self.prediction.pop('prediction')
        self.now = self.observed + timedelta(minutes=10)
        self.assertTrue(await self.produce())
        self.assertEqual(self.save.call_args.args[1]['reason'], ENDED)
        self.assertIsNone(self.save.call_args.args[2])
        self.map.assert_not_called()

    async def test_manual_mode_still_updates_state_without_queuing_or_rendering(self):
        for row in self.rows:
            row['mode'] = 'manual'
        self.assertTrue(await self.produce())
        self.assertEqual(self.save.call_count, 2)
        self.assertTrue(all(call.args[2] is None for call in self.save.call_args_list))
        self.map.assert_not_called()

    async def test_missing_observed_radar_does_not_advance_state(self):
        self.prediction.pop('actual_radar')
        self.assertFalse(await self.produce())
        self.locations.assert_not_called()
        self.save.assert_not_called()
        self.map.assert_not_called()


if __name__ == '__main__':
    unittest.main()
