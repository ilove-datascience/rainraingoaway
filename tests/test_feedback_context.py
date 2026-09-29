"""Feedback must describe the same location, time and values as the weather shown."""
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from telegram_code import feedback, feedback_context as context
from telegram_code.local_rain import local_rain
from telegram_code.notification_text import encode_alert
from telegram_code.rain_state import OBSERVED, START


class FeedbackContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observed = datetime(2026, 9, 29, 12)
        self.target = self.observed + timedelta(minutes=5)
        self.row = dict(latitude=1.3, longitude=103.8, label='Main', location_id=0,
                        rain_settings_version=3)
        self.radar = np.arange(35, dtype=np.float32).reshape(5, 7) / 100
        self.prediction = dict(observed_at=self.observed, next_tick=self.target,
                               actual_radar=self.radar, prediction=self.radar / 2)
        self.prepare = AsyncMock(return_value='feedback-keyboard')
        prepare_patch = patch.object(context, 'prepare_feedback', self.prepare)
        prepare_patch.start()
        self.addCleanup(prepare_patch.stop)

    async def test_forecast_freezes_local_values_radius_location_and_times(self):
        rows = [self.row]
        markup = await context.forecast_feedback(-1001, rows, self.prediction)
        self.assertEqual(markup, 'feedback-keyboard')
        chat_id, snapshots = self.prepare.await_args.args
        self.assertEqual(chat_id, -1001)
        actual = snapshots[0]
        self.assertEqual(actual['kind'], 'forecast')
        self.assertEqual(actual['observed_at'], self.observed)
        self.assertEqual(actual['target_at'], self.target)
        self.assertEqual(actual['radar_value'], local_rain(self.radar, 1.3, 103.8)[0])
        expected = local_rain(self.prediction['prediction'], 1.3, 103.8)
        self.assertEqual(actual['forecast_value'], expected[0])
        self.assertEqual(actual['radius_m'], expected[2])
        self.assertEqual(actual['location_id'], 0)
        self.assertEqual(actual['settings_version'], 3)
        self.assertEqual(actual['source'], 'manual_forecast')
        self.row.update(latitude=1.4, label='Moved', rain_settings_version=4)
        self.prediction['prediction'][:] = 0
        self.assertEqual(actual['latitude'], 1.3)
        self.assertEqual(actual['label'], 'Main')
        self.assertEqual(actual['settings_version'], 3)
        self.assertEqual(actual['forecast_value'], expected[0])

    def test_snapshot_is_json_safe_and_does_not_follow_row_mutations(self):
        frozen = context.snapshot(self.row, 'forecast', self.observed, self.target,
                                  radar_value=np.float32(.1), forecast_value=np.float32(.2), radius_m=250)
        persisted = json.loads(json.dumps(frozen, allow_nan=False))
        self.row.update(longitude=104., label='Changed')
        self.assertEqual(persisted['longitude'], 103.8)
        self.assertEqual(persisted['label'], 'Main')
        self.assertEqual(persisted['target_at'], self.target.isoformat())
        self.assertEqual(persisted['radius_m'], 250)

    async def test_radar_uses_same_decoder_cleaning_orientation_and_file_time(self):
        path = Path('202609291155.png')
        # Imports inside _radar_snapshots are replaced with local dependencies;
        # no PNG, model checkpoint or feedback database is accessed.
        decoder = SimpleNamespace(decode_png=Mock(return_value='decoded-radar'), SOURCE='source-decoder')
        loading = SimpleNamespace(remove_small_echoes=Mock(return_value=self.radar))
        with patch.dict(sys.modules, {'data_processing.radar_codec': decoder,
                                      'data_processing.data_loading': loading}):
            markup = await context.radar_feedback(42, [self.row], path)
        self.assertEqual(markup, 'feedback-keyboard')
        decoder.decode_png.assert_called_once_with(path, 'source-decoder')
        loading.remove_small_echoes.assert_called_once_with('decoded-radar')
        snapshot = self.prepare.await_args.args[1][0]
        self.assertEqual(snapshot['kind'], 'radar')
        self.assertEqual(snapshot['observed_at'], datetime(2026, 9, 29, 11, 55))
        self.assertEqual(snapshot['target_at'], snapshot['observed_at'])
        value, _, radius = local_rain(np.flipud(self.radar), 1.3, 103.8)
        self.assertEqual(snapshot['radar_value'], value)
        self.assertEqual(snapshot['radius_m'], radius)
        self.assertIsNone(snapshot['forecast_value'])

    async def test_no_location_radar_overview_does_not_offer_local_feedback(self):
        with patch.object(context, '_radar_snapshots') as build:
            self.assertIsNone(await context.radar_feedback(42, [], Path('missing.png')))
        build.assert_not_called()
        self.prepare.assert_not_awaited()

    async def test_out_of_coverage_rows_never_enter_feedback_snapshots(self):
        outside = dict(self.row, latitude=2., label='Outside')
        await context.forecast_feedback(-1001, [self.row, outside], self.prediction)
        snapshots = self.prepare.await_args.args[1]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['label'], 'Main')

    async def test_feedback_preparation_failure_returns_no_keyboard(self):
        self.prepare.side_effect = OSError('storage unavailable')
        self.assertIsNone(await context.forecast_feedback(42, [self.row], self.prediction))
        with patch.object(context, '_radar_snapshots', return_value=[]):
            self.assertIsNone(await context.radar_feedback(42, [self.row], Path('unused.png')))

    async def test_legacy_and_malformed_notices_never_guess_feedback_coordinates(self):
        items = [dict(message='Main\nRAIN EXPECTED | FORECAST'),
                 dict(message=encode_alert('Main', START, 'Light')),
                 dict(message=json.dumps({'rain_alert': 1, 'feedback': {'kind': 'radar'}})),
                 dict(message='not json'), dict(message='[]')]
        self.assertIsNone(await context.notification_feedback(-1001, items))
        self.prepare.assert_not_awaited()

    async def test_mixed_notification_claims_restore_each_frozen_target(self):
        forecast = context.snapshot(self.row, 'forecast', self.observed, self.target,
                                    radar_value=0., forecast_value=.2, radius_m=250)
        office = dict(self.row, latitude=1.35, label='Office', location_id=7)
        radar = context.snapshot(office, 'radar', self.observed, self.observed,
                                 radar_value=.3, forecast_value=.4, radius_m=250)
        items = [dict(message=encode_alert('Main', START, 'Light', feedback=forecast)),
                 dict(message='legacy'),
                 dict(message=encode_alert('Office', OBSERVED, 'Light', feedback=radar))]
        await context.notification_feedback(-1001, items)
        chat, snapshots = self.prepare.await_args.args
        self.assertEqual(chat, -1001)
        self.assertEqual([row['kind'] for row in snapshots], ['forecast', 'radar'])
        self.assertEqual([row['target_at'] for row in snapshots], [self.target, self.observed])
        self.assertEqual([row['latitude'] for row in snapshots], [1.3, 1.35])
        self.assertEqual([row['radius_m'] for row in snapshots], [250, 250])

    async def test_stale_radar_cannot_create_buttons_or_write_store(self):
        stale = context._grid_snapshots([self.row], 'radar', self.observed, self.observed, radar=self.radar)
        now = (self.observed + timedelta(minutes=30)).replace(tzinfo=feedback.SGT).astimezone(timezone.utc)
        with patch.object(context, 'prepare_feedback', feedback.prepare_feedback), \
             patch.object(context, '_radar_snapshots', return_value=stale), \
             patch.object(feedback, '_utc_now', return_value=now), \
             patch.object(feedback, '_store_snapshots') as store:
            self.assertIsNone(await context.radar_feedback(42, [self.row], Path('202609291200.png')))
        store.assert_not_called()


class WeatherPhotoReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.update = SimpleNamespace(message=SimpleNamespace(reply_photo=AsyncMock(), reply_text=AsyncMock()))

    async def test_short_caption_keeps_prompt_and_buttons_on_photo(self):
        photo = BytesIO(b'photo')
        await context.reply_weather_photo(self.update, photo, 'Forecast for Main', 'buttons')
        self.update.message.reply_photo.assert_awaited_once_with(
            photo=photo, caption='Forecast for Main\n\n' + feedback.FEEDBACK_PROMPT, reply_markup='buttons')
        self.update.message.reply_text.assert_not_awaited()
        self.assertTrue(photo.closed)

    async def test_long_unicode_caption_preserves_full_text_and_feedback(self):
        photo = BytesIO(b'photo')
        caption = '🌧' * 512
        await context.reply_weather_photo(self.update, photo, caption, 'buttons')
        self.update.message.reply_photo.assert_awaited_once_with(photo=photo)
        self.update.message.reply_text.assert_awaited_once_with(
            caption + '\n\n' + feedback.FEEDBACK_PROMPT, reply_markup='buttons')
        self.assertTrue(photo.closed)

    async def test_missing_feedback_preserves_caption_and_closes_failed_send(self):
        photo = BytesIO(b'photo')
        self.update.message.reply_photo.side_effect = TimeoutError('telegram timeout')
        with self.assertRaises(TimeoutError):
            await context.reply_weather_photo(self.update, photo, 'Radar snapshot')
        self.assertEqual(self.update.message.reply_photo.await_args.kwargs['caption'], 'Radar snapshot')
        self.assertTrue(self.update.message.reply_photo.await_args.kwargs['reply_markup'].remove_keyboard)
        self.assertTrue(photo.closed)


if __name__ == '__main__':
    unittest.main()
