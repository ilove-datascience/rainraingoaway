"""Real SQLite report persistence, time boundaries and independent group voters."""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from telegram_code import feedback


class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'reports.sqlite3'
        self.now = datetime(2026, 9, 29, 4, 0, tzinfo=timezone.utc)  # noon SGT
        for patcher in (patch.object(feedback, 'DB_PATH', self.db),
                        patch.object(feedback, '_utc_now', side_effect=lambda: self.now)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.facts = dict(kind='radar', latitude=1.31, longitude=103.8, label='Main',
                          location_id=0, settings_version=4, observed_at=self.now,
                          target_at=self.now, radar_value=.22, forecast_value=None,
                          radius_m=250, source='manual_radar')

    def rows(self, table):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute('SELECT * FROM ' + table)]

    async def buttons(self, **changes):
        return await feedback.prepare_feedback(-1001, [dict(self.facts, **changes)])

    def update(self, data, chat_id=-1001, user_id=42):
        user = SimpleNamespace(id=user_id, is_bot=False)
        chat = SimpleNamespace(id=chat_id)
        query = SimpleNamespace(data=data, from_user=user,
                                message=SimpleNamespace(chat=chat, message_id=123),
                                answer=AsyncMock())
        return SimpleNamespace(callback_query=query, effective_chat=chat, effective_user=user)

    async def vote(self, markup, wet=False, **kwargs):
        button = markup.inline_keyboard[int(wet)][0]
        update = self.update(button.callback_data, **kwargs)
        await feedback.handle_feedback(update, None)
        self.assertFalse(update.callback_query.answer.await_args.kwargs['show_alert'])
        return update.callback_query.answer.await_args.args[0]

    async def test_frozen_snapshot_and_small_opaque_callbacks(self):
        markup = await self.buttons()
        for row in markup.inline_keyboard:
            self.assertEqual(len(row), 1)
            self.assertLessEqual(len(row[0].callback_data.encode()), 64)
            self.assertNotIn('103.8', row[0].callback_data)
            self.assertIn('Radar 12:00', row[0].text)
        saved = self.rows('feedback_snapshots')[0]
        self.assertEqual((saved['latitude'], saved['longitude'], saved['radius_m']), (1.31, 103.8, 250))
        self.assertEqual(saved['settings_version'], 4)
        self.facts.update(label='Moved', latitude=1.4)
        self.assertEqual(await self.vote(markup), 'Saved: Main — not raining. Thanks!')
        self.assertEqual(self.rows('feedback_snapshots')[0]['label'], 'Main')

    async def test_repeat_vote_and_correction_replace_only_own_report(self):
        markup = await self.buttons()
        self.assertEqual(await self.vote(markup), 'Saved: Main — not raining. Thanks!')
        first = self.rows('feedback_reports')[0]
        self.assertEqual(first['raining'], 0)
        self.now += timedelta(minutes=2)
        await self.vote(markup)
        self.assertEqual(len(self.rows('feedback_reports')), 1)
        self.assertEqual(await self.vote(markup, wet=True), 'Saved: Main — raining. Thanks!')
        corrected = self.rows('feedback_reports')[0]
        self.assertEqual(corrected['raining'], 1)
        self.assertEqual(corrected['first_reported_at'], first['first_reported_at'])
        self.assertEqual(corrected['reported_at'], self.now.isoformat())
        self.assertEqual(corrected['message_id'], 123)

    async def test_group_members_can_disagree_without_overwriting_each_other(self):
        markup = await self.buttons()
        await asyncio.gather(self.vote(markup, user_id=41), self.vote(markup, wet=True, user_id=42))
        reports = sorted(self.rows('feedback_reports'), key=lambda row: row['user_id'])
        self.assertEqual([(row['user_id'], row['raining']) for row in reports], [(41, 0), (42, 1)])

    async def test_forecast_opens_at_target_and_closes_after_thirty_minutes(self):
        target = self.now + timedelta(minutes=5)
        markup = await self.buttons(kind='forecast', target_at=target, forecast_value=.2)
        self.assertIn('Forecast 12:05', markup.inline_keyboard[0][0].text)
        self.assertIn('Please report then', await self.vote(markup))
        self.assertFalse(self.rows('feedback_reports'))
        self.now = target
        self.assertEqual(await self.vote(markup), 'Saved: Main — not raining. Thanks!')
        self.now = target + timedelta(minutes=30)
        self.assertIn('closed', await self.vote(markup, wet=True))
        self.assertEqual(self.rows('feedback_reports')[0]['raining'], 0)

    async def test_expired_and_future_radar_do_not_get_buttons(self):
        self.assertIsNone(await self.buttons(observed_at=self.now - timedelta(minutes=30),
                                             target_at=self.now - timedelta(minutes=30)))
        self.assertIsNone(await self.buttons(observed_at=self.now + timedelta(minutes=1),
                                             target_at=self.now + timedelta(minutes=1)))
        self.assertFalse(self.db.exists())

    async def test_existing_radar_button_expires(self):
        markup = await self.buttons()
        self.now += timedelta(minutes=30)
        self.assertIn('closed', await self.vote(markup))
        self.assertFalse(self.rows('feedback_reports'))

    async def test_naive_singapore_time_matches_aware_utc_time(self):
        markup = await self.buttons(observed_at=datetime(2026, 9, 29, 12),
                                    target_at=datetime(2026, 9, 29, 12))
        self.assertEqual(await self.vote(markup), 'Saved: Main — not raining. Thanks!')
        self.assertEqual(self.rows('feedback_snapshots')[0]['target_at'], self.now.isoformat())

    async def test_forwarded_cross_chat_button_cannot_submit(self):
        markup = await self.buttons()
        self.assertIn('does not belong', await self.vote(markup, chat_id=-999))
        self.assertFalse(self.rows('feedback_reports'))

    async def test_malformed_unknown_and_unverified_callbacks_do_not_write_reports(self):
        markup = await self.buttons()
        for data in ('rainfb:bad:wet', 'rainfb:' + 'a' * 16 + ':wet', None):
            update = self.update(data)
            await feedback.handle_feedback(update, None)
            self.assertNotIn('Saved:', update.callback_query.answer.await_args.args[0])
        update = self.update(markup.inline_keyboard[0][0].callback_data)
        update.effective_user = SimpleNamespace(id=999)
        await feedback.handle_feedback(update, None)
        self.assertIn('could not be verified', update.callback_query.answer.await_args.args[0])
        self.assertFalse(self.rows('feedback_reports'))

    async def test_invalid_snapshot_does_not_create_database(self):
        for changed in (dict(latitude=float('nan')), dict(longitude=float('inf')),
                        dict(radius_m=-1), dict(kind='unknown'), dict(label=''),
                        dict(target_at=self.now - timedelta(minutes=1)), dict(display_index=7)):
            with self.subTest(changed=changed):
                self.assertIsNone(await self.buttons(**changed))
        self.assertFalse(self.db.exists())

    async def test_store_failure_is_optional_and_failed_report_is_honest(self):
        with patch.object(feedback, '_store_snapshots', side_effect=sqlite3.OperationalError('locked')):
            self.assertIsNone(await self.buttons())
        markup = await self.buttons()
        await self.vote(markup)
        with patch.object(feedback, '_record_report', side_effect=sqlite3.OperationalError('locked')):
            self.assertIn('could not be saved', await self.vote(markup, wet=True))
        self.assertEqual(self.rows('feedback_reports')[0]['raining'], 0)

    async def test_new_module_instance_can_read_buttons_from_before_restart(self):
        markup = await self.buttons()
        spec = importlib.util.spec_from_file_location('restarted_feedback', feedback.__file__)
        restarted = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(restarted)
        restarted.DB_PATH = self.db
        restarted._utc_now = lambda: self.now
        update = self.update(markup.inline_keyboard[0][0].callback_data)
        await restarted.handle_feedback(update, None)
        update.callback_query.answer.assert_awaited_once_with('Saved: Main — not raining. Thanks!', show_alert=False)
        self.assertEqual(len(self.rows('feedback_reports')), 1)

    async def test_numbered_full_width_buttons_disambiguate_similar_long_labels(self):
        snapshots = [dict(self.facts, label='Very long matching prefix home', display_index=2),
                     dict(self.facts, label='Very long matching prefix work', display_index=3)]
        markup = await feedback.prepare_feedback(-1001, snapshots)
        self.assertTrue(markup.inline_keyboard[0][0].text.startswith('2. Radar'))
        self.assertTrue(markup.inline_keyboard[2][0].text.startswith('3. Radar'))
        self.assertEqual(await self.vote(SimpleNamespace(inline_keyboard=markup.inline_keyboard[2:])),
                         'Saved: Very long matching prefix work — not raining. Thanks!')
        report = self.rows('feedback_reports')[0]
        saved = next(row for row in self.rows('feedback_snapshots') if row['token'] == report['token'])
        self.assertEqual(saved['label'], 'Very long matching prefix work')


if __name__ == '__main__':
    unittest.main()
