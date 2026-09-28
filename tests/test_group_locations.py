"""Group-location handler checks plus opt-in MySQL tests using temporary tables only."""
import ast
import asyncio
import json
from datetime import datetime, timedelta
from io import BytesIO
import os
from pathlib import Path
import secrets
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

ROOT = Path(__file__).resolve().parents[1]


def functions(path, env, names=None):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), env)
    return env


class GroupHandlersTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rows = [dict(location_id=0, label='Main', latitude=1.3, longitude=103.8),
                     dict(location_id=7, label='Office', latitude=1.35, longitude=103.9)]
        self.env = functions('src/telegram_code/group_locations.py', dict(
            ReplyKeyboardRemove=lambda: "removed", asyncio=asyncio, ConversationHandler=SimpleNamespace(END=-1), MAX_CHAT_LOCATIONS=6, WAITING_FOR_EXTRA_LOCATION=3, WAITING_FOR_LOCATION_NAME=4, WAITING_FOR_REMOVAL_NAME=5,
            ForceReply=lambda: None, main_menu=lambda *args: None,
            in_coverage=lambda *args: True, settings_lock=lambda ctx: asyncio.Lock()))
        self.env.update(list_locations=Mock(return_value=self.rows), add_named_location=Mock(return_value=True))
        self.context = SimpleNamespace(args=['School'], chat_data={}, application=SimpleNamespace(bot_data={}))
        self.update = SimpleNamespace(effective_chat=SimpleNamespace(id=-1001, type='supergroup'),
            message=SimpleNamespace(reply_text=AsyncMock(), reply_photo=AsyncMock(),
                                    location=SimpleNamespace(latitude=1.3, longitude=103.8)))

    async def test_menu_reopens_buttons_and_clears_pending_setup(self):
        self.context.chat_data['new_location_label'] = 'Unfinished'
        env = functions('src/telegram_code/menus.py', dict(
            ReplyKeyboardMarkup=lambda rows, **kwargs: rows,
            ConversationHandler=SimpleNamespace(END=-1)))
        self.assertEqual(await env['show_menu'](self.update, self.context), -1)
        self.assertNotIn('new_location_label', self.context.chat_data)
        self.assertIn(['Add location', 'Saved locations'], self.update.message.reply_text.await_args.kwargs['reply_markup'])

    async def test_completion_and_cancel_remove_keyboard(self):
        await self.env['add_location_command'](self.update, self.context)
        await self.env['receive_extra_location'](self.update, self.context)
        self.assertEqual(self.update.message.reply_text.await_args.kwargs['reply_markup'], 'removed')
        await self.env['cancel_location'](self.update, self.context)
        self.assertEqual(self.update.message.reply_text.await_args.kwargs['reply_markup'], 'removed')

    async def test_button_add_and_remove_flow(self):
        self.context.args = None
        self.assertEqual(await self.env['add_location_button'](self.update, self.context), 4)
        self.update.message.text = 'School'
        self.assertEqual(await self.env['receive_location_name'](self.update, self.context), 3)
        self.assertEqual(await self.env['receive_extra_location'](self.update, self.context), -1)
        self.assertEqual(await self.env['remove_location_button'](self.update, self.context), 5)
        self.env['remove_named_location'] = Mock(return_value=True)
        self.assertEqual(await self.env['receive_removal_name'](self.update, self.context), -1)
        self.env['remove_named_location'].assert_called_once_with(-1001, 'School')

    def test_menu_buttons_in_private_and_group_chats(self):
        env = functions('src/telegram_code/menus.py', dict(ReplyKeyboardMarkup=lambda rows, **kwargs: rows))
        group = sum(env['main_menu'](-1001), [])
        private = sum(env['main_menu'](42), [])
        for label in ('Add location', 'Saved locations', 'Remove location'):
            self.assertIn(label, group)
            self.assertIn(label, private)

    async def test_add_flow_keeps_name_and_chat(self):
        self.assertEqual(await self.env['add_location_command'](self.update, self.context), 3)
        self.assertEqual(await self.env['receive_extra_location'](self.update, self.context), -1)
        self.env['add_named_location'].assert_called_once_with(-1001, 'School', 1.3, 103.8)
        self.assertNotIn('new_location_label', self.context.chat_data)

    async def test_private_chat_supported_and_duplicate_names_rejected(self):
        self.update.effective_chat.type = 'private'
        self.update.effective_chat.id = 42
        self.assertEqual(await self.env['add_location_command'](self.update, self.context), 3)
        self.assertEqual(await self.env['receive_extra_location'](self.update, self.context), -1)
        self.env['add_named_location'].assert_called_once_with(42, 'School', 1.3, 103.8)
        self.update.effective_chat.type = 'group'
        self.context.args = ['office']
        self.assertEqual(await self.env['add_location_command'](self.update, self.context), -1)
        self.assertNotIn('new_location_label', self.context.chat_data)

    async def test_private_multiple_locations_route_to_combined_map(self):
        self.update.effective_chat.id = 42
        self.update.effective_chat.type = 'private'
        self.update.message.text = 'My forecast'
        combined = AsyncMock()
        env = functions('src/telegram_code/methods.py', dict(asyncio=asyncio,
            list_locations=lambda chat: self.rows, handle_group_forecasts=combined), {'handle_msg'})
        await env['handle_msg'](self.update, self.context, None, '.')
        combined.assert_awaited_once_with(self.update, self.context, None, '.', None)

    async def test_group_sends_one_image_and_closes_it(self):
        image = BytesIO(b'group-map')
        prediction = {'prediction': 'grid', 'next_tick': datetime(2026,9,17,12)}
        env = functions('src/telegram_code/methods.py', dict(asyncio=asyncio,
            ReplyKeyboardRemove=lambda: 'removed', list_locations=lambda chat: self.rows,
            is_fresh=bool, build_group_map=Mock(return_value=(image, '1. Main: No rain expected\n2. Office: Rain expected')),
            run_model=AsyncMock(return_value=prediction)), {'handle_group_forecasts'})
        await env['handle_group_forecasts'](self.update, self.context, None, '.')
        self.update.message.reply_photo.assert_awaited_once()
        self.update.message.reply_text.assert_not_awaited()
        self.assertTrue(image.closed)
        env['build_group_map'].assert_called_once_with(self.rows, 'grid', prediction['next_tick'])

    async def test_delayed_group_uses_one_combined_fallback(self):
        fallback = AsyncMock()
        env = functions('src/telegram_code/methods.py', dict(asyncio=asyncio,
            ReplyKeyboardRemove=lambda: 'removed', list_locations=lambda chat: self.rows,
            is_fresh=bool, run_model=AsyncMock(return_value=None), send_group_actual=fallback), {'handle_group_forecasts'})
        await env['handle_group_forecasts'](self.update, self.context, None, '.')
        fallback.assert_awaited_once_with(self.update, '.', self.rows)
        self.update.message.reply_photo.assert_not_awaited()

    def test_group_map_colours_locations_and_omits_area(self):
        render = Mock(return_value=BytesIO(b'map'))
        env = functions('src/telegram_code/methods.py', dict(
            in_coverage=lambda *a: True, local_rain=Mock(side_effect=[(0,(1,2),3),(.5,(3,4),3)]),
            render_heatmap=render, intensity_label=lambda value: 'Moderate'), {'build_group_map'})
        _, caption = env['build_group_map'](self.rows, 'grid', datetime(2026,9,17,12))
        self.assertEqual(render.call_args.kwargs['marker'], [('Main', '#e83288', (1,2)), ('Office', '#0072b2', (3,4))])
        self.assertIn('Main: No rain expected', caption)
        self.assertNotIn('1.', caption)
        self.assertIn('Office: Rain expected — moderate (radar scale)', caption)
        self.assertIn('Main: No rain expected\n', caption)
        self.assertNotIn('Main: No rain expected —', caption)
        self.assertNotIn('Area', caption)

    async def test_cap_rejected_before_location_prompt(self):
        self.env['list_locations'].return_value = self.rows * 3
        self.assertEqual(await self.env['add_location_command'](self.update, self.context), -1)
        self.assertIn('6 locations', self.update.message.reply_text.await_args.args[0])


@unittest.skipUnless(os.getenv('TEST_MYSQL_GROUP_LOCATIONS') == '1', 'Opt-in MySQL temporary-table integration')
class GroupDatabaseTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(ROOT / 'src'))
        from telegram_code.database import _get_db_connection
        self.conn = _get_db_connection()
        self.addCleanup(self.conn.close)
        cur = self.conn.cursor()
        for table in ('users', 'user_location', 'rain_notifications'):
            cur.execute(f'SHOW CREATE TABLE {table}')
            ddl = cur.fetchone()[1].replace('CREATE TABLE', 'CREATE TEMPORARY TABLE', 1)
            # Temporary tables cannot carry foreign keys. All DDL and writes below
            # target session-local tables, never the production tables.
            import re
            ddl = re.sub(r',\n  CONSTRAINT [^\n]+', '', ddl)
            cur.execute(ddl)
        cur.close()
        # Production functions may close their connection; keep the one session alive
        # until tearDown so all queries see only our empty temporary tables.
        proxy = SimpleNamespace(cursor=self.conn.cursor, commit=self.conn.commit,
                                rollback=self.conn.rollback, close=lambda: None,
                                is_connected=self.conn.is_connected)
        self.env = dict(_get_db_connection=lambda: proxy, secrets=secrets, MAX_CHAT_LOCATIONS=6, json=json)
        functions('src/telegram_code/rain_state_db.py', self.env)
        self.env['ensure_rain_state_schema']()
        self.env['ensure_rain_state_schema']()  # Re-running migration must be safe.
        functions('src/telegram_code/group_locations.py', self.env)
        functions('src/telegram_code/notification_delivery.py', self.env,
                  {'claim_notifications', 'claim_notification', 'finish_notification', 'release_episode', 'recover_abandoned_notifications'})
        from telegram_code import rain_state
        self.env.update({name: getattr(rain_state, name) for name in ('START', 'ENDED', 'OBSERVED', 'CANCELLED', 'ENDING')})
        self.env['timedelta'] = timedelta
        cur = self.conn.cursor()
        for chat in (-1001, -1002, 42):
            cur.execute("INSERT INTO users (userid,mode) VALUES (%s,'automatic')", (chat,))
            cur.execute('INSERT INTO user_location (userid,latitude,longitude) VALUES (%s,1.3,103.8)', (chat,))
        self.conn.commit()
        cur.close()

    def tearDown(self):
        self.conn.close()  # MySQL automatically drops only this session's temporary tables.

    def queue_notice(self, chat, reason, observed, location_id=0, available_at=None):
        row = next(r for r in self.env['get_rain_locations']()
                   if r['userid'] == chat and r['location_id'] == location_id)
        wet = reason == self.env['OBSERVED']
        predicted = reason == self.env['START']
        result = dict(state='confirmed' if wet else 'predicted' if predicted else 'predicted norain',
                      rain_observed_at=observed, rain_forecast_at=observed+timedelta(minutes=5),
                      radar_raining=wet, rain_forecast_value=.2 if predicted else None,
                      reason=reason)
        self.assertTrue(self.env['save_rain_state'](row, result, row['label'], available_at or observed))
        cur = self.conn.cursor()
        cur.execute('SELECT id FROM rain_notifications WHERE userid=%s AND location_id=%s ORDER BY id DESC LIMIT 1',
                    (chat, location_id))
        notification_id = cur.fetchone()[0]
        cur.close()
        return notification_id

    def notification_statuses(self):
        cur = self.conn.cursor()
        cur.execute('SELECT id,status,telegram_message_id FROM rain_notifications ORDER BY id')
        rows = {row[0]: row[1:] for row in cur.fetchall()}
        cur.close()
        return rows

    def test_six_location_cap_includes_main_and_recovers_after_removal(self):
        for index in range(5):
            self.assertTrue(self.env['add_named_location'](-1001, f'Place {index}', 1.35, 103.9))
        self.assertFalse(self.env['add_named_location'](-1001, 'Seventh', 1.35, 103.9))
        self.assertEqual(len(self.env['list_locations'](-1001)), 6)
        self.env['remove_named_location'](-1001, 'Place 0')
        self.assertTrue(self.env['add_named_location'](-1001, 'Replacement', 1.35, 103.9))

    def test_private_cap_and_group_settings_remain_separate(self):
        for index in range(5):
            self.assertTrue(self.env['add_named_location'](42, f'Personal {index}', 1.35, 103.9))
        self.assertFalse(self.env['add_named_location'](42, 'Seventh', 1.35, 103.9))
        self.assertEqual(len(self.env['list_locations'](42)), 6)
        self.assertEqual(len(self.env['list_locations'](-1001)), 1)

    def test_add_remove_and_chat_isolation(self):
        add = self.env['add_named_location']
        self.assertTrue(add(-1001, 'Office', 1.35, 103.9))
        self.assertFalse(add(-1001, 'office', 1.36, 103.9))
        self.assertTrue(add(-1002, 'Office', 1.36, 103.9))
        self.assertTrue(add(42, 'Office', 1.35, 103.9))
        self.assertEqual(len(self.env['list_locations'](42)), 2)
        self.assertTrue(self.env['remove_named_location'](42, 'Office'))
        self.assertEqual(len(self.env['list_locations'](-1001)), 2)
        self.assertTrue(self.env['remove_named_location'](-1001, 'Office'))
        self.assertEqual(len(self.env['list_locations'](-1001)), 1)
        self.assertEqual(len(self.env['list_locations'](-1002)), 2)
        self.assertFalse(self.env['remove_named_location'](-1001, 'Main'))

    def test_main_change_and_group_mode_leave_other_chats_unchanged(self):
        import mysql.connector
        self.env['Error'] = mysql.connector.Error
        functions('src/telegram_code/database.py', self.env, {'add_location', 'get_location', 'save_mode_choice'})
        self.env['add_named_location'](-1001, 'Office', 1.35, 103.9)
        self.assertTrue(self.env['add_location'](-1001, 1.31, 103.81))
        rows = self.env['list_locations'](-1001)
        self.assertEqual(float(rows[1]['latitude']), 1.35)
        self.assertEqual(float(self.env['get_location'](-1001)[0]), 1.31)
        self.assertTrue(self.env['save_mode_choice'](-1001, 'manual'))
        rows = self.env['get_rain_locations']()
        self.assertTrue(all(row['mode'] == 'manual' for row in rows if row['userid'] == -1001))
        self.assertTrue(all(row['mode'] == 'automatic' for row in rows if row['userid'] != -1001))

    def test_simultaneous_alerts_independent_and_removal_cancels(self):
        self.env['add_named_location'](-1001, 'Office', 1.35, 103.9)
        now = datetime(2026, 9, 17, 10)
        result = dict(state='predicted', rain_observed_at=now, rain_forecast_at=now+timedelta(minutes=5),
                      radar_raining=False, rain_forecast_value=.8, reason='Rain expected')
        rows = [r for r in self.env['get_rain_locations']() if r['userid'] == -1001]
        for row in rows:
            self.assertTrue(self.env['save_rain_state'](row, result, row['label'], now))
        first = self.env['claim_notification'](now)
        self.env['finish_notification'](first, 'sent', now, 1)
        cur = self.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM user_location WHERE userid=-1001 AND rain_alert_at IS NOT NULL')
        self.assertEqual(cur.fetchone()[0], 1)
        cur.close()
        self.env['remove_named_location'](-1001, 'Office')
        self.assertIsNone(self.env['claim_notification'](now))
        self.assertTrue(self.env['add_named_location'](-1001, 'Office', 1.35, 103.9))
        self.assertIsNone(self.env['claim_notification'](now))

    def test_grouped_claim_acknowledges_each_location_with_one_message_id(self):
        self.env['add_named_location'](-1001, 'Office', 1.35, 103.9)
        now = datetime(2026, 9, 28, 12)
        ids = []
        for location in self.env['list_locations'](-1001):
            ids.append(self.queue_notice(-1001, self.env['OBSERVED'], now, location['location_id']))
        other = self.queue_notice(-1002, self.env['OBSERVED'], now)
        items = self.env['claim_notifications'](now)
        self.assertEqual([item['id'] for item in items], ids)
        self.assertTrue(all(item['first_today'] for item in items))
        for item in items:
            self.env['finish_notification'](item, 'sent', now, 777)
        statuses = self.notification_statuses()
        self.assertTrue(all(statuses[event_id] == ('sent', 777) for event_id in ids))
        self.assertEqual(statuses[other], ('pending', None))
        locations = [r for r in self.env['get_rain_locations']() if r['userid'] == -1001]
        self.assertTrue(all(r['rain_alert_at'] == now and r['rain_alert_reason'] == self.env['OBSERVED'] for r in locations))

    def test_removing_one_location_excludes_only_its_pending_notice(self):
        self.env['add_named_location'](-1001, 'Office', 1.35, 103.9)
        office = next(r for r in self.env['list_locations'](-1001) if r['label'] == 'Office')
        now = datetime(2026, 9, 28, 12)
        main_id = self.queue_notice(-1001, self.env['OBSERVED'], now)
        office_id = self.queue_notice(-1001, self.env['OBSERVED'], now, office['location_id'])
        scope = sorted([r['location_id'], r['rain_settings_version']]
                       for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        cur = self.conn.cursor()
        cur.execute('UPDATE rain_notifications SET message=%s,photo=%s WHERE id=%s',
                    (json.dumps({'rain_alert': 1, 'photo_locations': scope}), b'combined-map', main_id))
        self.conn.commit()
        cur.close()
        self.assertTrue(self.env['remove_named_location'](-1001, 'Office'))
        claimed = self.env['claim_notifications'](now)
        self.assertEqual([item['id'] for item in claimed], [main_id])
        self.assertIsNone(claimed[0]['photo'])  # Do not expose the removed location's saved marker.
        self.assertEqual(self.notification_statuses()[office_id], ('cancelled', None))

    def test_unchanged_saved_locations_keep_the_shared_map(self):
        now = datetime(2026, 9, 28, 12)
        event_id = self.queue_notice(-1001, self.env['OBSERVED'], now)
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        cur = self.conn.cursor()
        cur.execute('UPDATE rain_notifications SET message=%s,photo=%s WHERE id=%s',
                    (json.dumps({'rain_alert': 1, 'photo_locations': [[0, row['rain_settings_version']]]}),
                     b'combined-map', event_id))
        self.conn.commit()
        cur.close()
        claimed = self.env['claim_notifications'](now)
        self.assertEqual(claimed[0]['photo'], b'combined-map')

    def test_expired_clear_keeps_forecast_rearm_until_fifteen_minutes(self):
        from telegram_code.rain_state import next_rain_state
        now = datetime(2026, 9, 28, 12)
        for chat, reason in [(-1001, self.env['ENDED']), (-1002, self.env['CANCELLED'])]:
            with self.subTest(reason=reason):
                event_id = self.queue_notice(chat, reason, now)
                cur = self.conn.cursor()
                cur.execute('UPDATE user_location SET rain_dry_since=%s WHERE userid=%s AND location_id=0',
                            (now-timedelta(minutes=15), chat))
                self.conn.commit()
                cur.close()
                self.assertEqual(self.env['claim_notifications'](now+timedelta(minutes=10)), [{'cancelled': True}])
                self.assertEqual(self.notification_statuses()[event_id], ('cancelled', None))
                row = next(r for r in self.env['get_rain_locations']() if r['userid'] == chat)
                self.assertEqual(row['rain_episode_reason'], reason)
                self.assertEqual(row['rain_dry_since'], now-timedelta(minutes=15))
                early = next_rain_state(row, .2, 0, now+timedelta(minutes=10), now+timedelta(minutes=15))
                self.assertIsNone(early['reason'])
                allowed = next_rain_state(row, .2, 0, now+timedelta(minutes=15), now+timedelta(minutes=20))
                self.assertEqual(allowed['reason'], self.env['START'])

    def test_new_rain_supersedes_pending_clear_without_sending_both(self):
        now = datetime(2026, 9, 28, 12)
        clear = self.queue_notice(-1001, self.env['ENDED'], now)
        start = self.queue_notice(-1001, self.env['OBSERVED'], now+timedelta(minutes=5))
        claimed = self.env['claim_notifications'](now+timedelta(minutes=5))
        self.assertEqual([item['id'] for item in claimed], [start])
        self.assertEqual(self.notification_statuses()[clear], ('cancelled', None))
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        self.assertEqual(row['rain_episode_reason'], self.env['OBSERVED'])

    def test_first_today_is_per_chat_and_resets_next_day(self):
        now = datetime(2026, 9, 28, 12)
        self.queue_notice(-1001, self.env['OBSERVED'], now)
        first = self.env['claim_notifications'](now)[0]
        self.assertTrue(first['first_today'])
        self.env['finish_notification'](first, 'sent', now, 777)
        self.queue_notice(-1001, self.env['ENDED'], now+timedelta(minutes=5))
        second = self.env['claim_notifications'](now+timedelta(minutes=5))[0]
        self.assertFalse(second['first_today'])
        self.env['finish_notification'](second, 'sent', now+timedelta(minutes=5), 778)
        self.queue_notice(-1002, self.env['OBSERVED'], now+timedelta(minutes=5))
        other = self.env['claim_notifications'](now+timedelta(minutes=5))[0]
        self.assertTrue(other['first_today'])
        self.env['finish_notification'](other, 'sent', now+timedelta(minutes=5), 779)
        tomorrow = now+timedelta(days=1)
        self.queue_notice(-1001, self.env['OBSERVED'], tomorrow)
        self.assertTrue(self.env['claim_notifications'](tomorrow)[0]['first_today'])

    def test_blocked_and_uncertain_receipts_do_not_repeat_first_today(self):
        now = datetime(2026, 9, 28, 12)
        for chat, outcome in [(-1001, 'sending'), (-1002, 'uncertain')]:
            with self.subTest(outcome=outcome):
                self.queue_notice(chat, self.env['OBSERVED'], now)
                first = self.env['claim_notifications'](now)[0]
                if outcome == 'uncertain':
                    self.env['finish_notification'](first, outcome, now)
                # Protected receipts model Telegram success with a blocked DB ack.
                at = now+timedelta(minutes=16) if outcome == 'sending' else now+timedelta(minutes=5)
                if outcome == 'sending':
                    self.env['recover_abandoned_notifications'](at, (first['id'],))
                self.queue_notice(chat, self.env['ENDED'], at)
                next_batch = self.env['claim_notifications'](at)
                self.assertFalse(next_batch[0]['first_today'])

    def test_abandoned_uncertain_attempt_does_not_repeat_daily_outlook(self):
        now = datetime(2026, 9, 28, 12)
        self.queue_notice(-1001, self.env['OBSERVED'], now)
        first = self.env['claim_notifications'](now)[0]
        self.assertTrue(first['first_today'])
        self.env['finish_notification'](first, 'uncertain', now)
        later = now+timedelta(minutes=16)
        self.env['recover_abandoned_notifications'](later)
        self.assertEqual(self.notification_statuses()[first['id']], ('abandoned', None))
        self.queue_notice(-1001, self.env['OBSERVED'], later)
        claimed = self.env['claim_notifications'](later)
        self.assertFalse(claimed[0]['first_today'])

    def test_rate_limited_group_retry_keeps_other_location_and_latest_change(self):
        self.env['add_named_location'](-1001, 'Office', 1.35, 103.9)
        now = datetime(2026, 9, 28, 12)
        original_ids = []
        for location in self.env['list_locations'](-1001):
            original_ids.append(self.queue_notice(-1001, self.env['OBSERVED'], now, location['location_id']))
        original = self.env['claim_notifications'](now)
        for item in original:
            self.env['finish_notification'](item, 'pending', now, retry_seconds=30)
        latest = self.queue_notice(-1001, self.env['ENDED'], now+timedelta(minutes=5))
        claimed = self.env['claim_notifications'](now+timedelta(minutes=5))
        self.assertEqual({item['id'] for item in claimed}, {original_ids[1], latest})
        self.assertEqual(self.notification_statuses()[original_ids[0]], ('cancelled', None))
        self.assertTrue(all(item['first_today'] for item in claimed))
        for item in claimed:
            self.env['finish_notification'](item, 'sent', now+timedelta(minutes=5), 888)
        self.assertTrue(all(self.notification_statuses()[item['id']] == ('sent', 888) for item in claimed))



    def test_observation_upgrade_and_failed_start_releases_episode(self):
        now = datetime(2026, 9, 22, 15)
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        result = dict(state='predicted norain', rain_observed_at=now,
                      rain_forecast_at=now+timedelta(minutes=5), radar_raining=False,
                      rain_forecast_value=None, reason=None)
        self.assertTrue(self.env['save_rain_state'](row, result))
        result.update(rain_forecast_value=.2, state='predicted', reason='start')
        self.assertTrue(self.env['save_rain_state'](row, result, 'rain expected', now))
        self.assertFalse(self.env['save_rain_state'](row, result, 'duplicate', now))
        item = self.env['claim_notification'](now)
        self.env['finish_notification'](item, 'failed', now)
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        self.assertIsNone(row['rain_episode_reason'])

    def test_abandoned_claim_does_not_resend_original(self):
        now = datetime(2026, 9, 22, 15)
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        result = dict(state='predicted', rain_observed_at=now,
                      rain_forecast_at=now+timedelta(minutes=5), radar_raining=False,
                      rain_forecast_value=.2, reason='start')
        self.env['save_rain_state'](row, result, 'rain expected', now)
        item = self.env['claim_notification'](now)
        self.env['recover_abandoned_notifications'](now+timedelta(minutes=16))
        self.assertIsNone(self.env['claim_notification'](now+timedelta(minutes=16)))
        cur = self.conn.cursor()
        cur.execute('SELECT status FROM rain_notifications WHERE id=%s', (item['id'],))
        self.assertEqual(cur.fetchone()[0], 'abandoned')
        cur.close()
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        self.assertIsNone(row['rain_episode_reason'])


    def test_recovery_preserves_newer_episode_and_known_receipt(self):
        now = datetime(2026, 9, 22, 15)
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        result = dict(state='predicted', rain_observed_at=now,
                      rain_forecast_at=now+timedelta(minutes=5), radar_raining=False,
                      rain_forecast_value=.2, reason='start')
        self.env['save_rain_state'](row, result, 'first', now)
        first = self.env['claim_notification'](now)
        self.env['recover_abandoned_notifications'](now+timedelta(minutes=16), (first['id'],))
        cur = self.conn.cursor()
        cur.execute('SELECT status FROM rain_notifications WHERE id=%s', (first['id'],))
        self.assertEqual(cur.fetchone()[0], 'sending')
        result['rain_observed_at'] = now+timedelta(minutes=5)
        result['rain_forecast_at'] = now+timedelta(minutes=10)
        self.env['save_rain_state'](row, result, 'newer', now+timedelta(minutes=5))
        self.env['recover_abandoned_notifications'](now+timedelta(minutes=16))
        row = next(r for r in self.env['get_rain_locations']() if r['userid'] == -1001)
        self.assertEqual(row['rain_episode_reason'], 'start')
        cur.close()


    def test_dry_confirmation_survives_database_reload(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('dry_state', ROOT/'src/telegram_code/rain_state.py')
        state = importlib.util.module_from_spec(spec);spec.loader.exec_module(state)
        now = datetime(2026,9,28,1,50)
        for offset, wet in [(0,True),(5,False),(10,False),(15,False),(20,False)]:
            row = next(r for r in self.env['get_rain_locations']() if r['userid']==-1001)
            at = now+timedelta(minutes=offset)
            result = state.next_rain_state(row,None,.2 if wet else 0,at,at+timedelta(minutes=10))
            expected = state.OBSERVED if offset==0 else state.ENDED if offset==20 else None
            self.assertEqual(result['reason'],expected)
            self.env['save_rain_state'](row,result,'alert' if result['reason'] else None,at)
            saved = next(r for r in self.env['get_rain_locations']() if r['userid']==-1001)
            self.assertEqual(saved['rain_dry_since'],result['rain_dry_since'])
            self.assertEqual(bool(saved['rain_episode_wet']),result['rain_episode_wet'])
        cur=self.conn.cursor()
        cur.execute('SELECT reason FROM rain_notifications WHERE userid=-1001 ORDER BY id')
        self.assertEqual([r[0] for r in cur.fetchall()],[state.OBSERVED,state.ENDED])
        cur.close()
