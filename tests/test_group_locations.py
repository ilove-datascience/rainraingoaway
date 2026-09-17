"""Group-location handler checks plus opt-in MySQL tests using temporary tables only."""
import ast
import asyncio
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
        self.env = dict(_get_db_connection=lambda: proxy, secrets=secrets, MAX_CHAT_LOCATIONS=6)
        functions('src/telegram_code/rain_state_db.py', self.env)
        self.env['ensure_rain_state_schema']()
        self.env['ensure_rain_state_schema']()  # Re-running migration must be safe.
        functions('src/telegram_code/group_locations.py', self.env)
        functions('src/telegram_code/notification_delivery.py', self.env,
                  {'claim_notification', 'finish_notification'})
        self.env['ENDED'] = 'ended'
        self.env['timedelta'] = timedelta
        cur = self.conn.cursor()
        for chat in (-1001, -1002, 42):
            cur.execute("INSERT INTO users (userid,mode) VALUES (%s,'automatic')", (chat,))
            cur.execute('INSERT INTO user_location (userid,latitude,longitude) VALUES (%s,1.3,103.8)', (chat,))
        self.conn.commit()
        cur.close()

    def tearDown(self):
        self.conn.close()  # MySQL automatically drops only this session's temporary tables.

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


if __name__ == '__main__':
    unittest.main()
