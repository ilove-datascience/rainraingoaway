"""Scheduled forecasts: real SQLite, simulated clocks and Telegram transport."""
import asyncio
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests
from telegram.error import Forbidden, RetryAfter
from telegram_code import daily_forecast as source
from telegram_code import weather_schedule as module


NOW = datetime(2026, 9, 30, 5, tzinfo=source.SGT)


def forecast(now=NOW):
    return dict(outlook='Afternoon thundery showers', low=25, high=33,
                issued_at=now-timedelta(hours=1),
                valid_from=now.replace(hour=6), valid_until=now.replace(hour=6)+timedelta(days=1))


def four_day_payload():
    # Shape verified against NEA's official v2 API; weather values are synthetic.
    return {'code': 0, 'data': {'records': [{
        'timestamp': '2026-09-30T17:30:00+08:00', 'forecasts': [
            {'timestamp': '2026-10-01T00:00:00+08:00',
             'forecast': {'text': 'Thundery Showers', 'summary': 'Afternoon thundery showers\n'},
             'temperature': {'low': 25, 'high': 33, 'unit': 'Degrees Celsius'}}]}]}}


def daily_payload(start, end, issued):
    return {'code': 0, 'data': {'records': [{
        'timestamp': issued.isoformat(), 'general': {
            'validPeriod': {'start': start.isoformat(), 'end': end.isoformat()},
            'forecast': {'text': 'Thundery Showers'},
            'temperature': {'low': 25, 'high': 33, 'unit': 'Degrees Celsius'}}}]}}


def response(payload):
    result = Mock()
    result.json.return_value = payload
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class ScheduledSourceTests(unittest.TestCase):
    def test_night_selects_tomorrows_exact_date_across_month_boundary(self):
        now = NOW.replace(hour=22)
        result = source.parse_dated_outlook(four_day_payload(), now.date()+timedelta(days=1), now)
        self.assertEqual(result['outlook'], 'Afternoon thundery showers')
        self.assertEqual(result['valid_from'].isoformat(), '2026-10-01T00:00:00+08:00')
        self.assertIsNone(source.parse_dated_outlook(four_day_payload(), now.date(), now))
        text = module.format_bulletin(result, 'night', now)
        self.assertIn("Tomorrow's weather · Thu 01 Oct", text)
        self.assertIn('25–33°C', text)

    def test_malformed_stale_and_future_dated_issues_are_rejected(self):
        now = NOW.replace(hour=22)
        variants = [None, {}, {'code': 500}, {'code': 0, 'data': {'records': {}}}]
        for value in ['2026-09-27T17:00:00+08:00', '2026-10-01T01:00:00+08:00', 'not a date']:
            data = four_day_payload()
            data['data']['records'][0]['timestamp'] = value
            variants.append(data)
        for data in variants:
            self.assertIsNone(source.parse_dated_outlook(data, now.date()+timedelta(days=1), now))
        for change in [{'low': float('nan'), 'high': 33, 'unit': 'Degrees Celsius'},
                       {'low': 25, 'high': 33, 'unit': 'Fahrenheit'}]:
            data = four_day_payload()
            data['data']['records'][0]['forecasts'][0]['temperature'] = change
            self.assertIsNone(source.parse_dated_outlook(data, now.date()+timedelta(days=1), now))

    def test_dated_outlook_selects_newest_issue_not_response_order(self):
        data = four_day_payload()
        older = copy.deepcopy(data['data']['records'][0])
        older['timestamp'] = '2026-09-30T05:00:00+08:00'
        older['forecasts'][0]['forecast']['summary'] = 'Older forecast'
        data['data']['records'].extend([older, None])
        result = source.parse_dated_outlook(data, datetime(2026,10,1).date(), NOW.replace(hour=22))
        self.assertEqual(result['outlook'], 'Afternoon thundery showers')

    def test_morning_accepts_upcoming_daylight_window_but_not_last_hours_of_yesterday(self):
        fresh = daily_payload(NOW.replace(hour=6), NOW.replace(hour=6)+timedelta(days=1), NOW-timedelta(minutes=5))
        self.assertIsNotNone(source.parse_daily_forecast(fresh, NOW, scheduled=True))
        self.assertIsNone(source.parse_daily_forecast(fresh, NOW))  # /weather still means valid now.
        old = daily_payload(NOW.replace(hour=6)-timedelta(days=1), NOW.replace(hour=6), NOW-timedelta(hours=6))
        self.assertIsNone(source.parse_daily_forecast(old, NOW, scheduled=True))

    def test_morning_fallback_uses_yesterdays_dated_outlook_for_today(self):
        dated = four_day_payload()
        dated['data']['records'][0]['timestamp'] = '2026-09-29T17:30:00+08:00'
        dated['data']['records'][0]['forecasts'][0]['timestamp'] = '2026-09-30T00:00:00+08:00'
        with patch.object(source.requests, 'get', side_effect=[response({'code': 0, 'data': {'records': []}}), response(dated)]) as get:
            result = source.fetch_scheduled_forecast('morning', NOW)
        self.assertEqual(result['valid_from'].date(), NOW.date())
        self.assertEqual(get.call_args.kwargs['params'], {'date': '2026-09-29'})
        self.assertEqual(get.call_args.kwargs['timeout'], (3, 5))

    def test_night_uses_four_day_source_and_noon_uses_24_hour_source(self):
        with patch.object(source.requests, 'get', return_value=response(four_day_payload())) as get:
            self.assertIsNotNone(source.fetch_scheduled_forecast('night', NOW.replace(hour=22)))
        get.assert_called_once_with(source.OUTLOOK_URL, params={}, timeout=(3, 5))
        noon = NOW.replace(hour=12)
        payload = daily_payload(noon, noon+timedelta(days=1), noon-timedelta(minutes=5))
        with patch.object(source.requests, 'get', return_value=response(payload)) as get:
            self.assertIsNotNone(source.fetch_scheduled_forecast('noon', noon))
        get.assert_called_once_with(source.FORECAST_URL, timeout=(3, 5))


class ScheduleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = patch.object(module, 'DB_PATH', Path(tmp.name)/'weather.sqlite3')
        db.start()
        self.addCleanup(db.stop)
        clock = patch.object(module, '_sg_time', side_effect=lambda value=None: source._sg_time(value or self.now))
        clock.start()
        self.addCleanup(clock.stop)
        self.now = NOW
        self.context = SimpleNamespace(application=SimpleNamespace(bot_data={}), bot=SimpleNamespace(
            send_message=AsyncMock(), get_chat_member=AsyncMock(return_value=SimpleNamespace(status='administrator'))))
        self.update = SimpleNamespace(effective_chat=SimpleNamespace(id=42, type='private'),
            effective_user=SimpleNamespace(id=42), effective_message=SimpleNamespace(reply_text=AsyncMock()),
            callback_query=SimpleNamespace(data='weatheralerts:on', answer=AsyncMock(), edit_message_text=AsyncMock()))
        fetch = patch.object(module, 'fetch_scheduled_forecast', return_value=forecast())
        self.fetch = fetch.start()
        self.addCleanup(fetch.stop)

    def enable(self, chat=42):
        module.set_enabled(chat, True, NOW-timedelta(days=1))

    async def test_default_off_does_not_fetch_or_send(self):
        self.assertFalse(module.get_enabled(42))
        await module.send_scheduled_weather(self.context)
        self.fetch.assert_not_called()
        self.context.bot.send_message.assert_not_awaited()

    async def test_single_switch_enables_all_three_times_without_radar(self):
        self.enable()
        await module.send_scheduled_weather(self.context)
        self.now = NOW.replace(hour=12)
        await module.send_scheduled_weather(self.context)
        self.now = NOW.replace(hour=22)
        await module.send_scheduled_weather(self.context)
        texts = [call.kwargs['text'] for call in self.context.bot.send_message.await_args_list]
        self.assertEqual(len(texts), 3)
        self.assertTrue(texts[0].startswith("Today's weather"))
        self.assertTrue(texts[1].startswith('Midday weather update'))
        self.assertTrue(texts[2].startswith("Tomorrow's weather"))

    async def test_single_switch_disables_all_three_times(self):
        self.enable()
        module.set_enabled(42, False, NOW)
        for hour in (5, 12, 22):
            self.now = NOW.replace(hour=hour)
            await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_not_awaited()

    async def test_restart_and_parallel_jobs_cannot_repeat_a_sent_bulletin(self):
        self.enable()
        await asyncio.gather(module.send_scheduled_weather(self.context), module.send_scheduled_weather(self.context))
        self.context.application.bot_data = {}  # Process memory lost, real SQLite retained.
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()

    async def test_new_day_gets_new_delivery_and_other_chats_are_independent(self):
        self.enable()
        self.enable(chat=-100123)
        await module.send_scheduled_weather(self.context)
        self.assertEqual(self.fetch.call_count, 1)
        self.assertEqual({call.kwargs['chat_id'] for call in self.context.bot.send_message.await_args_list}, {42,-100123})
        self.now += timedelta(days=1)
        await module.send_scheduled_weather(self.context)
        self.assertEqual(self.context.bot.send_message.await_count, 4)

    async def test_short_restart_catchup_but_no_stale_backlog(self):
        self.enable()
        self.now = NOW+timedelta(minutes=29)
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()
        self.now = NOW+timedelta(days=1, minutes=30)
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()

    def test_utc_host_clock_and_singapore_clock_have_same_due_slot(self):
        self.enable()
        self.assertEqual(module.due_deliveries(NOW), module.due_deliveries(NOW.astimezone(timezone.utc)))
        self.assertEqual(module.due_deliveries(NOW), module.due_deliveries(NOW.replace(tzinfo=None)))

    async def test_late_opt_in_waits_until_next_slot(self):
        module.set_enabled(42, True, NOW+timedelta(seconds=1))
        self.now += timedelta(minutes=1)
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_not_awaited()
        self.now = NOW.replace(hour=22)
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()

    async def test_turning_off_after_due_selection_prevents_delivery(self):
        self.enable()
        def disable(*args):
            module.set_enabled(42, False, NOW)
            return forecast()
        self.fetch.side_effect = disable
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_not_awaited()

    async def test_network_timeout_is_backed_off_then_can_recover(self):
        self.enable()
        self.fetch.side_effect = [requests.Timeout(), forecast()]
        await module.send_scheduled_weather(self.context)
        self.now += timedelta(minutes=1)
        await module.send_scheduled_weather(self.context)
        self.assertEqual(self.fetch.call_count, 1)
        self.context.bot.send_message.assert_not_awaited()
        self.now += timedelta(minutes=4)
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()

    async def test_api_delay_cannot_send_after_window_closes(self):
        self.enable()
        def delayed(*args):
            self.now += timedelta(minutes=30)
            return forecast()
        self.fetch.side_effect = delayed
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_not_awaited()

    async def test_ambiguous_telegram_timeout_or_crash_never_resends(self):
        self.enable()
        self.context.bot.send_message.side_effect = TimeoutError()
        await module.send_scheduled_weather(self.context)
        self.context.application.bot_data = {}
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()
        self.now += timedelta(days=1)
        key = module.due_deliveries(self.now)[0]
        self.assertTrue(module.claim_delivery(key, self.now))  # Simulate death between claim and receipt.
        await module.send_scheduled_weather(self.context)
        self.context.bot.send_message.assert_awaited_once()

    async def test_telegram_rate_limit_retries_once_after_delay(self):
        self.enable()
        self.context.bot.send_message.side_effect = [RetryAfter(120), None]
        await module.send_scheduled_weather(self.context)
        self.now += timedelta(minutes=1)
        await module.send_scheduled_weather(self.context)
        self.assertEqual(self.context.bot.send_message.await_count, 1)
        self.now += timedelta(minutes=1)
        await module.send_scheduled_weather(self.context)
        self.context.application.bot_data = {}
        await module.send_scheduled_weather(self.context)
        self.assertEqual(self.context.bot.send_message.await_count, 2)

    async def test_blocked_bot_disables_updates_for_that_chat_only(self):
        self.enable()
        self.enable(chat=43)
        self.context.bot.send_message.side_effect = [Forbidden('blocked'), None]
        await module.send_scheduled_weather(self.context)
        self.assertFalse(module.get_enabled(42))
        self.assertTrue(module.get_enabled(43))

    async def test_enable_button_saves_defaults_and_no_modal_confirmation(self):
        await module.weather_alerts_callback(self.update, self.context)
        self.assertTrue(module.get_enabled(42))
        self.update.callback_query.answer.assert_awaited_once_with('Weather settings saved.', show_alert=False)
        self.assertIn('Daily weather updates: On', self.update.callback_query.edit_message_text.await_args.args[0])

    async def test_repeated_explicit_enable_taps_do_not_postpone_next_delivery(self):
        self.enable()
        await module.weather_alerts_callback(self.update, self.context)
        self.assertEqual(len(module.due_deliveries(NOW)), 1)

    async def test_group_members_cannot_change_shared_preferences_but_admins_can(self):
        self.update.effective_chat = SimpleNamespace(id=-100123, type='supergroup')
        self.context.bot.get_chat_member.return_value.status = 'member'
        await module.weather_alerts_callback(self.update, self.context)
        self.assertFalse(module.get_enabled(-100123))
        self.context.bot.get_chat_member.return_value.status = 'administrator'
        await module.weather_alerts_callback(self.update, self.context)
        self.assertTrue(module.get_enabled(-100123))
        self.assertFalse(module.get_enabled(42))

    async def test_switch_persists_after_process_memory_is_lost(self):
        self.enable()
        self.context.application.bot_data = {}
        self.assertTrue(module.get_enabled(42))
        self.update.callback_query.data = 'weatheralerts:off'
        await module.weather_alerts_callback(self.update, self.context)
        self.assertFalse(module.get_enabled(42))

    async def test_storage_failure_does_not_claim_settings_were_saved(self):
        with patch.object(module, 'set_enabled', side_effect=OSError()):
            await module.weather_alerts_callback(self.update, self.context)
        self.assertIn("Couldn't save", self.update.callback_query.answer.await_args.args[0])
        self.update.callback_query.edit_message_text.assert_not_awaited()

    async def test_settings_menu_shows_time_and_stays_off_until_action(self):
        await module.weather_alerts_command(self.update, self.context)
        text = self.update.effective_message.reply_text.await_args.args[0]
        self.assertIn('SGT', text)
        self.assertIn('Daily weather updates: Off', text)
        self.assertFalse(module.get_enabled(42))
        markup = self.update.effective_message.reply_text.await_args.kwargs['reply_markup']
        self.assertEqual(len(markup.inline_keyboard), 1)
        self.assertEqual(len(markup.inline_keyboard[0]), 1)

    def test_job_registration_is_independent_and_single_instance(self):
        queue = Mock()
        module.schedule_weather(queue)
        kwargs = queue.run_repeating.call_args.kwargs
        self.assertEqual(kwargs['interval'], 60)
        self.assertEqual(kwargs['job_kwargs']['max_instances'], 1)


class BotWiringTests(unittest.TestCase):
    def test_real_startup_registers_menu_command_callback_and_job(self):
        # Execute startup wiring without loading a model, bot token or network.
        import ast
        from functools import partial
        import logging
        from telegram import Chat, Message, Update
        from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, filters
        path = Path(__file__).resolve().parents[1]/'src/telegram_code/telegram_bot.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_bot')
        app = Mock()
        application = Mock()
        application.builder.return_value.bot.return_value.build.return_value = app
        env = dict(logging=logging, partial=partial, Application=application,
                   CommandHandler=CommandHandler, CallbackQueryHandler=CallbackQueryHandler,
                   MessageHandler=MessageHandler, filters=filters,
                   load_token=lambda: 'test', CuteBot=Mock(), HTTPXRequest=Mock(),
                   schedule_weather=module.schedule_weather,
                   weather_alerts_command=module.weather_alerts_command,
                   weather_alerts_callback=module.weather_alerts_callback)
        for name in ('cutemode', 'weather_command', 'handle_feedback', 'get_conversation_handler',
                     'locations_command', 'remove_location_command', 'schedule_heartbeat',
                     'check_radar_notifications', 'check_model_queue', 'start', 'handle_actual',
                     'handle_msg', 'handle_location', 'on_bot_error'):
            env[name] = Mock()
        exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), 'exec'), env)
        env['run_bot'](None, '.', None)
        handlers = [call.args[0] for call in app.add_handler.call_args_list]
        self.assertTrue(any(isinstance(h, CommandHandler) and 'weatheralerts' in h.commands
                            and h.callback is module.weather_alerts_command for h in handlers))
        self.assertTrue(any(isinstance(h, CallbackQueryHandler) and h.pattern.match('weatheralerts:on')
                            and h.callback is module.weather_alerts_callback for h in handlers))
        update = Update(1, message=Message(1, NOW, Chat(42, 'private'), text='Weather updates'))
        first = next(h for h in handlers if isinstance(h, MessageHandler) and h.check_update(update))
        self.assertIs(first.callback, module.weather_alerts_command)
        scheduled = [call for call in app.job_queue.run_repeating.call_args_list
                     if call.args and call.args[0] is module.send_scheduled_weather]
        self.assertEqual(len(scheduled), 1)


if __name__ == '__main__':
    unittest.main()
