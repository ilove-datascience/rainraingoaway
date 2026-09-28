"""Official v2 API shape; weather values below are synthetic test fixtures."""
import asyncio
import copy
from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code import daily_forecast as module


NOW = datetime(2026, 9, 28, 17, tzinfo=module.SGT)


def payload():
    # Shape verified against api-open.data.gov.sg/v2/real-time/api/
    # twenty-four-hr-forecast on 28 September 2026.
    return {'code': 0, 'data': {'records': [{
        'date': '2026-09-28',
        'timestamp': '2026-09-28T16:43:00+08:00',
        'updatedTimestamp': '2026-09-28T16:51:01+08:00',
        'general': {
            'validPeriod': {'start': '2026-09-28T12:00:00+08:00',
                            'end': '2026-09-29T12:00:00+08:00'},
            'temperature': {'low': 24, 'high': 33, 'unit': 'Degrees Celsius'},
            'forecast': {'code': 'CL', 'text': 'Cloudy'},
        },
        'periods': [],
    }]}}


class ForecastParsingTests(unittest.TestCase):
    def test_official_shape_keeps_true_overnight_window(self):
        forecast = module.parse_daily_forecast(payload(), NOW)
        text = module.format_daily_forecast(forecast)
        self.assertIn('Singapore daily outlook · NEA/MSS', text)
        self.assertIn('Cloudy · 24–33°C', text)
        self.assertIn('28 Sep 12:00–29 Sep 12:00 SGT', text)
        self.assertNotIn('your location', text)

    def test_aware_and_naive_singapore_clocks_agree(self):
        self.assertEqual(module.parse_daily_forecast(payload(), NOW),
                         module.parse_daily_forecast(payload(), NOW.replace(tzinfo=None)))
        utc = NOW.astimezone(module.timezone.utc)
        self.assertEqual(module.parse_daily_forecast(payload(), NOW),
                         module.parse_daily_forecast(payload(), utc))

    def test_expired_future_and_old_issued_data_are_omitted(self):
        for now in (datetime(2026, 9, 29, 12, tzinfo=module.SGT),
                    datetime(2026, 9, 28, 11, tzinfo=module.SGT),
                    datetime(2026, 9, 29, 5, tzinfo=module.SGT)):
            with self.subTest(now=now):
                self.assertIsNone(module.parse_daily_forecast(payload(), now))
        self.assertIsNone(module.parse_daily_forecast(payload(), NOW - timedelta(hours=1)))

    def test_latest_valid_record_selected_without_assuming_order(self):
        data = payload()
        older = copy.deepcopy(data['data']['records'][0])
        older['timestamp'] = '2026-09-28T11:00:00+08:00'
        older['general']['forecast']['text'] = 'Showers'
        malformed = {'general': {}}
        data['data']['records'].extend([older, malformed])
        self.assertEqual(module.parse_daily_forecast(data, NOW)['outlook'], 'Cloudy')

    def test_bad_types_missing_fields_and_non_celsius_fail_closed(self):
        variants = [None, {}, {'code': 500}, {'code': 0, 'data': {'records': {}}}]
        for field, value in [('forecast', {'text': 'x' * 181}),
                             ('forecast', {'text': '  '}),
                             ('temperature', {'low': 30, 'high': 20, 'unit': 'Degrees Celsius'}),
                             ('temperature', {'low': 24, 'high': float('nan'), 'unit': 'Degrees Celsius'}),
                             ('temperature', {'low': 24, 'high': 33, 'unit': 'Fahrenheit'}),
                             ('validPeriod', {'start': '2026-09-28T12:00:00',
                                              'end': '2026-09-29T12:00:00'})]:
            data = payload()
            data['data']['records'][0]['general'][field] = value
            variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                self.assertIsNone(module.parse_daily_forecast(data, NOW))

    def test_fetch_uses_public_url_and_bounded_timeout(self):
        response = Mock()
        response.json.return_value = payload()
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(module.requests, 'get', return_value=context) as get:
            self.assertIsNotNone(module.fetch_daily_forecast(NOW))
        get.assert_called_once_with(module.FORECAST_URL, timeout=(3, 5))
        response.raise_for_status.assert_called_once()


class ForecastCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_chats_share_one_request(self):
        cache = {}
        forecast = module.parse_daily_forecast(payload(), NOW)
        with patch.object(module, 'fetch_daily_forecast', return_value=forecast) as fetch:
            results = await asyncio.gather(*(module.get_daily_forecast(cache, NOW) for _ in range(6)))
        self.assertEqual(results, [forecast] * 6)
        fetch.assert_called_once()

    async def test_refresh_after_fifteen_minutes(self):
        cache = {}
        forecast = module.parse_daily_forecast(payload(), NOW)
        with patch.object(module, 'fetch_daily_forecast', return_value=forecast) as fetch:
            await module.get_daily_forecast(cache, NOW)
            await module.get_daily_forecast(cache, NOW + timedelta(minutes=14))
            await module.get_daily_forecast(cache, NOW + timedelta(minutes=15))
        self.assertEqual(fetch.call_count, 2)

    async def test_network_failure_is_backed_off_and_never_raises(self):
        cache = {}
        with patch.object(module, 'fetch_daily_forecast', side_effect=requests.Timeout) as fetch:
            self.assertIsNone(await module.get_daily_forecast(cache, NOW))
            self.assertIsNone(await module.get_daily_forecast(cache, NOW + timedelta(minutes=4)))
            self.assertIsNone(await module.get_daily_forecast(cache, NOW + timedelta(minutes=5)))
        self.assertEqual(fetch.call_count, 2)

    async def test_failure_preserves_only_still_current_cache(self):
        cache = {}
        forecast = module.parse_daily_forecast(payload(), NOW)
        with patch.object(module, 'fetch_daily_forecast', return_value=forecast):
            await module.get_daily_forecast(cache, NOW)
        with patch.object(module, 'fetch_daily_forecast', side_effect=requests.ConnectionError):
            self.assertEqual(await module.get_daily_forecast(cache, NOW + timedelta(minutes=20)), forecast)
            self.assertIsNone(await module.get_daily_forecast(cache, NOW + timedelta(hours=13)))

    async def test_expired_cached_value_is_not_served_within_cache_ttl(self):
        cache = {}
        data = payload()
        data['data']['records'][0]['general']['validPeriod']['end'] = '2026-09-28T17:03:00+08:00'
        forecast = module.parse_daily_forecast(data, NOW)
        with patch.object(module, 'fetch_daily_forecast', return_value=forecast) as fetch:
            self.assertIsNotNone(await module.get_daily_forecast(cache, NOW))
            self.assertIsNone(await module.get_daily_forecast(cache, NOW + timedelta(minutes=4)))
        fetch.assert_called_once()


class WeatherCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Intentionally no saved location, preferences, user/chat data or database.
        self.context = SimpleNamespace(application=SimpleNamespace(bot_data={}))
        self.update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))

    async def test_fresh_outlook_includes_source_and_true_window_without_preview(self):
        forecast = module.parse_daily_forecast(payload(), NOW)
        with patch.object(module, 'get_daily_forecast', new=AsyncMock(return_value=forecast)) as get:
            await module.weather_command(self.update, self.context)
        get.assert_awaited_once_with(self.context.application.bot_data)
        reply = self.update.effective_message.reply_text
        reply.assert_awaited_once()
        text = reply.await_args.args[0]
        self.assertIn('Cloudy · 24–33°C', text)
        self.assertIn('Valid 28 Sep 12:00–29 Sep 12:00 SGT', text)
        self.assertIn('Issued 28 Sep 16:43 SGT', text)
        self.assertIn(module.SOURCE_URL, text)
        self.assertEqual(reply.await_args.kwargs, {'disable_web_page_preview': True})

    async def test_unavailable_reply_does_not_present_old_weather(self):
        with patch.object(module, 'get_daily_forecast', new=AsyncMock(return_value=None)):
            await module.weather_command(self.update, self.context)
        self.update.effective_message.reply_text.assert_awaited_once_with(
            'Daily outlook unavailable right now. Please try /weather again shortly.',
            disable_web_page_preview=True)

    async def test_nonmessage_update_does_not_fetch_or_send(self):
        self.update.effective_message = None
        with patch.object(module, 'get_daily_forecast', new=AsyncMock()) as get:
            await module.weather_command(self.update, self.context)
        get.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
