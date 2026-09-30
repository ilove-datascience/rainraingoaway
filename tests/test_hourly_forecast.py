"""Real timestamp semantics and isolated hourly-provider/Telegram integration."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests
from telegram_code import hourly_forecast as module
from telegram_code import daily_forecast as nea
from telegram_code.weather_schedule import format_bulletin


NOW = datetime(2026, 9, 30, 5, tzinfo=nea.SGT)


def payload():
    base = NOW.replace(hour=0)
    return {'utc_offset_seconds': 28800,
            'hourly_units': {'time': 'unixtime', 'precipitation_probability': '%', 'precipitation': 'mm'},
            'hourly': {'time': [int((base+timedelta(hours=h)).timestamp()) for h in range(25)],
                       'precipitation_probability': [70 if h in (14,15) else 10 for h in range(25)],
                       'precipitation': [1.2 if h == 15 else .3 if h == 14 else 0 for h in range(25)]}}


def nea_payload():
    start = NOW.replace(hour=6)
    periods = []
    for hour, end_hour, text in [(6,12,'Partly Cloudy'), (12,18,'Thundery Showers'), (18,30,'Partly Cloudy (Night)')]:
        base = NOW.replace(hour=0)
        periods.append({'timePeriod': {'start': (base+timedelta(hours=hour)).isoformat(),
                                       'end': (base+timedelta(hours=end_hour)).isoformat(),
                                       'text': 'INCORRECT provider display date'},
                        'regions': {region: {'text': text} for region in nea.REGIONS}})
    return {'code': 0, 'data': {'records': [{
        'timestamp': (NOW-timedelta(minutes=10)).isoformat(), 'periods': periods,
        'general': {'forecast': {'text': 'Thundery Showers'},
                    'temperature': {'low': 25, 'high': 34, 'unit': 'Degrees Celsius'},
                    'relativeHumidity': {'low': 60, 'high': 95, 'unit': 'Percentage'},
                    'wind': {'direction': 'E', 'speed': {'low': 5, 'high': 15}},
                    'validPeriod': {'start': start.isoformat(), 'end': (start+timedelta(days=1)).isoformat()}}}]}}


class HourlyParsingTests(unittest.TestCase):
    def test_rain_at_14_means_13_to_14_and_midnight_means_previous_hour(self):
        rows = module.parse_hourly(payload(), NOW.date())
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[13]['start'].hour, 13)
        self.assertEqual(rows[13]['chance'], 70)
        self.assertEqual(rows[-1]['start'].hour, 23)
        self.assertEqual(rows[-1]['end'].date(), NOW.date()+timedelta(days=1))
        text = module.format_hourly_location('Main', rows, NOW.date(), NOW)
        self.assertIn('1 pm–3 pm: rain possible; hourly chance up to 70%', text)
        self.assertIn('Highest chance: 2 pm–3 pm (70% chance; 1.2 mm forecast in that hour)', text)

    def test_gaps_and_dry_hours_do_not_merge_separate_rain_windows(self):
        p = payload()
        p['hourly']['precipitation_probability'][16] = 80
        p['hourly']['precipitation_probability'][15] = None
        p['hourly']['precipitation'][15] = None
        rows = module.parse_hourly(p, NOW.date())
        text = module.format_hourly_location('Main', rows, NOW.date(), NOW)
        self.assertIn('1 pm–2 pm:', text)
        self.assertIn('3 pm–4 pm:', text)
        self.assertNotIn('1 pm–4 pm:', text)
        self.assertIn('Some hourly data is unavailable', text)

    def test_noon_omits_elapsed_rain_and_utc_clock_matches(self):
        p = payload()
        p['hourly']['precipitation_probability'][8] = 90
        rows = module.parse_hourly(p, NOW.date())
        noon = NOW.replace(hour=12)
        text = module.format_hourly_location('Main', rows, NOW.date(), noon)
        self.assertNotIn('7 am', text)
        self.assertIn('1 pm–3 pm', text)
        self.assertEqual(text, module.format_hourly_location('Main', rows, NOW.date(), noon.astimezone(timezone.utc)))

    def test_units_timezone_length_and_duplicate_hours_fail_closed(self):
        variants = [None, {}, {'error': True}]
        for field, value in [('utc_offset_seconds',0), ('hourly_units', {'precipitation':'inch'})]:
            p = payload(); p[field] = value; variants.append(p)
        p = payload(); p['hourly']['precipitation'].pop(); variants.append(p)
        p = payload(); p['hourly']['time'][2] = p['hourly']['time'][1]; variants.append(p)
        for p in variants:
            self.assertIsNone(module.parse_hourly(p, NOW.date()))

    def test_missing_or_invalid_values_are_never_reported_as_zero_risk(self):
        p = payload()
        for index in range(25):
            p['hourly']['precipitation_probability'][index] = float('nan')
            p['hourly']['precipitation'][index] = -1
        self.assertIsNone(module.parse_hourly(p, NOW.date()))
        p['hourly']['precipitation'][14] = .8
        rows = module.parse_hourly(p, NOW.date())
        text = module.format_hourly_location('Main', rows, NOW.date(), NOW)
        self.assertIn('chance unavailable', text)
        self.assertNotIn('0% chance', text)

    def test_dry_signal_is_qualified_not_a_guarantee(self):
        p = payload()
        p['hourly']['precipitation_probability'] = [10]*25
        p['hourly']['precipitation'] = [0]*25
        text = module.format_hourly_location('Main', module.parse_hourly(p, NOW.date()), NOW.date(), NOW)
        self.assertIn('Low rain signal', text)
        self.assertNotIn('No rain', text)

    def test_fetch_rounds_location_and_requests_next_midnight(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.json.return_value = payload()
        with patch.object(module.requests, 'get', return_value=response) as get:
            self.assertIsNotNone(module.fetch_hourly(1.35123, 103.82123, NOW.date()))
        kwargs = get.call_args.kwargs
        self.assertEqual(kwargs['timeout'], (3,5))
        self.assertEqual(kwargs['params']['latitude'], 1.35)
        self.assertEqual(kwargs['params']['longitude'], 103.82)
        self.assertEqual(kwargs['params']['end_date'], '2026-10-01')
        self.assertNotIn('label', kwargs['params'])


class NeaDetailsTests(unittest.TestCase):
    def parse(self, data=None):
        return nea.parse_daily_forecast(data or nea_payload(), NOW, scheduled=True)

    def test_periods_use_real_timestamps_and_include_wind_humidity(self):
        forecast = self.parse()
        text = '\n'.join(nea.forecast_detail_lines(forecast, NOW, NOW.date()))
        self.assertIn('12 noon–6 pm: Thundery Showers across Singapore', text)
        self.assertIn('6 pm–midnight', text)
        self.assertIn('Wind: E 5–15 km/h', text)
        self.assertIn('Humidity: 60–95%', text)
        self.assertNotIn('INCORRECT', text)

    def test_regional_differences_and_noon_filtering(self):
        data = nea_payload()
        data['data']['records'][0]['periods'][1]['regions']['east']['text'] = 'Partly Cloudy'
        text = '\n'.join(nea.forecast_detail_lines(self.parse(data), NOW.replace(hour=12), NOW.date()))
        self.assertNotIn('6 am', text)
        self.assertIn('Thundery Showers (north, south, west, central)', text)
        self.assertIn('Partly Cloudy (east)', text)

    def test_official_variable_wind_direction_is_preserved(self):
        data = nea_payload()
        data['data']['records'][0]['general']['wind']['direction'] = 'VARIABLE'
        text = '\n'.join(nea.forecast_detail_lines(self.parse(data), NOW, NOW.date()))
        self.assertIn('Wind: Variable 5–15 km/h', text)

    def test_invalid_optional_data_preserves_daily_summary(self):
        data = nea_payload()
        record = data['data']['records'][0]
        record['general']['wind']['speed']['high'] = float('nan')
        record['general']['relativeHumidity']['high'] = 101
        record['periods'][0]['timePeriod']['start'] = 'bad'
        forecast = self.parse(data)
        self.assertEqual(forecast['outlook'], 'Thundery Showers')
        self.assertNotIn('wind', forecast)
        self.assertNotIn('humidity', forecast)
        self.assertEqual(len(forecast['periods']), 2)

    def test_overlapping_periods_omit_uncertain_timeline(self):
        data = nea_payload()
        data['data']['records'][0]['periods'].append(copy.deepcopy(data['data']['records'][0]['periods'][0]))
        self.assertEqual(self.parse(data)['periods'], [])

    def test_tomorrow_summary_without_periods_does_not_invent_exact_time(self):
        forecast = self.parse()
        forecast['periods'] = []
        forecast['outlook'] = 'Afternoon thundery showers'
        text = '\n'.join(nea.forecast_detail_lines(forecast, NOW, NOW.date()))
        self.assertIn('Detailed time windows are not available', text)
        self.assertNotIn('1 pm', text)


class HourlyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.locations = [{'label':'Main','latitude':1.351,'longitude':103.821},
                          {'label':'Office','latitude':1.352,'longitude':103.822}]
        get_locations = patch.object(module, '_locations', return_value=self.locations)
        get_locations.start(); self.addCleanup(get_locations.stop)
        fetch = patch.object(module, 'fetch_hourly', return_value=module.parse_hourly(payload(), NOW.date()))
        self.fetch = fetch.start(); self.addCleanup(fetch.stop)
        self.cache = {}

    async def test_saved_locations_share_provider_request_and_refresh(self):
        text = await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW)
        self.assertIn('Around Main', text)
        self.assertIn('Around Office', text)
        self.assertIn('Open-Meteo', text)
        self.assertEqual(self.fetch.call_count, 1)
        await module.hourly_forecast_text(self.cache, 43, NOW.date(), NOW+timedelta(minutes=14))
        self.assertEqual(self.fetch.call_count, 1)
        await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW+timedelta(minutes=15))
        self.assertEqual(self.fetch.call_count, 2)

    async def test_source_failure_backs_off_and_never_breaks_bulletin(self):
        self.fetch.side_effect = requests.Timeout()
        self.assertIsNone(await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW))
        self.assertIsNone(await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW+timedelta(minutes=4)))
        self.assertEqual(self.fetch.call_count, 1)
        await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW+timedelta(minutes=5))
        self.assertEqual(self.fetch.call_count, 2)

    async def test_no_location_uses_labelled_reference_but_db_failure_does_not(self):
        with patch.object(module, '_locations', return_value=[]):
            text = await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW)
        self.assertIn('Around Singapore reference point', text)
        with patch.object(module, '_locations', side_effect=OSError()):
            self.assertIsNone(await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW))

    async def test_message_size_with_six_long_names_and_many_separate_rain_windows(self):
        p = payload()
        p['hourly']['precipitation_probability'] = [90 if h%2 else 10 for h in range(25)]
        p['hourly']['precipitation'] = [0]*25
        self.fetch.return_value = module.parse_hourly(p, NOW.date())
        locations = [dict(label='🐈'*79+str(i), latitude=1.3,longitude=103.8) for i in range(6)]
        with patch.object(module, '_locations', return_value=locations):
            text = await module.hourly_forecast_text(self.cache, 42, NOW.date(), NOW)
        forecast = nea.parse_daily_forecast(nea_payload(), NOW, scheduled=True)
        bulletin = format_bulletin(forecast, 'morning', NOW, hourly=text)
        self.assertEqual(bulletin.count('Around '), 6)
        self.assertIn('Showing the 2 strongest', bulletin)
        self.assertLessEqual(len(bulletin.encode('utf-16-le'))//2, 4096)

    async def test_weather_command_includes_hourly_times_with_distinct_attribution(self):
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=42),
                                 effective_message=SimpleNamespace(reply_text=AsyncMock()))
        context = SimpleNamespace(application=SimpleNamespace(bot_data={}))
        forecast = nea.parse_daily_forecast(nea_payload(), NOW, scheduled=True)
        with patch.object(nea, 'get_daily_forecast', AsyncMock(return_value=forecast)), \
                patch.object(nea, '_sg_time', return_value=NOW):
            await nea.weather_command(update, context)
        text = update.effective_message.reply_text.await_args.args[0]
        self.assertIn('1 pm–3 pm', text)
        self.assertIn('NEA source:', text)
        self.assertIn('Hourly source: Open-Meteo', text)


if __name__ == '__main__':
    unittest.main()
