"""Weather retries preserve the requested timestamp and do not lose gaps."""
import ast
from pathlib import Path
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import tempfile
import queue
import unittest

class WeatherRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(__file__).resolve().parents[1] / 'src/scraping/gov_api.py'
        nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name in ('collect_pending_weather', 'main')]
        class Frame:
            empty = False
            def __len__(self): return 1
        class Clock:
            @staticmethod
            def now(tz): return datetime(2026,9,18,12,20)
        self.env = dict(datetime=Clock, SINGAPORE_TZ=None, WEATHER_RETRY_MAX_AGE=timedelta(hours=1), OUTPUT_DIR=Path(self.tmp.name), fetch_once=Mock(return_value=Frame()), save_to_csv=Mock())
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'weather','exec'),self.env)

    def test_failed_timestamp_retried_and_announced_only_after_save(self):
        tick = datetime(2026,9,18,12)
        pending = {tick}
        ready = queue.Queue()
        frame = self.env['fetch_once'].return_value
        self.env['fetch_once'].side_effect = [RuntimeError('429'),frame]
        self.env['collect_pending_weather'](pending,ready)
        self.assertEqual(pending,{tick})
        self.assertTrue(ready.empty())
        self.env['collect_pending_weather'](pending,ready)
        self.assertFalse(pending)
        self.assertEqual(ready.get_nowait(),202609181200)
        self.assertEqual([c.args[0] for c in self.env['fetch_once'].call_args_list],[tick,tick])
        self.env['save_to_csv'].assert_called_once_with(frame,timestamp='202609181200')

    def test_newest_and_oldest_processed_without_losing_middle(self):
        ticks = [datetime(2026,9,18,12)+timedelta(minutes=5*i) for i in range(4)]
        pending = set(ticks)
        self.env['collect_pending_weather'](pending)
        self.assertEqual([c.args[0] for c in self.env['fetch_once'].call_args_list],[ticks[-1],ticks[0]])
        self.assertEqual(pending,set(ticks[1:3]))

    def test_retries_expire_at_one_hour(self):
        expired = datetime(2026,9,18,11,20)
        fresh = datetime(2026,9,18,12,15)
        pending = {expired,fresh}
        self.env['collect_pending_weather'](pending)
        self.assertFalse(pending)
        self.env['fetch_once'].assert_called_once_with(fresh)

    def test_slow_request_does_not_start_expired_older_retry(self):
        ticks = [datetime(2026,9,18,11,30),datetime(2026,9,18,12,20)]
        self.env['datetime'] = SimpleNamespace(now=Mock(side_effect=[
            datetime(2026,9,18,12,20), datetime(2026,9,18,12,20), datetime(2026,9,18,12,31)]))
        pending = set(ticks)
        self.env['collect_pending_weather'](pending)
        self.env['fetch_once'].assert_called_once_with(ticks[-1])
        self.assertFalse(pending)

    def test_slow_loop_queues_every_elapsed_five_minute_tick(self):
        self.env['timedelta'] = timedelta
        self.env['datetime'] = SimpleNamespace(now=Mock(side_effect=[
            datetime(2026,9,18,12), datetime(2026,9,18,12), datetime(2026,9,18,12,12)]))
        captured = []
        def collect(pending, ready):
            captured.append(sorted(pending))
            pending.clear()
        self.env['collect_pending_weather'] = collect
        self.env['time'] = SimpleNamespace(sleep=Mock(side_effect=[None,KeyboardInterrupt]))
        with self.assertRaises(KeyboardInterrupt):
            self.env['main']()
        self.assertEqual(captured[1], [datetime(2026,9,18,12,5),datetime(2026,9,18,12,10)])

    def test_existing_file_is_not_refetched(self):
        (Path(self.tmp.name)/'weather_202609181200.csv').write_text('existing')
        pending = {datetime(2026,9,18,12)}
        self.env['collect_pending_weather'](pending)
        self.assertFalse(pending)
        self.env['fetch_once'].assert_not_called()

    def test_empty_response_remains_pending(self):
        self.env['fetch_once'].return_value.empty = True
        pending = {datetime(2026,9,18,12)}
        self.env['collect_pending_weather'](pending)
        self.assertTrue(pending)
        self.env['save_to_csv'].assert_not_called()

if __name__ == '__main__': unittest.main()
