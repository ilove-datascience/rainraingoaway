"""Offline regression tests; no API credentials or model runtime required."""
import ast
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
import queue
import tempfile
import types
import unittest
from unittest.mock import Mock
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frame_cache', ROOT / 'src/scraping/frame_cache.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def radar_functions(data_dir):
    # Isolate HTTP/filesystem functions from unrelated optional runtime imports.
    source = ast.parse((ROOT / 'src/scraping/rain_areas.py').read_text())
    funcs = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in {'fetch_radar_snapshot', 'scrape_once'}]
    requests = types.SimpleNamespace(
        get=Mock(), exceptions=types.SimpleNamespace(Timeout=TimeoutError, HTTPError=RuntimeError))
    env = dict(DATA_DIR=Path(data_dir), Path=Path, datetime=datetime, timedelta=timedelta,
               requests=requests, time=types.SimpleNamespace(sleep=Mock()), uuid=uuid)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'radar-functions', 'exec'), env)
    return env, requests


class ScraperTests(unittest.TestCase):
    def test_cached_current_makes_no_request(self):
        with tempfile.TemporaryDirectory() as directory:
            env, requests = radar_functions(directory)
            path = Path(directory) / '70km/png/202609101200.png'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'existing')
            self.assertEqual(env['fetch_radar_snapshot']('70km', 202609101200), (202609101200, path, False))
            requests.get.assert_not_called()

    def test_missing_current_reuses_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            env, requests = radar_functions(directory)
            path = Path(directory) / '70km/png/202609101155.png'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'existing')
            requests.get.return_value = types.SimpleNamespace(status_code=404, raise_for_status=Mock(side_effect=RuntimeError('404')))
            self.assertEqual(env['fetch_radar_snapshot']('70km', 202609101200), (202609101155, path, True))
            self.assertEqual(requests.get.call_count, 1)
            self.assertEqual(path.read_bytes(), b'existing')

    def test_download_published_without_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            env, requests = radar_functions(directory)
            requests.get.return_value = types.SimpleNamespace(status_code=200, raise_for_status=Mock(), content=b'new')
            _, path, fallback = env['fetch_radar_snapshot']('70km', 202609101200)
            self.assertFalse(fallback)
            self.assertEqual(path.read_bytes(), b'new')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failure_keeps_fast_retry(self):
        env, _ = radar_functions('.')
        env['fetch_radar_snapshot'] = Mock(side_effect=RuntimeError('network'))
        self.assertTrue(env['scrape_once'](('70km',)))

    def test_either_arrival_order_builds_once(self):
        for order in [('radar', 'weather'), ('weather', 'radar')]:
            with self.subTest(order=order):
                available = set()
                events = iter([*order, 'radar', None])
                tick = 202609101200
                def get():
                    event = next(events)
                    if event is None:
                        return None
                    available.add(event)
                    return tick
                build = Mock(side_effect=lambda *a, **kw: object() if len(available) == 2 else None)
                ready = queue.Queue()
                worker.run_frame_cache_worker(types.SimpleNamespace(get=get), ready, build)
                self.assertEqual(build.call_count, 2)
                self.assertEqual(ready.get_nowait(), tick)
                self.assertTrue(ready.empty())


if __name__ == '__main__':
    unittest.main()
