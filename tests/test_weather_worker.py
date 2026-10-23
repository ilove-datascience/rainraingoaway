"""Slow inference must not block radar replies or the default I/O executor."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import threading
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock
import numpy as np
import torch


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_observations_use_source_radar_and_preserve_forecast_output(self):
        source = Path('src/telegram_code/methods.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_run_model_sync')
        raw = torch.tensor([[[[.2, .4], [.6, .8]]]])
        model = Mock(return_value=(raw, None, None, torch.zeros_like(raw)))
        model.parameters.return_value = iter([torch.zeros(1)])
        observed = np.array([[0., .03], [.07, .1]])
        decode = Mock(return_value=observed)
        env = dict(torch=torch, np=np, Path=Path, datetime=datetime, timedelta=timedelta,
                   SEQUENCE_LENGTH=3, SOURCE='source',
                   get_previous_ticks=lambda *a, **k: [202610240130, 202610240125, 202610240120],
                   build_multimodal_input=lambda *a, **k: torch.full((1, 3, 7, 2, 2), .12345),
                   clean_rain_mask=lambda p: np.array([[1, 0], [1, 1]]),
                   remove_small_echoes=lambda grid: grid, decode_png=decode)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'worker', 'exec'), env)
        result = env['_run_model_sync'](model, 'radar', tick=202610240130)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result['actual_radar'], np.flipud(observed))
        np.testing.assert_allclose(result['prediction'], [[.6, .8], [.2, 0]])
        decode.assert_called_once_with(Path('radar/202610240130.png'), 'source')

    async def test_radar_io_runs_while_model_is_blocked(self):
        source = Path('src/telegram_code/methods.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_model')
        entered, release = threading.Event(), threading.Event()
        def slow(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('test model was not released')
            return 'prediction'
        with ThreadPoolExecutor(max_workers=1) as pool:
            env = dict(asyncio=asyncio, partial=partial, _model_executor=pool,
                       _run_model_sync=slow)
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'worker', 'exec'), env)
            task = asyncio.create_task(env['run_model'](None, '.'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.assertEqual(await asyncio.wait_for(asyncio.to_thread(lambda: 'radar'), .5), 'radar')
                self.assertFalse(task.done())
            finally:
                release.set()
                self.assertEqual(await task, 'prediction')
