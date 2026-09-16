import ast
import queue
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock


class QueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.now = datetime(2026, 9, 17, 0, 50, 30)
        self.prediction = {'observed_at': self.now.replace(second=0), 'next_tick': datetime(2026,9,17,0,55)}
        self.run = AsyncMock(return_value=self.prediction)
        self.notify = AsyncMock()
        self.deliver = AsyncMock()
        self.clock = Mock(side_effect=lambda: self.now)
        tree = ast.parse((Path(__file__).resolve().parents[1]/'src/telegram_code/methods.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'check_model_queue')
        env = dict(queue=queue, run_model=self.run, send_auto_update=self.notify,
                   deliver_notifications=self.deliver, main_menu=lambda:None,
                   datetime=datetime, timedelta=timedelta, Path=Path, SEQUENCE_LENGTH=3, sg_now=self.clock)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'queue','exec'),env)
        self.check = env['check_model_queue']
        self.context = types.SimpleNamespace(application=types.SimpleNamespace(bot_data={}))
        self.arrivals = queue.Queue()

    def frames(self):
        for minute in [40,45,50]:
            (self.folder/f'2026091700{minute}.png').write_bytes(b'frame')

    async def poll(self):
        await self.check(self.context,None,self.folder,None,self.arrivals)

    async def test_recent_failure_retried_then_succeeds(self):
        self.frames()
        self.run.side_effect = [None,self.prediction]
        self.arrivals.put(202609170050)
        await self.poll()
        self.assertIn(202609170050,self.context.application.bot_data['pending_model_ticks'])
        await self.poll()
        self.assertFalse(self.context.application.bot_data['pending_model_ticks'])
        self.assertEqual(self.run.await_count,2)
        self.notify.assert_awaited_once_with(self.context,self.prediction)
        self.assertEqual(self.deliver.await_count,2)

    async def test_old_and_exact_deadline_ticks_expire_without_inference(self):
        self.frames()
        for tick in [202609161120,202609161530,202609170045]:
            self.arrivals.put(tick)
        self.now = datetime(2026,9,17,0,50)
        await self.poll()
        self.run.assert_not_awaited()
        self.notify.assert_not_awaited()
        self.assertFalse(self.context.application.bot_data['pending_model_ticks'])
        self.deliver.assert_awaited_once()

    async def test_missing_history_waits_then_expires(self):
        self.arrivals.put(202609170050)
        await self.poll()
        self.run.assert_not_awaited()
        self.assertIn(202609170050,self.context.application.bot_data['pending_model_ticks'])
        self.now = datetime(2026,9,17,0,55)
        await self.poll()
        self.assertFalse(self.context.application.bot_data['pending_model_ticks'])

    async def test_repaired_history_runs_before_deadline(self):
        self.arrivals.put(202609170050)
        await self.poll()
        self.frames()
        await self.poll()
        self.run.assert_awaited_once()

    async def test_inference_crossing_deadline_does_not_send_or_publish(self):
        self.frames()
        async def slow(*args,**kwargs):
            self.now = datetime(2026,9,17,0,55)
            return self.prediction
        self.run.side_effect = slow
        self.arrivals.put(202609170050)
        await self.poll()
        self.notify.assert_not_awaited()
        self.assertNotIn('latest_prediction',self.context.application.bot_data)
        self.assertFalse(self.context.application.bot_data['pending_model_ticks'])

    async def test_newest_first_and_duplicate_ignored(self):
        self.frames()
        for tick in [202609170045,202609170050]:
            self.arrivals.put(tick)
        await self.poll()
        self.arrivals.put(202609170050)
        await self.poll()
        self.run.assert_awaited_once()
        self.assertEqual(self.run.await_args.kwargs['tick'],202609170050)

if __name__ == '__main__':
    unittest.main()
