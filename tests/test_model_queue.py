import ast
import asyncio
import queue
from datetime import datetime
from pathlib import Path
import types
import unittest
from unittest.mock import AsyncMock


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_tick_retained_and_exact_tick_used(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'src/telegram_code/methods.py').read_text(encoding='utf-8'))
        function=next(node for node in tree.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='check_model_queue')
        prediction={'observed_at':datetime(2026,9,10,12)}
        run=AsyncMock(side_effect=[None,prediction])
        notify=AsyncMock(return_value=True)
        heartbeat=AsyncMock()
        deliver=AsyncMock()
        env=dict(queue=queue,run_model=run,send_auto_update=notify,deliver_notifications=deliver,main_menu=lambda:None,
                 send_heartbeat=heartbeat,is_fresh=lambda result:True)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'queue','exec'),env)
        context=types.SimpleNamespace(application=types.SimpleNamespace(bot_data={}))
        arrivals=queue.Queue()
        arrivals.put(202609101200)
        await env['check_model_queue'](context,None,'.',None,arrivals)
        self.assertIn(202609101200,context.application.bot_data['pending_model_ticks'])
        heartbeat.assert_not_awaited()
        await env['check_model_queue'](context,None,'.',None,arrivals)
        self.assertFalse(context.application.bot_data['pending_model_ticks'])
        self.assertEqual([call.kwargs['tick'] for call in run.await_args_list],[202609101200,202609101200])
        notify.assert_awaited_once_with(context,prediction)
        self.assertEqual(deliver.await_count,2)
        heartbeat.assert_awaited_once()

        # Replaying stale data must not make a stopped scraper look healthy.
        env['is_fresh'] = lambda result: False
        run.side_effect = None
        run.return_value = prediction
        arrivals.put(202609101205)
        await env['check_model_queue'](context,None,'.',None,arrivals)
        heartbeat.assert_awaited_once()

        env['is_fresh'] = lambda result: True
        notify.side_effect = RuntimeError('database unavailable')
        arrivals.put(202609101210)
        with self.assertRaises(RuntimeError):
            await env['check_model_queue'](context,None,'.',None,arrivals)
        heartbeat.assert_awaited_once()


if __name__=='__main__':
    unittest.main()
