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
        notify=AsyncMock()
        deliver=AsyncMock()
        env=dict(queue=queue,run_model=run,send_auto_update=notify,deliver_notifications=deliver,main_menu=lambda:None)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'queue','exec'),env)
        context=types.SimpleNamespace(application=types.SimpleNamespace(bot_data={}))
        arrivals=queue.Queue()
        arrivals.put(202609101200)
        await env['check_model_queue'](context,None,'.',None,arrivals)
        self.assertIn(202609101200,context.application.bot_data['pending_model_ticks'])
        await env['check_model_queue'](context,None,'.',None,arrivals)
        self.assertFalse(context.application.bot_data['pending_model_ticks'])
        self.assertEqual([call.kwargs['tick'] for call in run.await_args_list],[202609101200,202609101200])
        notify.assert_awaited_once_with(context,prediction)
        self.assertEqual(deliver.await_count,2)


if __name__=='__main__':
    unittest.main()
