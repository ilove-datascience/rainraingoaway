import ast
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

class RadarNotificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1]/'src/telegram_code/methods.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='check_radar_notifications')
        self.now = datetime(2026,9,22,15,6)
        self.notify = AsyncMock()
        self.deliver = AsyncMock()
        self.decode = Mock(return_value='radar')
        self.env = dict(asyncio=asyncio,datetime=datetime,timedelta=timedelta,
            sg_now=lambda:self.now,get_latest_radar_png=Mock(return_value=Path('202609221500.png')),
            decode_png=self.decode,SOURCE='source',remove_small_echoes=lambda x:x,
            np=SimpleNamespace(flipud=lambda x:x),send_auto_update=self.notify,deliver_notifications=self.deliver)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'radar-job','exec'),self.env)

    async def test_fresh_observation_runs_without_model_or_weather(self):
        await self.env['check_radar_notifications'](None,'.')
        result=self.notify.await_args.args[1]
        self.assertEqual(result['actual_radar'],'radar')
        self.assertNotIn('prediction',result)
        self.deliver.assert_awaited_once()

    async def test_stale_and_future_radar_never_alert(self):
        for now in [datetime(2026,9,22,15,10),datetime(2026,9,22,14,59)]:
            self.now=now
            await self.env['check_radar_notifications'](None,'.')
        self.notify.assert_not_awaited()
        self.decode.assert_not_called()
        self.assertEqual(self.deliver.await_count,2)

    async def test_bad_radar_still_drains_notifications(self):
        self.decode.side_effect=ValueError('bad image')
        await self.env['check_radar_notifications'](None,'.')
        self.notify.assert_not_awaited()
        self.deliver.assert_awaited_once()
