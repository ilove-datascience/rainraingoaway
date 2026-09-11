import ast
import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock


class ForecastFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/telegram_code/methods.py').read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name in
                 {'handle_msg', 'handle_location', 'send_actual_fallback'}]
        self.env = dict(asyncio=asyncio, Update=object, ContextTypes=SimpleNamespace(DEFAULT_TYPE=object),
                        main_menu=lambda:None, get_location=Mock(return_value=(1.3,103.8)),
                        in_coverage=lambda *a:True, is_fresh=lambda p:bool(p), run_model=AsyncMock(return_value=None),
                        build_location_forecast=Mock(return_value=(BytesIO(b'forecast'), 'Forecast')),
                        get_latest_radar_png=Mock(return_value=Path('202609111050.png')),
                        build_radar_snapshot_plot=Mock(return_value=(BytesIO(b'radar'), 'NO RAIN DETECTED | ACTUAL RADAR\nRadar observed: 11 Sep, 10:50 SGT')))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'fallback','exec'),self.env)
        self.update=SimpleNamespace(effective_user=SimpleNamespace(id=1),message=SimpleNamespace(
            text='My forecast',location=SimpleNamespace(latitude=1.32,longitude=103.85),
            reply_text=AsyncMock(),reply_photo=AsyncMock()))
        self.context=SimpleNamespace(application=SimpleNamespace(bot_data={}))

    async def test_missing_forecast_sends_actual_for_saved_location(self):
        await self.env['handle_msg'](self.update,self.context,None,'.')
        self.env['build_radar_snapshot_plot'].assert_called_once_with(Path('202609111050.png'),(1.3,103.8))
        caption=self.update.message.reply_photo.await_args.kwargs['caption']
        self.assertIn('FORECAST DELAYED',caption)
        self.assertIn('10:50 SGT',caption)
        self.env['build_location_forecast'].assert_not_called()

    async def test_shared_location_uses_requested_coordinates(self):
        await self.env['handle_location'](self.update,self.context,None,'.')
        self.env['build_radar_snapshot_plot'].assert_called_once_with(Path('202609111050.png'),(1.32,103.85))

    async def test_fresh_forecast_does_not_fallback(self):
        self.context.application.bot_data['latest_prediction']={'fresh':True}
        await self.env['handle_msg'](self.update,self.context,None,'.')
        self.env['build_radar_snapshot_plot'].assert_not_called()
        self.env['run_model'].assert_not_awaited()

    async def test_no_radar_gives_clear_message(self):
        self.env['get_latest_radar_png'].return_value=None
        await self.env['send_actual_fallback'](self.update,'.',(1.3,103.8))
        self.update.message.reply_photo.assert_not_awaited()
        self.assertIn('No radar image',self.update.message.reply_text.await_args.args[0])


if __name__=='__main__':
    unittest.main()
