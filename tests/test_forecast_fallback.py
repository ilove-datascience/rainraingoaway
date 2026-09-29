import ast
import asyncio
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code.feedback import FEEDBACK_PROMPT
from telegram_code.feedback_context import location_row, reply_weather_photo


class ForecastFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/telegram_code/methods.py').read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name in
                 {'handle_msg', 'handle_location', 'send_actual_fallback', 'handle_actual',
                  'handle_group_forecasts', 'send_group_actual'}]
        self.forecast_feedback = AsyncMock(return_value='forecast-buttons')
        self.radar_feedback = AsyncMock(return_value='radar-buttons')
        self.env = dict(ReplyKeyboardRemove=lambda: "removed", asyncio=asyncio, Update=object, ContextTypes=SimpleNamespace(DEFAULT_TYPE=object),
                        list_locations=lambda chat: [], main_menu=lambda *args:None, get_location=Mock(return_value=(1.3,103.8)),
                        in_coverage=lambda *a:True, is_fresh=lambda p:bool(p), run_model=AsyncMock(return_value=None),
                        build_location_forecast=Mock(return_value=(BytesIO(b'forecast'), 'Forecast')),
                        build_group_map=Mock(return_value=(BytesIO(b'group-forecast'), 'Group forecast')),
                        build_group_actual=Mock(return_value=(BytesIO(b'group-radar'), 'Group actual radar')),
                        get_latest_radar_png=Mock(return_value=Path('202609111050.png')),
                        build_radar_snapshot_plot=Mock(return_value=(BytesIO(b'radar'), 'NO RAIN DETECTED | ACTUAL RADAR\nRadar observed: 11 Sep, 10:50 SGT')),
                        forecast_feedback=self.forecast_feedback, radar_feedback=self.radar_feedback,
                        location_row=location_row, reply_weather_photo=reply_weather_photo)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'fallback','exec'),self.env)
        self.update=SimpleNamespace(effective_user=SimpleNamespace(id=1),effective_chat=SimpleNamespace(id=1),message=SimpleNamespace(
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
        self.radar_feedback.assert_awaited_once_with(1, [{'latitude': 1.3, 'longitude': 103.8, 'label': 'Main'}], Path('202609111050.png'))
        self.assertEqual(self.update.message.reply_photo.await_args.kwargs['reply_markup'], 'radar-buttons')
        self.assertIn(FEEDBACK_PROMPT, caption)

    async def test_shared_location_uses_requested_coordinates(self):
        await self.env['handle_location'](self.update,self.context,None,'.')
        self.env['build_radar_snapshot_plot'].assert_called_once_with(Path('202609111050.png'),(1.32,103.85))
        self.radar_feedback.assert_awaited_once_with(1, [{'latitude': 1.32, 'longitude': 103.85, 'label': 'Shared location'}], Path('202609111050.png'))
        self.forecast_feedback.assert_not_awaited()

    async def test_fresh_forecast_does_not_fallback(self):
        self.context.application.bot_data['latest_prediction']={'fresh':True}
        await self.env['handle_msg'](self.update,self.context,None,'.')
        self.env['build_radar_snapshot_plot'].assert_not_called()
        self.env['run_model'].assert_not_awaited()
        self.forecast_feedback.assert_awaited_once_with(1, [{'latitude': 1.3, 'longitude': 103.8, 'label': 'Main'}], {'fresh': True})
        self.assertEqual(self.update.message.reply_photo.await_args.kwargs['reply_markup'], 'forecast-buttons')
        self.assertTrue(self.env['build_location_forecast'].return_value[0].closed)

    async def test_shared_forecast_feedback_uses_transient_coordinates(self):
        prediction = {'fresh': True}
        self.context.application.bot_data['latest_prediction'] = prediction
        await self.env['handle_location'](self.update, self.context, None, '.')
        self.forecast_feedback.assert_awaited_once_with(1, [{'latitude': 1.32, 'longitude': 103.85, 'label': 'Shared location'}], prediction)
        self.env['get_location'].assert_not_called()
        self.radar_feedback.assert_not_awaited()

    async def test_current_radar_feedback_matches_saved_main(self):
        self.update.message.text = 'Current radar'
        await self.env['handle_msg'](self.update, self.context, None, '.')
        self.radar_feedback.assert_awaited_once_with(1, [{'latitude': 1.3, 'longitude': 103.8, 'label': 'Main'}], Path('202609111050.png'))
        self.forecast_feedback.assert_not_awaited()

    async def test_group_forecast_feedback_covers_exact_rendered_rows(self):
        rows = [{'latitude': 1.3, 'longitude': 103.8, 'label': 'Main', 'location_id': 0},
                {'latitude': 1.35, 'longitude': 103.9, 'label': 'Office', 'location_id': 7}]
        prediction = dict(prediction='forecast-grid', next_tick='future-time')
        self.env['list_locations'] = Mock(return_value=rows)
        self.update.effective_chat.id = -1001
        self.context.application.bot_data['latest_prediction'] = prediction
        await self.env['handle_msg'](self.update, self.context, None, '.')
        self.forecast_feedback.assert_awaited_once_with(-1001, rows, prediction)
        self.env['build_group_map'].assert_called_once_with(rows, 'forecast-grid', 'future-time')
        self.radar_feedback.assert_not_awaited()

    async def test_group_fallback_feedback_covers_exact_rendered_rows(self):
        rows = [{'latitude': 1.3, 'longitude': 103.8, 'label': 'Main', 'location_id': 0},
                {'latitude': 1.35, 'longitude': 103.9, 'label': 'Office', 'location_id': 7}]
        self.env['list_locations'] = Mock(return_value=rows)
        self.update.effective_chat.id = -1001
        await self.env['handle_msg'](self.update, self.context, None, '.')
        self.radar_feedback.assert_awaited_once_with(-1001, rows, Path('202609111050.png'))
        self.env['build_group_actual'].assert_called_once_with(rows, Path('202609111050.png'))
        self.assertEqual(self.update.message.reply_photo.await_args.kwargs['reply_markup'], 'radar-buttons')
        self.forecast_feedback.assert_not_awaited()

    async def test_forecast_expiring_during_feedback_preparation_falls_back_to_radar(self):
        self.context.application.bot_data['latest_prediction'] = {'fresh': True}
        self.env['is_fresh'] = Mock(side_effect=[True, True, False])
        await self.env['handle_msg'](self.update, self.context, None, '.')
        self.forecast_feedback.assert_awaited_once()
        self.radar_feedback.assert_awaited_once()
        self.update.message.reply_photo.assert_awaited_once()
        self.assertEqual(self.update.message.reply_photo.await_args.kwargs['reply_markup'], 'radar-buttons')
        self.assertTrue(self.env['build_location_forecast'].return_value[0].closed)

    async def test_no_radar_gives_clear_message(self):
        self.env['get_latest_radar_png'].return_value=None
        await self.env['send_actual_fallback'](self.update,'.',(1.3,103.8))
        self.update.message.reply_photo.assert_not_awaited()
        self.assertIn('No radar image',self.update.message.reply_text.await_args.args[0])
        self.radar_feedback.assert_not_awaited()


if __name__=='__main__':
    unittest.main()
