"""Exercise chat isolation without loading the forecast model or a database."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock


class ChatSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'src/telegram_code/methods.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names = {'start', 'receive_location', 'receive_mode', 'update_mode', 'handle_msg'}
        nodes = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name in names]
        self.locations = {42: (1.3, 103.8)}
        self.modes = {42: 'automatic'}
        def add_user(chat):
            self.modes.setdefault(chat, 'manual')
        def add_location(chat, lat, long):
            self.locations[chat] = (lat, long)
            return True
        def save_mode(chat, mode):
            self.modes[chat] = mode
            return True
        self.env = dict(asyncio=asyncio, Update=object,
                        ContextTypes=SimpleNamespace(DEFAULT_TYPE=object),
                        ConversationHandler=SimpleNamespace(END=-1),
                        WAITING_FOR_LOCATION=1, WAITING_FOR_MODE=2,
                        get_location=self.locations.get, add_user=add_user,
                        add_location=add_location, save_mode_choice=save_mode,
                        get_user_mode=self.modes.get, in_coverage=lambda *a: True,
                        main_menu=lambda: None, ReplyKeyboardMarkup=lambda **k: None,
                        settings_lock=lambda c: asyncio.Lock())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), self.env)
        self.context = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    def update(self, chat, user=42, text='My forecast'):
        return SimpleNamespace(effective_chat=SimpleNamespace(id=chat),
                               effective_user=SimpleNamespace(id=user),
                               message=SimpleNamespace(text=text,
                                   location=SimpleNamespace(latitude=1.35, longitude=103.9),
                                   reply_text=AsyncMock()))

    async def test_new_groups_start_fresh_and_private_settings_survive(self):
        for chat in (-1001234567890, -1009876543210):
            self.assertEqual(await self.env['start'](self.update(chat), self.context), 1)
            self.assertEqual(self.modes[chat], 'manual')
            self.assertNotIn(chat, self.locations)
        self.assertEqual(await self.env['start'](self.update(42), self.context), -1)
        self.assertEqual(self.modes[42], 'automatic')

    async def test_members_share_group_settings_without_changing_personal_settings(self):
        chat = -1001234567890
        await self.env['start'](self.update(chat), self.context)
        await self.env['receive_location'](self.update(chat, user=99), self.context)
        await self.env['receive_mode'](self.update(chat, text='1 - Automatic rain updates'), self.context)
        self.assertEqual(self.locations[chat], (1.35, 103.9))
        self.assertEqual(self.modes[chat], 'automatic')
        self.assertEqual(self.locations[42], (1.3, 103.8))
        self.assertNotIn(99, self.locations)
        self.assertEqual(await self.env['start'](self.update(chat, user=99), self.context), -1)

    async def test_unconfigured_group_forecast_does_not_use_personal_location(self):
        update = self.update(-1001234567890)
        await self.env['handle_msg'](update, self.context, None, '.')
        self.assertIn('/start', update.message.reply_text.await_args.args[0])


if __name__ == '__main__':
    unittest.main()
