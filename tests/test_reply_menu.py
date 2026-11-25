"""Exercise persistent reply keyboard navigation without Telegram requests."""
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code import menus


def immediate_test(function):
    """Drive mocked, non-I/O callbacks without Windows' loopback-based event loop.

    This deliberately rejects any actual scheduling. Full asyncio integration
    should also run in an environment where Windows socketpair is available.
    """
    @wraps(function)
    def run(*args, **kwargs):
        coroutine = function(*args, **kwargs)
        try:
            coroutine.send(None)
        except StopIteration:
            return
        finally:
            coroutine.close()
        raise AssertionError('Unexpected asynchronous I/O in isolated menu test')
    return run


@pytest.fixture
def flow(monkeypatch):
    async def immediate(function, /, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr(menus.asyncio, 'to_thread', immediate)
    rows = [dict(location_id=0, label='Main', latitude=1.3, longitude=103.8, alerts_enabled=True),
            dict(location_id=7, label='Office', latitude=1.35, longitude=103.9, alerts_enabled=False)]
    monkeypatch.setattr(menus, 'list_locations', Mock(return_value=rows))
    monkeypatch.setattr(menus, 'get_user_mode', Mock(return_value='automatic'))
    message = SimpleNamespace(message_id=10, reply_text=AsyncMock(return_value=SimpleNamespace(message_id=10)))
    update = SimpleNamespace(effective_message=message, message=message,
        effective_chat=SimpleNamespace(id=42, type='private'), effective_user=SimpleNamespace(id=42))
    context = SimpleNamespace(chat_data={}, bot=SimpleNamespace(edit_message_text=AsyncMock()),
                              application=SimpleNamespace(bot_data={}))
    return update, context, rows


def press(update, text):
    update.effective_message.text = text


@immediate_test
async def test_menu_opens_persistent_keyboard_and_second_command_removes_it(flow):
    update, context, _ = flow
    await menus.show_menu(update, context)
    markup = update.message.reply_text.await_args.kwargs['reply_markup']
    assert isinstance(markup, ReplyKeyboardMarkup)
    assert markup.is_persistent and not markup.one_time_keyboard
    assert [len(row) for row in markup.keyboard] == [2, 1, 2]
    assert menus.menu_state(update, context)['open']
    await menus.show_menu(update, context)
    assert isinstance(update.message.reply_text.await_args.kwargs['reply_markup'], ReplyKeyboardRemove)
    assert not menus.menu_state(update, context)


@immediate_test
async def test_navigation_and_help_use_keyboard_not_message_buttons(flow):
    update, context, _ = flow
    await menus.show_menu(update, context)
    for text in ('Locations & alerts', '○ Office', '‹ Back', '‹ Back', 'Help'):
        press(update, text)
        await menus.menu_text(update, context)
        assert isinstance(update.message.reply_text.await_args.kwargs['reply_markup'], ReplyKeyboardMarkup)
    assert 'New locations start with alerts on' in update.message.reply_text.await_args.args[0]
    context.bot.edit_message_text.assert_not_awaited()


@immediate_test
async def test_alert_switch_is_explicit_and_stale_button_does_not_retoggle(flow, monkeypatch):
    update, context, rows = flow
    def save(chat, location, enabled):
        assert (chat, location, enabled) == (42, 7, True)
        rows[1]['alerts_enabled'] = enabled
        return True
    setter = Mock(side_effect=save)
    monkeypatch.setattr(menus, 'set_location_alert', setter)
    await menus.open_location(update, context, '7')
    press(update, 'Turn alerts on · Office')
    await menus.menu_text(update, context)
    setter.assert_called_once_with(42, 7, True)
    markup = update.message.reply_text.await_args.kwargs['reply_markup']
    assert markup.keyboard[0][0].text == 'Turn alerts off · Office'
    await menus.menu_text(update, context)
    setter.assert_called_once()
    assert 'no longer active' in update.message.reply_text.await_args.args[0]


@immediate_test
async def test_other_member_cannot_act_on_another_members_selection(flow, monkeypatch):
    update, context, _ = flow
    update.effective_chat = SimpleNamespace(id=-1001, type='supergroup')
    await menus.open_location(update, context, '7')
    assert update.message.reply_text.await_args.kwargs['reply_to_message_id'] == 10
    assert update.message.reply_text.await_args.kwargs['reply_markup'].selective
    update.effective_user = SimpleNamespace(id=99)
    setter = Mock()
    monkeypatch.setattr(menus, 'set_location_alert', setter)
    press(update, 'Turn alerts on · Office')
    await menus.menu_text(update, context)
    setter.assert_not_called()


@immediate_test
async def test_close_button_clears_setup_and_keyboard(flow):
    update, context, _ = flow
    await menus.show_menu(update, context)
    context.chat_data['new_location_label'] = 'School'
    press(update, 'Close')
    assert await menus.menu_text(update, context) == -1
    assert not menus.menu_state(update, context)
    assert 'new_location_label' not in context.chat_data
    assert isinstance(update.message.reply_text.await_args.kwargs['reply_markup'], ReplyKeyboardRemove)


@immediate_test
async def test_remove_requires_confirmation_for_exact_location(flow, monkeypatch):
    update, context, _ = flow
    remove = Mock(return_value=True)
    monkeypatch.setattr(menus, 'remove_named_location', remove)
    await menus.open_location(update, context, '7')
    press(update, 'Remove location')
    await menus.menu_text(update, context)
    remove.assert_not_called()
    press(update, 'Confirm remove · Office')
    await menus.menu_text(update, context)
    remove.assert_called_once_with(42, 'Office', 7)


@immediate_test
async def test_old_inline_buttons_are_retired_without_setting_changes(flow):
    update, context, _ = flow
    update.callback_query = SimpleNamespace(message=update.message, answer=AsyncMock(),
                                            edit_message_reply_markup=AsyncMock())
    assert await menus.menu_callback(update, context) is None
    update.callback_query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)


@immediate_test
async def test_forecast_and_radar_keep_keyboard_state(flow, monkeypatch):
    update, context, _ = flow
    forecast, radar = AsyncMock(), AsyncMock()
    monkeypatch.setattr(menus, 'bounded_weather_reply', lambda callback: callback)
    monkeypatch.setitem(sys.modules, 'telegram_code.methods', SimpleNamespace(handle_group_forecasts=forecast, handle_actual=radar))
    await menus.show_menu(update, context)
    for text in ('My forecast', 'Current radar'):
        press(update, text)
        await menus.menu_text(update, context, model='model', folder_path='data', norm_stats='norm')
        assert menus.menu_state(update, context)['open']
    forecast.assert_awaited_once_with(update, context, model='model', folder_path='data', norm_stats='norm')
    radar.assert_awaited_once_with(update, context, folder_path='data')
    update.message.reply_text.assert_awaited_once()


def test_forecast_handlers_do_not_remove_keyboard():
    import ast
    path = Path(__file__).resolve().parents[1] / 'src/telegram_code/methods.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ReplyKeyboardRemove')
