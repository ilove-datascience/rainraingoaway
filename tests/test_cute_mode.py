import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram.ext import ExtBot
from telegram_code import cute_mode


@pytest.fixture
def preferences(tmp_path, monkeypatch):
    monkeypatch.setattr(cute_mode, 'PREFERENCES_PATH', tmp_path / 'preferences.sqlite3')


def test_persistent_toggle_is_isolated_per_user(preferences):
    assert not cute_mode.cute_enabled(1)
    assert cute_mode.toggle_cute(1)
    assert cute_mode.cute_enabled(1)
    assert not cute_mode.cute_enabled(2)
    assert not cute_mode.toggle_cute(1)
    assert not cute_mode.cute_enabled(1)


@pytest.mark.asyncio
async def test_replies_and_alert_captions_use_current_preference(preferences):
    bot = cute_mode.CuteBot('123:fake')
    text = 'RAIN EXPECTED | FORECAST\nForecast for: 15 Sep, 12:05 SGT'
    markup = object()
    with patch.object(ExtBot, 'send_message', new_callable=AsyncMock) as message, \
         patch.object(ExtBot, 'send_photo', new_callable=AsyncMock) as photo:
        await bot.send_message(1, text, reply_markup=markup)
        assert message.call_args.args[1] == text
        cute_mode.toggle_cute(1)
        await bot.send_message(1, text, reply_markup=markup)
        assert message.call_args.args[1].startswith(text + '\n\n')
        assert 'meow' in message.call_args.args[1]
        assert message.call_args.kwargs['reply_markup'] is markup
        await bot.send_photo(1, b'png', caption=text)
        assert photo.call_args.args[2] == message.call_args.args[1]
        await bot.send_message(2, text)
        assert message.call_args.args[1] == text
        cute_mode.toggle_cute(1)
        await bot.send_photo(1, b'png', caption=text)
        assert photo.call_args.args[2] == text


@pytest.mark.asyncio
async def test_hidden_command_toggles_without_changing_conversation(preferences):
    reply = AsyncMock()
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=123, type='private'),
                             effective_message=SimpleNamespace(reply_text=reply))
    assert await cute_mode.cutemode(update, None) is None
    assert cute_mode.cute_enabled(123)
    assert 'reply_markup' not in reply.call_args.kwargs
    await cute_mode.cutemode(update, None)
    assert not cute_mode.cute_enabled(123)
    update.effective_chat.type = 'group'
    await cute_mode.cutemode(update, None)
    assert not cute_mode.cute_enabled(123)


def test_preserves_facts_and_caption_limits():
    text = 'RAIN EXPECTED TO CLEAR | FORECAST\nExpected intensity: Light\nArea: ~500 m'
    styled = cute_mode.cat_text(text, 1024)
    assert styled.startswith(text)
    assert 'may be padding away' in styled
    assert cute_mode.cat_text('x' * 1024, 1024) == 'x' * 1024
    assert cute_mode.cat_text(None, 1024) is None
