import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram_code.request_budget import bounded_weather_reply


@pytest.mark.asyncio
async def test_success_preserves_return_and_sends_no_extra_reply():
    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
    callback = AsyncMock(return_value='sent')
    assert await bounded_weather_reply(callback)(update, None) == 'sent'
    update.effective_message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_stalled_photo_is_cancelled_without_duplicate_retry():
    photo = AsyncMock(side_effect=lambda: None)
    cancelled = asyncio.Event()
    async def stalled(*args):
        await photo()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
    await asyncio.wait_for(bounded_weather_reply(stalled, timeout=.02)(update, None), .5)
    assert cancelled.is_set()
    photo.assert_awaited_once()
    update.effective_message.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeout_notice_also_has_a_deadline():
    async def stalled(*args):
        await asyncio.Event().wait()
    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=stalled))
    await asyncio.wait_for(bounded_weather_reply(stalled, timeout=.02, error_timeout=.02)(update, None), .5)


@pytest.mark.asyncio
async def test_shutdown_cancellation_is_not_swallowed():
    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=AsyncMock()))
    callback = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await bounded_weather_reply(callback)(update, None)
    update.effective_message.reply_text.assert_not_awaited()
