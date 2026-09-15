import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code import heartbeat
from telegram_code.heartbeat import send_heartbeat


@pytest.mark.asyncio
async def test_disabled_without_url():
    with patch.dict('os.environ', {'KUMA_PUSH_URL': ''}), patch('telegram_code.heartbeat.requests.get') as get:
        await send_heartbeat()
        get.assert_not_called()


@pytest.mark.asyncio
async def test_push_and_timeout_failure(caplog):
    url = 'http://kuma/api/push/secret'
    with patch.dict('os.environ', {'KUMA_PUSH_URL': url}), patch('telegram_code.heartbeat.requests.get') as get:
        response = MagicMock()
        get.return_value.__enter__.return_value = response
        await send_heartbeat()
        get.assert_called_once_with(url, timeout=5)
        response.raise_for_status.assert_called_once()
        get.side_effect = requests.Timeout(url)
        await send_heartbeat()
        assert 'heartbeat failed' in caplog.text
        assert url not in caplog.text


@pytest.mark.asyncio
async def test_delayed_frame_regular_push_and_stale_cutoff():
    now = datetime(2026, 9, 16, 1, 50, 7)
    state = {'kuma_processed_observed_at': datetime(2026, 9, 16, 1, 45)}
    with patch.object(heartbeat, 'sg_now', return_value=now) as clock, \
         patch.object(heartbeat.time, 'monotonic', return_value=100) as mono, \
         patch.object(heartbeat, 'send_heartbeat', new_callable=AsyncMock, return_value=True) as push:
        await heartbeat.send_pipeline_heartbeat(state)
        push.assert_awaited_once()
        mono.return_value = 130
        await heartbeat.send_pipeline_heartbeat(state)
        push.assert_awaited_once()
        mono.return_value = 160
        await heartbeat.send_pipeline_heartbeat(state)
        assert push.await_count == 2
        mono.return_value = 1000
        clock.return_value = datetime(2026, 9, 16, 2, 0)
        await heartbeat.send_pipeline_heartbeat(state)
        assert push.await_count == 2
        await heartbeat.send_pipeline_heartbeat({})
        await heartbeat.send_pipeline_heartbeat({'kuma_processed_observed_at': clock.return_value + timedelta(minutes=1)})
        assert push.await_count == 2
