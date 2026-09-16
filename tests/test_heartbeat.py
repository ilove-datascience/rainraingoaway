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


def test_schedule_independent_job_and_disabled_configuration():
    queue = MagicMock()
    with patch.dict('os.environ', {'KUMA_PUSH_URL': ''}):
        heartbeat.schedule_heartbeat(queue)
        queue.run_repeating.assert_not_called()
    with patch.dict('os.environ', {'KUMA_PUSH_URL': 'http://kuma/api/push/secret'}):
        heartbeat.schedule_heartbeat(queue)
    queue.run_repeating.assert_called_once_with(heartbeat.heartbeat_job,
        interval=30, first=1, name='kuma-bot-heartbeat')


@pytest.mark.asyncio
async def test_job_needs_no_radar_prediction_or_database_state():
    with patch.object(heartbeat, 'send_heartbeat', new_callable=AsyncMock) as push:
        await heartbeat.heartbeat_job(None)
        push.assert_awaited_once()
