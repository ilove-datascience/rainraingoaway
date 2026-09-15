import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
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
