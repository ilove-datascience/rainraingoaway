"""Optional Uptime Kuma push notification for a successful forecast cycle."""
import asyncio
import logging
import os
import time
from datetime import timedelta
from telegram_code.forecast_policy import sg_now

import requests

logger = logging.getLogger(__name__)


def _push(url):
    try:
        with requests.get(url, timeout=5) as response:
            response.raise_for_status()
        return True
    except requests.RequestException:
        # Exception text can contain the secret URL; never log it.
        logger.warning("Uptime Kuma heartbeat failed; bot will continue")
        return False


async def send_heartbeat():
    # The application's database module loads the project .env at startup.
    url = os.getenv("KUMA_PUSH_URL", "").strip()
    if url:
        return await asyncio.to_thread(_push, url)
    return False


async def send_pipeline_heartbeat(bot_data):
    """Allow source publication lag, but never report indefinitely stale work as healthy."""
    observed = bot_data.get('kuma_processed_observed_at')
    if observed is None or not timedelta(0) <= sg_now() - observed < timedelta(minutes=15):
        return
    now = time.monotonic()
    last = bot_data.get('kuma_last_push')
    if last is not None and now - last < 60:
        return
    if await send_heartbeat():
        bot_data['kuma_last_push'] = now
