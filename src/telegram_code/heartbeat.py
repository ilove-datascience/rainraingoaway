"""Optional Uptime Kuma heartbeat for the running Telegram event loop."""
import asyncio
import logging
import os

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


async def heartbeat_job(context):
    """Runs on the bot event loop, independently of radar, inference and database jobs."""
    await send_heartbeat()


def schedule_heartbeat(job_queue):
    if os.getenv('KUMA_PUSH_URL', '').strip():
        job_queue.run_repeating(heartbeat_job, interval=30, first=1,
                                name='kuma-bot-heartbeat')
        logger.info('Kuma bot heartbeat enabled: every 30 seconds, independent of radar')
