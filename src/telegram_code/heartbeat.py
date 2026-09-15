"""Optional Uptime Kuma push notification for a successful forecast cycle."""
import asyncio
import logging
import os

import requests

logger = logging.getLogger(__name__)


def _push(url):
    try:
        with requests.get(url, timeout=5) as response:
            response.raise_for_status()
    except requests.RequestException:
        # Exception text can contain the secret URL; never log it.
        logger.warning("Uptime Kuma heartbeat failed; bot will continue")


async def send_heartbeat():
    # The application's database module loads the project .env at startup.
    url = os.getenv("KUMA_PUSH_URL", "").strip()
    if url:
        await asyncio.to_thread(_push, url)
