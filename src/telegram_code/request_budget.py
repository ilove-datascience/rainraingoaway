"""Bound interactive weather work without changing conversation concurrency."""
import asyncio
from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


def bounded_weather_reply(callback, *, timeout=20, error_timeout=5):
    @wraps(callback)
    async def reply(update, context):
        started = time.perf_counter()
        outcome = 'completed'
        try:
            async with asyncio.timeout(timeout):
                return await callback(update, context)
        except TimeoutError:
            outcome = 'timed_out'
            # Do not retry a photo: Telegram may have accepted it before timeout.
            try:
                async with asyncio.timeout(error_timeout):
                    await update.effective_message.reply_text(
                        'The weather request is taking too long. Please try again shortly.')
            except Exception as exc:
                logger.warning('Weather timeout notice failed: %s', type(exc).__name__)
        except BaseException:
            outcome = 'failed_or_cancelled'
            raise
        finally:
            logger.info('Weather request outcome=%s handler_elapsed_s=%.3f',
                        outcome, time.perf_counter() - started)
    return reply
