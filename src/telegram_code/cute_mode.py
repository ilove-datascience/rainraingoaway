"""Hidden, persistent cat-style messages for private bot chats."""
import asyncio
import logging
import sqlite3
from pathlib import Path

from telegram.ext import ExtBot

PREFERENCES_PATH = Path(__file__).resolve().parents[2] / 'data' / 'bot_preferences.sqlite3'
logger = logging.getLogger(__name__)


def cute_enabled(chat_id):
    if not PREFERENCES_PATH.exists():
        return False
    try:
        with sqlite3.connect(PREFERENCES_PATH) as conn:
            row = conn.execute('SELECT enabled FROM cute_mode WHERE chat_id=?', (str(chat_id),)).fetchone()
        return bool(row and row[0])
    except sqlite3.Error:
        logger.exception('Could not read cute mode preference')
        return False


def toggle_cute(chat_id):
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PREFERENCES_PATH) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS cute_mode (chat_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL)')
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT enabled FROM cute_mode WHERE chat_id=?', (str(chat_id),)).fetchone()
        enabled = not bool(row and row[0])
        conn.execute('INSERT OR REPLACE INTO cute_mode VALUES (?, ?)', (str(chat_id), int(enabled)))
    return enabled


def cat_text(text, limit):
    """Style weather updates; leave menus and confirmations concise."""
    if not text:
        return text
    title = text.split('\n', 1)[0]
    if 'DELAYED' in title or 'UNAVAILABLE' in title:
        extra = 'My weather whiskers need a moment. Please try again soon, meow! 🐾'
    elif 'NO LONGER EXPECTED' in title or 'NO RAIN' in title:
        extra = 'No rain on my whiskers for now, meow! 🐾'
    elif 'EXPECTED TO CLEAR' in title:
        extra = 'The rain may be padding away soon, meow! 🐾'
    elif 'RAIN CLEARED' in title:
        extra = 'The rain has padded away, meow! 🐾'
    elif 'RAIN EXPECTED' in title:
        extra = 'Rain might be padding over—keep your paws dry, meow! 🐾'
    elif 'RAIN DETECTED' in title:
        extra = 'Rain is here—keep those little paws dry, meow! 🐾'
    else:
        return text
    result = f'{text}\n\n{extra}'
    return result if len(result.encode('utf-16-le')) // 2 <= limit else text


class CuteBot(ExtBot):
    """Style at delivery so both direct replies and queued alerts use current preferences."""

    async def send_message(self, chat_id, text, *args, **kwargs):
        if await asyncio.to_thread(cute_enabled, chat_id):
            text = cat_text(text, 4096)
        return await super().send_message(chat_id, text, *args, **kwargs)

    async def send_photo(self, chat_id, photo, caption=None, *args, **kwargs):
        if caption and await asyncio.to_thread(cute_enabled, chat_id):
            caption = cat_text(caption, 1024)
        return await super().send_photo(chat_id, photo, caption, *args, **kwargs)


async def cutemode(update, context):
    if update.effective_chat.type != 'private':
        await update.effective_message.reply_text('Use /cutemode in a private chat with me to change your message style.')
        return
    try:
        enabled = await asyncio.to_thread(toggle_cute, update.effective_chat.id)
    except (OSError, sqlite3.Error):
        logger.exception('Could not save cute mode preference')
        await update.effective_message.reply_text("Couldn't save your message style. Please try again.")
        return
    await update.effective_message.reply_text(
        'Cute mode is on! Send /cutemode again to turn it off.' if enabled
        else 'Cute mode is off. Back to normal messages.'
    )
