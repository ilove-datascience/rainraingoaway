"""Hidden, persistent cat-style messages for private bot chats."""
import asyncio
import logging
import random
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
    """Add a randomly chosen cat line without changing facts or entity offsets."""
    if not text:
        return text
    title = text.split('\n', 1)[0]
    if 'DELAYED' in title or 'UNAVAILABLE' in title:
        lines = ('My weather whiskers need a moment. Please try again soon, meow! 🐾',
                 'One moment, please—my radar whiskers are untangling. 🐱',
                 'A tiny weather hiccup. Please check back soon, meow. 🐾')
    elif 'NO LONGER EXPECTED' in title or 'NO RAIN' in title:
        lines = ('No rain on my whiskers for now, meow! 🐾',
                 'No rain expected for now. A little window-watching break, perhaps? 🐈',
                 'The forecast says no rain for now. Purr-fect for a whisker break. 🐱')
    elif 'EXPECTED TO CLEAR' in title:
        lines = ('The rain may be padding away soon, meow! 🐾',
                 'The rain might be packing up its tiny suitcase. 🧳🐈',
                 'A break in the rain may be coming. Whiskers crossed! 🐱')
    elif 'RAIN CLEARED' in title:
        lines = ('The rain has padded away, meow! 🐾',
                 'Radar says the rain has cleared. Cue the little cat stretch. 🐈',
                 'Rain cleared! Time to peek out from under the umbrella. 🐱☂️')
    elif 'RAIN EXPECTED' in title:
        lines = ('Rain might be padding over—keep your paws dry, meow! 🐾',
                 'Possible rain incoming. Your umbrella has been summoned, meow! ☂️🐱',
                 'Rain may visit soon. I suggest the cosy side of the window. 🐈',
                 'A little heads-up from your weather cat: umbrella at the ready! 🐾☂️')
    elif 'RAIN DETECTED' in title:
        lines = ('Rain is here—keep those little paws dry, meow! 🐾',
                 'Radar spotted rain. Deploy the umbrella, human! 🐱☂️',
                 'Wet-paw weather detected. A cosy shelter sounds nice. 🐈')
    elif title.startswith('Current alert setting:'):
        lines = ('A peek at your weather-cat preferences. 🐱',
                 'Here is how your little forecast assistant is set up. 🐾',
                 'Checking the cat control panel… 🐈')
    elif title.startswith('Select mode:'):
        lines = ('How would you like your weather served, human? 🐱',
                 'Your paws, your choice. Pick an option below! 🐾',
                 'Choose my assignment, meow! 🐈')
    elif title.startswith('Automatic alerts enabled:'):
        lines = ('Reporting for weather duty, meow! 🫡🐱',
                 'Tiny paws, important weather business. 🐾',
                 'Your weather cat has clocked in. 🐈')
    elif title.startswith('Automatic alerts paused.'):
        lines = ('Taking a little catnap. Ask for a forecast whenever you like. 💤🐱',
                 'On-demand weather it is, meow! 🐾',
                 'I will be by the window when you need me. 🐈')
    else:
        return text
    extra = random.choice(lines)
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
