"""Opt-in Singapore weather bulletins, with durable per-chat delivery receipts."""
import asyncio
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, Forbidden, RetryAfter

from telegram_code.daily_forecast import _sg_time, fetch_scheduled_forecast


DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'weather_schedule.sqlite3'
SLOTS = {'morning': 5, 'noon': 12, 'night': 22}
DELIVERY_WINDOW = timedelta(minutes=30)
LOGGER = logging.getLogger(__name__)


@contextmanager
def _database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('''CREATE TABLE IF NOT EXISTS weather_preferences (
            chat_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL, enabled_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS weather_deliveries (
            chat_id TEXT NOT NULL, slot TEXT NOT NULL, scheduled_for TEXT NOT NULL,
            status TEXT NOT NULL, next_attempt TEXT,
            PRIMARY KEY(chat_id, slot, scheduled_for))''')
        with conn:
            yield conn


def get_enabled(chat_id):
    with _database() as conn:
        row = conn.execute('SELECT enabled FROM weather_preferences WHERE chat_id=?',
                           (str(chat_id),)).fetchone()
    return bool(row and row['enabled'])


def set_enabled(chat_id, enabled, now=None):
    if not isinstance(enabled, bool):
        raise ValueError('Invalid weather preference')
    now = _sg_time(now).isoformat()
    with _database() as conn:
        # Repeated taps on an already enabled setting keep its original time.
        conn.execute('''INSERT INTO weather_preferences VALUES (?,?,?)
            ON CONFLICT(chat_id) DO UPDATE SET enabled=excluded.enabled,
            enabled_at=CASE WHEN weather_preferences.enabled=excluded.enabled
                THEN weather_preferences.enabled_at ELSE excluded.enabled_at END''',
            (str(chat_id), int(enabled), now))


def due_deliveries(now):
    now = _sg_time(now)
    due = []
    with _database() as conn:
        conn.execute('DELETE FROM weather_deliveries WHERE scheduled_for<?',
                     ((now - timedelta(days=14)).isoformat(),))
        for row in conn.execute('SELECT * FROM weather_preferences WHERE enabled=1'):
            for slot, hour in SLOTS.items():
                scheduled = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if not scheduled <= now < scheduled + DELIVERY_WINDOW:
                    continue
                if row['enabled_at'] > scheduled.isoformat():
                    continue  # Opting in starts at the next scheduled time.
                key = (row['chat_id'], slot, scheduled.isoformat())
                receipt = conn.execute('SELECT status,next_attempt FROM weather_deliveries '
                                       'WHERE chat_id=? AND slot=? AND scheduled_for=?', key).fetchone()
                if receipt and (receipt['status'] != 'pending' or receipt['next_attempt'] > now.isoformat()):
                    continue
                due.append(key)
    return due


def claim_delivery(key, now):
    """Claim before sending; crashes/unknown outcomes must never cause a repeat."""
    now = _sg_time(now)
    scheduled = datetime.fromisoformat(key[2])
    if not scheduled <= now < scheduled + DELIVERY_WINDOW:
        return False
    with _database() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT enabled,enabled_at FROM weather_preferences WHERE chat_id=?',
                           (key[0],)).fetchone()
        if not row or not row['enabled'] or row['enabled_at'] > key[2]:
            return False
        receipt = conn.execute('SELECT status,next_attempt FROM weather_deliveries '
                               'WHERE chat_id=? AND slot=? AND scheduled_for=?', key).fetchone()
        if receipt and (receipt['status'] != 'pending' or receipt['next_attempt'] > now.isoformat()):
            return False
        conn.execute('INSERT OR REPLACE INTO weather_deliveries VALUES (?,?,?,?,NULL)', (*key, 'sending'))
    return True


def finish_delivery(key, status, retry_at=None):
    with _database() as conn:
        conn.execute('UPDATE weather_deliveries SET status=?,next_attempt=? '
                     "WHERE chat_id=? AND slot=? AND scheduled_for=? AND status='sending'",
                     (status, retry_at.isoformat() if retry_at else None, *key))


def format_bulletin(forecast, slot, now):
    now = _sg_time(now)
    target = now.date() + timedelta(days=slot == 'night')
    title = {'morning': "Today's weather", 'noon': 'Midday weather update',
             'night': "Tomorrow's weather"}[slot]
    start, end = forecast['valid_from'], forecast['valid_until']
    return (f'{title} · {target:%a %d %b}\n'
            f"{forecast['outlook']}\n{forecast['low']:g}–{forecast['high']:g}°C\n\n"
            f'Valid {start:%d %b %H:%M}–{end:%d %b %H:%M} SGT\n'
            f"Singapore-wide · NEA/MSS · Issued {forecast['issued_at']:%d %b %H:%M} SGT\n\n"
            'Manage these updates: /weatheralerts')


def _lock(context):
    return context.application.bot_data.setdefault('weather_schedule_lock', asyncio.Lock())


async def _get_forecast(context, slot, now):
    cache = context.application.bot_data.setdefault('scheduled_forecast_cache', {})
    # Only active slots are fetched; all subscribing chats share one request.
    cached = cache.get(slot)
    if cached and cached['date'] == now.date() and now < cached['until']:
        return cached['forecast']
    try:
        forecast = await asyncio.to_thread(fetch_scheduled_forecast, slot, now)
        if forecast is None:
            LOGGER.warning('No suitable forecast for scheduled weather slot %s; retrying within delivery window', slot)
    except (requests.RequestException, ValueError, TypeError) as exc:
        LOGGER.warning('Scheduled weather source unavailable: %s', type(exc).__name__)
        forecast = None
    cache[slot] = {'date': now.date(), 'until': now + timedelta(minutes=5), 'forecast': forecast}
    return forecast


async def send_scheduled_weather(context):
    """Run independently of radar/model health, with a short restart catch-up window."""
    async with _lock(context):
        due = await asyncio.to_thread(due_deliveries, _sg_time())
    for key in due:
        now = _sg_time()
        forecast = await _get_forecast(context, key[1], now)
        if forecast is None:
            continue  # Retry within the window; never invent or send old weather.
        async with _lock(context):
            now = _sg_time()
            if not await asyncio.to_thread(claim_delivery, key, now):
                continue
            try:
                await context.bot.send_message(chat_id=int(key[0]),
                    text=format_bulletin(forecast, key[1], now), disable_web_page_preview=True)
            except RetryAfter as exc:
                seconds = exc.retry_after.total_seconds() if hasattr(exc.retry_after, 'total_seconds') else float(exc.retry_after)
                await asyncio.to_thread(finish_delivery, key, 'pending', _sg_time() + timedelta(seconds=max(1, seconds)))
            except Forbidden:
                await asyncio.to_thread(finish_delivery, key, 'failed')
                await asyncio.to_thread(set_enabled, key[0], False)
            except BadRequest:
                await asyncio.to_thread(finish_delivery, key, 'failed')
                LOGGER.warning('Scheduled weather delivery rejected for slot %s', key[1])
            except Exception as exc:
                await asyncio.to_thread(finish_delivery, key, 'uncertain')
                LOGGER.warning('Scheduled weather outcome uncertain: %s; not resending', type(exc).__name__)
            else:
                await asyncio.to_thread(finish_delivery, key, 'sent')
                LOGGER.info('Scheduled weather bulletin delivered: %s', key[1])
        await asyncio.sleep(0)


def schedule_weather(job_queue):
    # Uses an explicit SGT clock inside the job; host/container timezone is irrelevant.
    job_queue.run_repeating(send_scheduled_weather, interval=60, first=10,
                            name='scheduled-weather', job_kwargs={'max_instances': 1, 'coalesce': True})


def settings_view(enabled):
    text = (f'Daily weather updates: {"On" if enabled else "Off"}\n\n'
            '5 am — Today’s weather\n12 noon — Weather update\n10 pm — Tomorrow’s weather\n\n'
            'All times Singapore time (SGT). One switch controls all three updates for this chat.'
              ' Changes start at the next scheduled time.'
              '\nRain alerts are controlled separately in Alert settings.')
    rows = [[InlineKeyboardButton('Turn daily updates off' if enabled else 'Turn daily updates on',
                                  callback_data=f'weatheralerts:{"off" if enabled else "on"}')]]
    return text, InlineKeyboardMarkup(rows)


async def _can_change(update, context):
    if update.effective_chat.type == 'private':
        return True
    if update.effective_user is None:
        return False
    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
        return member.status in ('creator', 'administrator')
    except Exception:
        LOGGER.warning('Could not verify weather settings administrator')
        return False


async def weather_alerts_command(update, context):
    try:
        enabled = await asyncio.to_thread(get_enabled, update.effective_chat.id)
    except (sqlite3.Error, OSError):
        await update.effective_message.reply_text('Weather settings are unavailable. Please try again shortly.')
        return
    text, markup = settings_view(enabled)
    if update.effective_chat.type != 'private':
        text += '\nOnly group admins can change these settings.'
    await update.effective_message.reply_text(text, reply_markup=markup)


async def weather_alerts_callback(update, context):
    query = update.callback_query
    if not await _can_change(update, context):
        await query.answer('Only group admins can change these settings.', show_alert=False)
        return
    if query.data not in ('weatheralerts:on', 'weatheralerts:off'):
        await query.answer('Please reopen /weatheralerts.', show_alert=False)
        return
    enabled = query.data == 'weatheralerts:on'
    try:
        async with _lock(context):
            await asyncio.to_thread(set_enabled, update.effective_chat.id, enabled)
    except (sqlite3.Error, OSError):
        await query.answer("Couldn't save settings. Please try again.", show_alert=False)
        return
    await query.answer('Weather settings saved.', show_alert=False)
    text, markup = settings_view(enabled)
    try:
        await query.edit_message_text(text, reply_markup=markup)
    except BadRequest:
        # Repeated taps can leave the message unchanged; preference was saved.
        LOGGER.info('Weather settings saved; menu could not be refreshed')
