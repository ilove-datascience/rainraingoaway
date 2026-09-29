"""Chat-bound, time-specific user reports; weather delivery never depends on this store."""
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import math
from numbers import Integral, Real
from pathlib import Path
import re
import secrets
import sqlite3

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'forecast_feedback.sqlite3'
SGT = timezone(timedelta(hours=8))
UTC = timezone.utc
REPORT_WINDOW = timedelta(minutes=30)
FEEDBACK_PROMPT = ('Feedback numbers follow location order above. Report at the button’s SGT time '
                   'only if you were there; closes after 30 min.')
CALLBACK_PATTERN = r'^rainfb:[A-Za-z0-9_-]{16}:(?:dry|wet)$'


class FeedbackRejected(ValueError):
    """An expected rejection safe to show to the reporter."""


def _utc_now():
    return datetime.now(UTC)


def _integer(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'Invalid {name}')
    value = int(value)
    if abs(value) > 2**63-1 or (minimum is not None and value < minimum):
        raise ValueError(f'Invalid {name}')
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'Invalid {name}')
    return float(value)


def _time(value):
    if not isinstance(value, datetime):
        raise ValueError('Snapshot times must be datetimes')
    # Existing radar filenames and database datetimes use Singapore local time.
    return (value.replace(tzinfo=SGT) if value.tzinfo is None else value).astimezone(UTC)


def _snapshot(value, now):
    if not isinstance(value, dict) or value.get('kind') not in ('forecast', 'radar'):
        raise ValueError('Invalid snapshot kind')
    latitude = _number(value.get('latitude'), 'latitude')
    longitude = _number(value.get('longitude'), 'longitude')
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError('Invalid snapshot coordinates')
    label = value.get('label')
    if not isinstance(label, str) or not label.strip() or len(label) > 80:
        raise ValueError('Invalid snapshot label')
    label = ' '.join(label.split())
    observed, target = _time(value.get('observed_at')), _time(value.get('target_at'))
    if observed > now or target < observed:
        raise ValueError('Invalid snapshot time order')
    if value['kind'] == 'radar' and target != observed:
        raise ValueError('Radar target must match its observation time')
    if now >= target + REPORT_WINDOW:
        raise FeedbackRejected('This report window has closed. Please use a recent weather update.')
    result = dict(kind=value['kind'], latitude=latitude, longitude=longitude,
                  label=label, observed_at=observed.isoformat(), target_at=target.isoformat())
    for key in ('location_id', 'settings_version'):
        result[key] = None if value.get(key) is None else _integer(value[key], key, minimum=0)
    for key in ('forecast_value', 'radar_value', 'radius_m'):
        result[key] = None if value.get(key) is None else _number(value[key], key)
    if result['radius_m'] is not None and result['radius_m'] < 0:
        raise ValueError('Invalid radius')
    source = value.get('source')
    if source is not None and (not isinstance(source, str) or len(source) > 512):
        raise ValueError('Invalid snapshot source')
    result['source'] = source
    # Presentation order can have gaps when a map includes an ineligible location.
    index = value.get('display_index')
    result['display_index'] = None if index is None else _integer(index, 'display index', minimum=1)
    if result['display_index'] is not None and result['display_index'] > 6:
        raise ValueError('Invalid display index')
    return result


def _connect():
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=1)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS feedback_snapshots (
                token TEXT PRIMARY KEY, chat_id INTEGER NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('forecast','radar')),
                latitude REAL NOT NULL, longitude REAL NOT NULL, label TEXT NOT NULL,
                location_id INTEGER, settings_version INTEGER,
                observed_at TEXT NOT NULL, target_at TEXT NOT NULL,
                forecast_value REAL, radar_value REAL, radius_m REAL, source TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback_reports (
                token TEXT NOT NULL REFERENCES feedback_snapshots(token),
                user_id INTEGER NOT NULL, raining INTEGER NOT NULL CHECK(raining IN (0,1)),
                message_id INTEGER NOT NULL,
                first_reported_at TEXT NOT NULL, reported_at TEXT NOT NULL,
                PRIMARY KEY(token,user_id)
            );
        ''')
        return connection
    except Exception:
        connection.close()
        raise


def _store_snapshots(chat_id, snapshots, now):
    connection = _connect()
    prepared = []
    try:
        with connection:
            for snapshot in snapshots:
                token = secrets.token_urlsafe(12)
                row = dict(snapshot, token=token, chat_id=chat_id, created_at=now.isoformat())
                connection.execute('''INSERT INTO feedback_snapshots
                    (token,chat_id,kind,latitude,longitude,label,location_id,settings_version,
                     observed_at,target_at,forecast_value,radar_value,radius_m,source,created_at)
                    VALUES (:token,:chat_id,:kind,:latitude,:longitude,:label,:location_id,:settings_version,
                            :observed_at,:target_at,:forecast_value,:radar_value,:radius_m,:source,:created_at)''', row)
                prepared.append((token, snapshot))
        return prepared
    finally:
        connection.close()


async def prepare_feedback(chat_id, snapshots):
    """Persist up to six immutable claims and return their time/place-specific buttons.

    Invalid or unavailable feedback must never prevent the original weather send.
    Snapshot times without a timezone follow the bot's existing SGT convention.
    """
    try:
        chat_id = _integer(chat_id, 'chat ID')
        if chat_id == 0 or not isinstance(snapshots, (list, tuple)) or not 1 <= len(snapshots) <= 6:
            raise ValueError('One to six snapshots and a nonzero chat ID required')
        now = _utc_now()
        normalized = [_snapshot(item, now) for item in snapshots]
        prepared = await asyncio.to_thread(_store_snapshots, chat_id, normalized, now)
        rows = []
        for index, (token, snapshot) in enumerate(prepared, 1):
            target = datetime.fromisoformat(snapshot['target_at']).astimezone(SGT)
            label = snapshot['label']
            short = label if len(label) <= 18 else label[:17] + '…'
            number = snapshot.get('display_index') or index
            place_time = f'{number}. {snapshot["kind"].title()} {target:%H:%M} · {short}'
            # Full-width buttons keep the location/time readable on phones.
            rows.append([InlineKeyboardButton(f'{place_time}: Not raining', callback_data=f'rainfb:{token}:dry')])
            rows.append([InlineKeyboardButton(f'{place_time}: Raining', callback_data=f'rainfb:{token}:wet')])
        return InlineKeyboardMarkup(rows)
    except Exception as exc:
        # Avoid logging tokens, coordinates, labels or reporter identities.
        logger.warning('Weather feedback could not be prepared (%s)', type(exc).__name__)
        return None


def _record_report(token, chat_id, user_id, message_id, raining, now):
    connection = _connect()
    try:
        with connection:
            row = connection.execute('SELECT * FROM feedback_snapshots WHERE token=?', (token,)).fetchone()
            if row is None or row['chat_id'] != chat_id:
                raise FeedbackRejected('This button does not belong to this chat or is no longer available.')
            snapshot = dict(row)
            for key in ('observed_at', 'target_at'):
                snapshot[key] = datetime.fromisoformat(snapshot[key])
            # Validate persisted facts too, including the age of an actual-radar claim.
            _snapshot(snapshot, now)
            target = snapshot['target_at']
            if now < target:
                local = target.astimezone(SGT)
                raise FeedbackRejected(f'This forecast is for {local:%H:%M} SGT. Please report then, only if you are there.')
            connection.execute('''INSERT INTO feedback_reports
                (token,user_id,raining,message_id,first_reported_at,reported_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(token,user_id) DO UPDATE SET
                    raining=excluded.raining, message_id=excluded.message_id,
                    reported_at=excluded.reported_at''',
                (token,user_id,int(raining),message_id,now.isoformat(),now.isoformat()))
        return snapshot
    finally:
        connection.close()


async def _answer(query, text=None):
    try:
        await query.answer(text[:200] if text else None, show_alert=False)
    except Exception as exc:
        logger.warning('Weather feedback acknowledgement failed (%s)', type(exc).__name__)


async def handle_feedback(update, context):
    """Save an individual member's report without editing a group's shared message."""
    query = getattr(update, 'callback_query', None)
    if query is None:
        return
    try:
        data = query.data
        if not isinstance(data, str) or re.fullmatch(CALLBACK_PATTERN, data) is None:
            raise FeedbackRejected('This feedback button is invalid. Please use a recent weather update.')
        user = getattr(query, 'from_user', None)
        if user is None or getattr(user, 'is_bot', False):
            raise FeedbackRejected('A signed-in person must submit this report.')
        user_id = _integer(user.id, 'reporter ID', minimum=1)
        effective_user = getattr(update, 'effective_user', None)
        if effective_user is not None and effective_user.id != user_id:
            raise FeedbackRejected('The reporter could not be verified.')
        message = getattr(query, 'message', None)
        chat = getattr(message, 'chat', None)
        effective_chat = getattr(update, 'effective_chat', None)
        if chat is None or effective_chat is None or chat.id != effective_chat.id:
            raise FeedbackRejected('Please use this button in the chat where the weather update was sent.')
        chat_id = _integer(chat.id, 'chat ID')
        message_id = _integer(message.message_id, 'message ID', minimum=1)
        _, token, condition = data.split(':')
        snapshot = await asyncio.to_thread(_record_report, token, chat_id, user_id, message_id,
                                          condition == 'wet', _utc_now())
    except FeedbackRejected as exc:
        await _answer(query, str(exc))
        return
    except (AttributeError, ValueError, TypeError):
        await _answer(query, 'This feedback button is invalid. Your report was not saved.')
        return
    except Exception as exc:
        logger.warning('Weather feedback could not be saved (%s)', type(exc).__name__)
        await _answer(query, 'Sorry, your report could not be saved. Please try again shortly.')
        return
    # Telegram shows a brief notice only to the reporter, without an OK dialog.
    condition_text = 'raining' if condition == 'wet' else 'not raining'
    await _answer(query, f'Saved: {snapshot["label"]} — {condition_text}. Thanks!')
