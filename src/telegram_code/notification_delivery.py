"""Durable alert delivery. Ambiguous Telegram outcomes are never blindly retried."""
import asyncio
import json
from datetime import timedelta
from io import BytesIO
from telegram.error import RetryAfter, Forbidden, BadRequest
from telegram_code.database import _get_db_connection
from telegram_code.forecast_policy import sg_now
from telegram_code.rain_state import ENDED, OBSERVED, CANCELLED, ENDING
from telegram_code.notification_text import compose_notice
from telegram_code.feedback_context import notification_feedback, with_feedback_prompt


def settings_lock(context):
    return context.application.bot_data.setdefault('notification_settings_lock', asyncio.Lock())


def claim_notifications(now, limit=100):
    """Claim one chat's current events together; discard stale/superseded changes."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("SELECT userid FROM rain_notifications WHERE status='pending' AND available_at<=%s ORDER BY id LIMIT 1 FOR UPDATE", (now,))
            first = cur.fetchone()
            if not first:
                return None
            cur.execute("""SELECT * FROM rain_notifications WHERE userid=%s AND status='pending'
                AND available_at<=%s ORDER BY id LIMIT %s FOR UPDATE""", (first['userid'], now, limit))
            candidates = cur.fetchall()
            day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            cur.execute("""SELECT id FROM rain_notifications WHERE userid=%s
                AND status IN ('sent','sending','uncertain','abandoned') AND available_at>=%s AND available_at<%s LIMIT 1""",
                (first['userid'], day, day + timedelta(days=1)))
            first_today = cur.fetchone() is None
            cur.execute('SELECT location_id,rain_settings_version FROM user_location WHERE userid=%s', (first['userid'],))
            photo_scope = sorted([row['location_id'], row['rain_settings_version']] for row in cur.fetchall())
            items = []
            for item in candidates:
                cur.execute('''SELECT u.mode,l.rain_settings_version,l.label FROM users u
                    JOIN user_location l ON l.userid=u.userid WHERE u.userid=%s AND l.location_id=%s''',
                    (item['userid'], item.get('location_id', 0)))
                user = cur.fetchone()
                cur.execute('''SELECT id FROM rain_notifications WHERE userid=%s AND location_id=%s
                    AND settings_version=%s AND id>%s LIMIT 1''',
                    (item['userid'], item.get('location_id', 0), item['settings_version'], item['id']))
                superseded = cur.fetchone() is not None
                actual = item['reason'] in (ENDED, OBSERVED, CANCELLED)
                deadline = item['observed_at'] + timedelta(minutes=10) if actual else item['forecast_at']
                expired = not item['observed_at'] <= now < deadline
                eligible = user and user['mode'] == 'automatic' and user['rain_settings_version'] == item['settings_version']
                status = 'sending' if eligible and not expired and not superseded and item['reason'] != ENDING else 'cancelled'
                cur.execute('UPDATE rain_notifications SET status=%s WHERE id=%s', (status, item['id']))
                if expired and eligible and not superseded:
                    release_episode(cur, item)
                if status == 'sending':
                    item['label'] = user['label']
                    item['first_today'] = first_today
                    try:
                        payload = json.loads(item['message'])
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict) and payload.get('rain_alert') == 1:
                        if payload.get('photo_locations') != photo_scope:
                            item['photo'] = None  # Saved markers changed while this event waited.
                    items.append(item)
            conn.commit()
            return items or [{'cancelled': True}]
        finally:
            cur.close()
    finally:
        conn.close()


def claim_notification(now):
    """Single-event compatibility for maintenance callers."""
    items = claim_notifications(now, limit=1)
    return items[0] if items else None


def finish_notification(item, status, now, message_id=None, retry_seconds=0):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('''UPDATE rain_notifications SET status=%s,telegram_message_id=%s,available_at=%s
                WHERE id=%s AND status='sending' ''',
                (status,message_id,now+timedelta(seconds=retry_seconds),item['id']))
            changed = cur.rowcount > 0
            if changed and status == 'sent':
                cur.execute('''UPDATE user_location SET rain_alert_at=%s,rain_alert_reason=%s
                    WHERE userid=%s AND location_id=%s AND rain_settings_version=%s
                    AND (rain_alert_at IS NULL OR rain_alert_at<=%s)''',
                    (now,item['reason'],item['userid'],item.get('location_id', 0),item['settings_version'],now))
            elif changed and status == 'failed':
                release_episode(cur, item)
            conn.commit()
            print(f"Notification {item['id']} delivery status: {status}")
        finally:
            cur.close()
    finally:
        conn.close()


def release_episode(cur, item):
    if item['reason'] in (ENDED, CANCELLED):
        return  # A dropped clear must not erase the persisted forecast-rearm timer.
    # Do not undo a newer queued/sent event or a user's settings change.
    cur.execute("""UPDATE user_location SET rain_episode_reason=NULL, rain_alert_reason=NULL
        WHERE userid=%s AND location_id=%s AND rain_settings_version=%s
        AND rain_episode_reason=%s AND NOT EXISTS (
            SELECT 1 FROM rain_notifications n WHERE n.userid=%s AND n.location_id=%s
            AND n.settings_version=%s AND n.id>%s)""",
        (item['userid'], item.get('location_id', 0), item['settings_version'], item['reason'],
         item['userid'], item.get('location_id', 0), item['settings_version'], item['id']))


def recover_abandoned_notifications(now, protected_ids=()):
    """Never resend an ambiguous event; allow new observations after a grace period."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("""SELECT * FROM rain_notifications
                WHERE status IN ('sending','uncertain','failed') AND available_at<%s
                ORDER BY id FOR UPDATE""", (now - timedelta(minutes=15),))
            for item in cur.fetchall():
                if item['id'] in protected_ids:
                    continue  # Receipt is known; retry acknowledgement only.
                cur.execute("UPDATE rain_notifications SET status='abandoned' WHERE id=%s", (item['id'],))
                release_episode(cur, item)
                print(f"Notification {item['id']} abandoned after uncertain delivery; original will not be resent")
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


async def deliver_notifications(context, reply_markup=None):
    lock = context.application.bot_data.setdefault('notification_delivery_lock', asyncio.Lock())
    async with lock:
        await _deliver_notifications(context, reply_markup)


async def _deliver_notifications(context, reply_markup=None):
    # A receipt whose DB acknowledgement failed is retried without sending again.
    receipts = context.application.bot_data.setdefault('notification_receipts', {})
    for key, receipt in list(receipts.items()):
        try:
            await asyncio.to_thread(finish_notification, *receipt)
            receipts.pop(key, None)
        except Exception as exc:
            print(f'Notification {key} receipt retry failed: {type(exc).__name__}')
    await asyncio.to_thread(recover_abandoned_notifications, sg_now(), tuple(receipts))
    for _ in range(100):
        # Release between recipients so a mode/location change can take effect promptly.
        async with settings_lock(context):
            state_lock = context.application.bot_data.setdefault('rain_state_lock', asyncio.Lock())
            async with state_lock:
                items = await asyncio.to_thread(claim_notifications, sg_now())
            if items is None:
                return
            if items[0].get('cancelled'):
                continue
            item = items[0]
            now = sg_now()
            markup = reply_markup(item['userid']) if callable(reply_markup) else reply_markup
            message = compose_notice(items)
            try:
                feedback = await notification_feedback(item['userid'], items)
            except Exception as exc:
                print(f'Optional weather feedback unavailable: {type(exc).__name__}')
                feedback = None
            if feedback is not None:
                markup = feedback
                message = with_feedback_prompt(message, feedback)
            # A single shared observation map contains all saved location markers.
            # Legacy/mixed timestamps use one text notice rather than an unrelated map.
            photo_bytes = item.get('photo')
            same_photo = photo_bytes and all(other.get('photo') == photo_bytes for other in items)
            use_photo = same_photo and len(message.encode('utf-16-le')) // 2 <= 1024
            try:
                # Optional storage may have taken long enough for a forecast to
                # expire. Requeue the batch so normal claiming cancels expired
                # events and retains still-current locations before any send.
                send_at = sg_now()
                expired = any(not event['observed_at'] <= send_at < (
                    event['observed_at'] + timedelta(minutes=10)
                    if event['reason'] in (ENDED, OBSERVED, CANCELLED) else event['forecast_at'])
                    for event in items)
                if expired:
                    outcome = ('pending', send_at, None, 0)
                elif use_photo:
                    with BytesIO(photo_bytes) as photo:
                        photo.name = 'radar.png'
                        sent = await context.bot.send_photo(chat_id=item['userid'], photo=photo,
                                                            caption=message, reply_markup=markup)
                    outcome = ('sent', sg_now(), sent.message_id, 0)
                else:
                    sent = await context.bot.send_message(chat_id=item['userid'], text=message, reply_markup=markup)
                    outcome = ('sent', sg_now(), sent.message_id, 0)
            except RetryAfter as exc:
                delay = exc.retry_after.total_seconds() if hasattr(exc.retry_after, 'total_seconds') else float(exc.retry_after)
                outcome = ('pending', now, None, max(1, delay))
            except (Forbidden, BadRequest):
                outcome = ('failed', now, None, 0)
            except Exception as exc:
                print(f"Notification batch {item['id']} delivery uncertain; not resending: {type(exc).__name__}")
                outcome = ('uncertain', now, None, 0)
            # Protect every constituent event before any DB acknowledgement can fail.
            for event in items:
                receipts[event['id']] = (event, *outcome)
            for event in items:
                try:
                    await asyncio.to_thread(finish_notification, *receipts[event['id']])
                    receipts.pop(event['id'], None)
                except Exception as exc:
                    print(f"Notification {event['id']} receipt save failed: {type(exc).__name__}")
        await asyncio.sleep(0)
