"""Durable alert delivery. Ambiguous Telegram outcomes are never blindly retried."""
import asyncio
from datetime import timedelta
from io import BytesIO
from telegram.error import RetryAfter, Forbidden, BadRequest
from telegram_code.database import _get_db_connection
from telegram_code.forecast_policy import sg_now
from telegram_code.rain_state import ENDED


def settings_lock(context):
    return context.application.bot_data.setdefault('notification_settings_lock', asyncio.Lock())


def claim_notification(now):
    conn = _get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            # Lock one pending item; transaction commits BEFORE the Telegram request.
            cur.execute("SELECT * FROM rain_notifications WHERE status='pending' AND available_at<=%s ORDER BY id LIMIT 1 FOR UPDATE", (now,))
            item = cur.fetchone()
            if not item:
                return None
            cur.execute('''SELECT u.mode,l.rain_settings_version FROM users u
                JOIN user_location l ON l.userid=u.userid WHERE u.userid=%s AND l.location_id=%s''', (item['userid'], item.get('location_id', 0)))
            user = cur.fetchone()
            expired = item['reason'] != ENDED and now >= item['forecast_at']
            eligible = user and user['mode'] == 'automatic' and user['rain_settings_version'] == item['settings_version']
            status = 'sending' if eligible and not expired else 'cancelled'
            cur.execute('UPDATE rain_notifications SET status=%s WHERE id=%s', (status,item['id']))
            if expired and eligible:
                release_episode(cur, item)
            conn.commit()
            return item if status == 'sending' else {'cancelled': True}
        finally:
            cur.close()
    finally:
        conn.close()


def finish_notification(item, status, now, message_id=None, retry_seconds=0):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('''UPDATE rain_notifications SET status=%s,telegram_message_id=%s,available_at=%s
                WHERE id=%s AND status='sending' ''',
                (status,message_id,now+timedelta(seconds=retry_seconds),item['id']))
            if status == 'sent':
                cur.execute('''UPDATE user_location SET rain_alert_at=%s,rain_alert_reason=%s
                    WHERE userid=%s AND location_id=%s AND rain_settings_version=%s''',
                    (now,item['reason'],item['userid'],item.get('location_id', 0),item['settings_version']))
            elif status == 'failed':
                release_episode(cur, item)
            conn.commit()
            print(f"Notification {item['id']} delivery status: {status}")
        finally:
            cur.close()
    finally:
        conn.close()


def release_episode(cur, item):
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
            item = await asyncio.to_thread(claim_notification, sg_now())
            if item is None:
                return
            if item.get('cancelled'):
                continue
            now = sg_now()
            markup = reply_markup(item['userid']) if callable(reply_markup) else reply_markup
            try:
                if item.get('photo'):
                    # The persisted PNG belongs to this event, not the latest model run.
                    with BytesIO(item['photo']) as photo:
                        photo.name = 'radar.png'
                        sent = await context.bot.send_photo(chat_id=item['userid'], photo=photo,
                                                            caption=item['message'], reply_markup=markup)
                else:
                    # Backward compatibility for text alerts queued before this migration.
                    sent = await context.bot.send_message(chat_id=item['userid'], text=item['message'], reply_markup=markup)
                receipt = (item, 'sent', sg_now(), sent.message_id, 0)
            except RetryAfter as exc:
                delay = exc.retry_after.total_seconds() if hasattr(exc.retry_after, 'total_seconds') else float(exc.retry_after)
                receipt = (item, 'pending', now, None, delay)
            except (Forbidden, BadRequest):
                receipt = (item, 'failed', now, None, 0)
            except Exception as exc:
                print(f"Notification {item['id']} delivery uncertain; not resending: {type(exc).__name__}")
                receipt = (item, 'uncertain', now, None, 0)
            receipts[item['id']] = receipt
            try:
                await asyncio.to_thread(finish_notification, *receipt)
                receipts.pop(item['id'], None)
            except Exception as exc:
                print(f"Notification {item['id']} receipt save failed: {type(exc).__name__}")
        await asyncio.sleep(0)
