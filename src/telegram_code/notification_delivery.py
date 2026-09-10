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
                JOIN user_location l ON l.userid=u.userid WHERE u.userid=%s''', (item['userid'],))
            user = cur.fetchone()
            expired = item['reason'] != ENDED and now >= item['forecast_at']
            eligible = user and user['mode'] == 'automatic' and user['rain_settings_version'] == item['settings_version']
            status = 'sending' if eligible and not expired else 'cancelled'
            cur.execute('UPDATE rain_notifications SET status=%s WHERE id=%s', (status,item['id']))
            if expired and eligible:
                # Expired, undelivered start must not suppress a later fresh start.
                cur.execute('''UPDATE user_location SET rain_episode_reason=rain_alert_reason
                    WHERE userid=%s AND rain_settings_version=%s AND rain_episode_reason=%s''',
                    (item['userid'],item['settings_version'],item['reason']))
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
                    WHERE userid=%s AND rain_settings_version=%s''',
                    (now,item['reason'],item['userid'],item['settings_version']))
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


async def deliver_notifications(context, reply_markup=None):
    # A receipt whose DB acknowledgement failed is retried without sending again.
    receipts = context.application.bot_data.setdefault('notification_receipts', {})
    for key, receipt in list(receipts.items()):
        await asyncio.to_thread(finish_notification, *receipt)
        receipts.pop(key, None)
    for _ in range(100):
        # Release between recipients so a mode/location change can take effect promptly.
        async with settings_lock(context):
            item = await asyncio.to_thread(claim_notification, sg_now())
            if item is None:
                return
            if item.get('cancelled'):
                continue
            now = sg_now()
            try:
                if item.get('photo'):
                    # The persisted PNG belongs to this event, not the latest model run.
                    with BytesIO(item['photo']) as photo:
                        photo.name = 'radar.png'
                        sent = await context.bot.send_photo(chat_id=item['userid'], photo=photo,
                                                            caption=item['message'], reply_markup=reply_markup)
                else:
                    # Backward compatibility for text alerts queued before this migration.
                    sent = await context.bot.send_message(chat_id=item['userid'], text=item['message'], reply_markup=reply_markup)
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
            await asyncio.to_thread(finish_notification, *receipt)
            receipts.pop(item['id'], None)
        await asyncio.sleep(0)
