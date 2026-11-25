"""Explicit per-location subscriptions, serialized with notification delivery."""
from telegram_code.database import _get_db_connection


def set_location_alert(chat_id, location_id, enabled):
    """Set a desired value, so duplicate button deliveries cannot toggle it twice."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SELECT mode FROM users WHERE userid=%s FOR UPDATE', (chat_id,))
            user = cur.fetchone()
            if not user:
                return False
            cur.execute('SELECT alerts_enabled FROM user_location WHERE userid=%s AND location_id=%s FOR UPDATE',
                        (chat_id, location_id))
            row = cur.fetchone()
            if row is None:
                return False
            # Enabling one location after a legacy global pause must not resume others.
            if enabled and user[0] != 'automatic':
                cur.execute('''UPDATE user_location SET alerts_enabled=0,
                    rain_settings_version=rain_settings_version+1,
                    rain_episode_reason=NULL,rain_alert_reason=NULL,rain_dry_since=NULL,rain_episode_wet=0 WHERE userid=%s''', (chat_id,))
                cur.execute("UPDATE rain_notifications SET status='cancelled' WHERE userid=%s AND status='pending'", (chat_id,))
                cur.execute("UPDATE users SET mode='automatic' WHERE userid=%s", (chat_id,))
                row = (False,)
            if bool(row[0]) != bool(enabled):
                cur.execute('''UPDATE user_location SET alerts_enabled=%s,
                    rain_settings_version=rain_settings_version+1,
                    rain_episode_reason=NULL,rain_alert_reason=NULL,rain_dry_since=NULL,rain_episode_wet=0 WHERE userid=%s AND location_id=%s''',
                            (bool(enabled), chat_id, location_id))
                cur.execute("UPDATE rain_notifications SET status='cancelled' WHERE userid=%s AND location_id=%s AND status='pending'",
                            (chat_id, location_id))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    finally:
        conn.close()


def turn_off_all(chat_id):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SELECT mode FROM users WHERE userid=%s FOR UPDATE', (chat_id,))
            if not cur.fetchone():
                return False
            cur.execute('''UPDATE user_location SET alerts_enabled=0,
                rain_settings_version=rain_settings_version+1,
                rain_episode_reason=NULL,rain_alert_reason=NULL,rain_dry_since=NULL,rain_episode_wet=0
                WHERE userid=%s AND alerts_enabled<>0''', (chat_id,))
            cur.execute("UPDATE rain_notifications SET status='cancelled' WHERE userid=%s AND status='pending'", (chat_id,))
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    finally:
        conn.close()
