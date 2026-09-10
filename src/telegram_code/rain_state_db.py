"""Additive schema migration and guarded location-state persistence."""
from telegram_code.database import _get_db_connection


def ensure_rain_state_schema():
    columns = {
        'state': "ENUM('predicted','confirmed','predicted norain') NOT NULL DEFAULT 'predicted norain'",
        'rain_observed_at': 'DATETIME NULL',
        'rain_forecast_at': 'DATETIME NULL',
        'radar_raining': 'BOOLEAN NULL',
        'rain_alert_at': 'DATETIME NULL',
        'rain_alert_reason': 'VARCHAR(160) NULL',
        'rain_forecast_value': 'DOUBLE NULL',
        'rain_alert_value': 'DOUBLE NULL',
        'rain_episode_reason': 'VARCHAR(160) NULL',
        'rain_settings_version': 'BIGINT NOT NULL DEFAULT 0',
    }
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SHOW COLUMNS FROM user_location')
            existing = {row[0] for row in cur.fetchall()}
            additions = [f'ADD COLUMN `{name}` {definition}' for name, definition in columns.items() if name not in existing]
            if additions:
                cur.execute('ALTER TABLE user_location ' + ', '.join(additions))
            cur.execute('''CREATE TABLE IF NOT EXISTS rain_notifications (
                id BIGINT AUTO_INCREMENT PRIMARY KEY, userid BIGINT NOT NULL,
                settings_version BIGINT NOT NULL, observed_at DATETIME NOT NULL,
                forecast_at DATETIME NOT NULL, reason VARCHAR(160) NOT NULL,
                message TEXT NOT NULL, status VARCHAR(20) NOT NULL DEFAULT 'pending',
                available_at DATETIME NOT NULL, telegram_message_id BIGINT NULL,
                UNIQUE KEY event_key(userid, settings_version, observed_at, reason),
                INDEX pending_idx(status, available_at)
            )''')
            cur.execute('SHOW COLUMNS FROM rain_notifications')
            if 'photo' not in {row[0] for row in cur.fetchall()}:
                cur.execute('ALTER TABLE rain_notifications ADD COLUMN photo MEDIUMBLOB NULL')
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def get_rain_locations():
    conn = _get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute('SELECT l.*, u.mode FROM user_location l JOIN users u ON u.userid = l.userid')
            return cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()


def save_rain_state(row, result, message=None, now=None, photo=None):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            # Lock settings with the same order used by mode changes.
            cur.execute('SELECT mode FROM users WHERE userid=%s FOR UPDATE', (row['userid'],))
            user = cur.fetchone()
            if not user:
                return False
            cur.execute('''UPDATE user_location SET state=%s, rain_observed_at=%s,
                rain_forecast_at=%s, radar_raining=%s, rain_forecast_value=%s
                WHERE userid=%s AND latitude=%s AND longitude=%s
                AND rain_settings_version=%s
                AND (rain_observed_at IS NULL OR rain_observed_at < %s)''',
                (result['state'], result['rain_observed_at'], result['rain_forecast_at'],
                 result['radar_raining'], result['rain_forecast_value'], row['userid'], row['latitude'], row['longitude'], row['rain_settings_version'], result['rain_observed_at']))
            changed = cur.rowcount > 0
            if changed and message and user[0] == 'automatic':
                cur.execute('''INSERT INTO rain_notifications
                    (userid,settings_version,observed_at,forecast_at,reason,message,available_at,photo)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (row['userid'], row['rain_settings_version'], result['rain_observed_at'],
                     result['rain_forecast_at'], result['reason'], message, now, photo))
                cur.execute('UPDATE user_location SET rain_episode_reason=%s WHERE userid=%s',
                            (result['reason'], row['userid']))
            conn.commit()
            return changed
        finally:
            cur.close()
    finally:
        conn.close()


def mark_rain_alert(row, result, sent_at):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('''UPDATE user_location SET rain_alert_at=%s, rain_alert_reason=%s, rain_alert_value=%s
                WHERE userid=%s AND latitude=%s AND longitude=%s AND rain_observed_at=%s''',
                (sent_at, result['reason'], result['rain_forecast_value'], row['userid'], row['latitude'], row['longitude'], result['rain_observed_at']))
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


if __name__ == '__main__':
    ensure_rain_state_schema()
    print('Rain-state schema is ready.')
