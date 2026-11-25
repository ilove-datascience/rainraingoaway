"""Run subscription transactions against isolated in-memory tables, never live MySQL."""
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from telegram_code import location_alerts, group_locations, notification_delivery, rain_state_db


class Cursor:
    def __init__(self, conn, dictionary=False):
        self.cursor = conn.cursor()
        self.dictionary = dictionary

    def execute(self, sql, params=()):
        self.cursor.execute(sql.replace('%s', '?').replace(' FOR UPDATE', ''), params)

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is not None and self.dictionary:
            return dict(zip([item[0] for item in self.cursor.description], row))
        return row

    def fetchall(self):
        rows = []
        while (row := self.fetchone()) is not None:
            rows.append(row)
        return rows

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def close(self):
        self.cursor.close()


class Connection:
    def __init__(self, conn):
        self.conn = conn

    def cursor(self, dictionary=False):
        return Cursor(self.conn, dictionary)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        pass


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.executescript('''
        CREATE TABLE users(userid INTEGER PRIMARY KEY, mode TEXT);
        CREATE TABLE user_location(userid INTEGER,location_id INTEGER,label TEXT,latitude REAL,longitude REAL,
            alerts_enabled INTEGER NOT NULL DEFAULT 1,rain_settings_version INTEGER NOT NULL DEFAULT 0,
            rain_episode_reason TEXT,rain_alert_reason TEXT,rain_dry_since TIMESTAMP,rain_episode_wet INTEGER DEFAULT 0,state TEXT,rain_observed_at TIMESTAMP,
            rain_forecast_at TIMESTAMP,radar_raining INTEGER,rain_forecast_value REAL,
            PRIMARY KEY(userid,location_id));
        CREATE TABLE rain_notifications(id INTEGER PRIMARY KEY,userid INTEGER,location_id INTEGER,
            settings_version INTEGER,observed_at TIMESTAMP,forecast_at TIMESTAMP,reason TEXT,message TEXT,
            status TEXT DEFAULT 'pending',available_at TIMESTAMP,photo BLOB);
        INSERT INTO users VALUES(42,'automatic'),(-1001,'automatic');
        INSERT INTO user_location(userid,location_id,label,latitude,longitude) VALUES
            (42,0,'Main',1.3,103.8),(42,7,'Office',1.35,103.9),(-1001,0,'Main',1.3,103.8);
    ''')
    proxy = Connection(conn)
    for module in (location_alerts, group_locations, notification_delivery, rain_state_db):
        monkeypatch.setattr(module, '_get_db_connection', lambda: proxy)
    yield conn
    conn.close()


def test_default_on_and_location_off_is_isolated_and_idempotent(db):
    assert group_locations.add_named_location(42, 'School', 1.3, 103.8)
    assert all(row['alerts_enabled'] for row in group_locations.list_locations(42))
    assert location_alerts.set_location_alert(42, 7, False)
    assert location_alerts.set_location_alert(42, 7, False)
    assert db.execute('SELECT alerts_enabled,rain_settings_version FROM user_location WHERE userid=42 AND location_id=7').fetchone() == (0, 1)
    assert db.execute('SELECT alerts_enabled FROM user_location WHERE userid=-1001').fetchone() == (1,)
    assert db.execute('SELECT alerts_enabled FROM user_location WHERE userid=42 AND location_id=0').fetchone() == (1,)


def test_new_location_after_global_pause_only_enables_new_location(db):
    db.execute("UPDATE users SET mode='manual' WHERE userid=42")
    assert group_locations.add_named_location(42, 'School', 1.3, 103.8)
    rows = group_locations.list_locations(42)
    assert {row['label']: row['alerts_enabled'] for row in rows} == {'Main': 0, 'Office': 0, 'School': 1}
    assert db.execute('SELECT mode FROM users WHERE userid=42').fetchone() == ('automatic',)


def test_select_after_pause_does_not_resume_other_locations(db):
    db.execute("UPDATE users SET mode='manual' WHERE userid=42")
    assert location_alerts.set_location_alert(42, 7, True)
    assert {row['label']: row['alerts_enabled'] for row in group_locations.list_locations(42)} == {'Main': 0, 'Office': 1}


def test_foreign_or_deleted_location_cannot_change_chat_mode(db):
    db.execute("UPDATE users SET mode='manual' WHERE userid=-1001")
    db.commit()
    assert not location_alerts.set_location_alert(-1001, 7, True)
    assert db.execute('SELECT mode FROM users WHERE userid=-1001').fetchone() == ('manual',)


def queue(db, location=7):
    now = datetime(2026, 11, 25, 12)
    db.execute('''INSERT INTO rain_notifications(userid,location_id,settings_version,observed_at,
        forecast_at,reason,message,available_at) VALUES(42,?,0,?,?,'expected','rain',?)''',
               (location, now, now + timedelta(minutes=5), now))
    db.commit()
    return now


def test_switch_off_cancels_pending_alerts_and_claim_rechecks_selection(db):
    now = queue(db)
    location_alerts.set_location_alert(42, 7, False)
    assert notification_delivery.claim_notification(now) is None
    assert db.execute('SELECT status FROM rain_notifications').fetchone() == ('cancelled',)
    # Even a pending item whose version matches must not bypass a disabled location.
    db.execute("UPDATE rain_notifications SET status='pending',settings_version=1")
    db.commit()
    assert notification_delivery.claim_notification(now) == {'cancelled': True}


def test_turn_off_all_keeps_locations_and_other_chats(db):
    queue(db)
    assert location_alerts.turn_off_all(42)
    assert len(group_locations.list_locations(42)) == 2
    assert not any(row['alerts_enabled'] for row in group_locations.list_locations(42))
    assert group_locations.list_locations(-1001)[0]['alerts_enabled']


def test_disabled_location_cannot_enqueue_and_old_snapshot_is_rejected(db):
    row = next(row for row in rain_state_db.get_rain_locations() if row['userid'] == 42 and row['location_id'] == 7)
    now = datetime(2026, 11, 25, 12)
    result = dict(state='predicted', rain_observed_at=now, rain_forecast_at=now+timedelta(minutes=5),
                  radar_raining=False, rain_forecast_value=.8, reason='expected')
    location_alerts.set_location_alert(42, 7, False)
    assert not rain_state_db.save_rain_state(row, result, 'rain', now)
    disabled = next(row for row in rain_state_db.get_rain_locations() if row['userid'] == 42 and row['location_id'] == 7)
    assert rain_state_db.save_rain_state(disabled, result, 'rain', now)
    assert db.execute('SELECT COUNT(*) FROM rain_notifications').fetchone()[0] == 0


def test_stale_removal_confirmation_does_not_remove_replacement(db):
    assert not group_locations.remove_named_location(42, 'Office', expected_location_id=99)
    assert len(group_locations.list_locations(42)) == 2
