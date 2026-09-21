"""Validate mounted assets and prepare the DB before starting live workers."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def check_assets():
    for key in ('tele_api_key',):
        if not os.getenv(key, '').strip():
            raise RuntimeError(f'Missing required environment variable: {key}')
    if not (os.getenv('gov_api_key') or os.getenv('GOV_API_KEY')):
        raise RuntimeError('Missing gov_api_key (or GOV_API_KEY)')
    if not (ROOT / 'models' / 'normalization_stats.json').is_file():
        raise RuntimeError('Mount matching models/normalization_stats.json; inference must not refit it')
    if not list((ROOT / 'models').glob('model_best_*.pkl')):
        raise RuntimeError('Mount a production model_best_*.pkl in models/')
    for directory in ('data/70km/png', 'data/240km/png', 'data/environment', 'logs'):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)


def validate_schema(cur):
    """Read-only readiness checks work with a normal CRUD database account."""
    columns = {
        'users': 'userid, mode, location_flag',
        'user_location': ('userid, latitude, longitude, location_id, label, state, '
                          'rain_observed_at, rain_forecast_at, radar_raining, rain_alert_at, '
                          'rain_alert_reason, rain_forecast_value, rain_alert_value, '
                          'rain_episode_reason, rain_settings_version'),
        'rain_notifications': ('id, userid, location_id, settings_version, observed_at, '
                               'forecast_at, reason, message, status, available_at, '
                               'telegram_message_id, photo'),
    }
    from mysql.connector import Error
    try:
        for table, names in columns.items():
            cur.execute(f'SELECT {names} FROM {table} LIMIT 0')
            cur.fetchall()
        for table, key, expected in (
            ('user_location', 'PRIMARY', ['userid', 'location_id']),
            ('rain_notifications', 'event_key',
             ['userid', 'location_id', 'settings_version', 'observed_at', 'reason']),
        ):
            cur.execute(f"SHOW INDEX FROM {table} WHERE Key_name='{key}'")
            actual = [row[4] for row in sorted(cur.fetchall(), key=lambda row: row[3])]
            if actual != expected:
                raise RuntimeError(f'Schema migration required: {table}.{key}; run --init-db once with a schema administrator')
    except Error as exc:
        raise RuntimeError(
            f'Database schema check failed (MySQL code {exc.errno}). '
            'Check SELECT permissions; if the schema is missing/outdated, run '
            '--init-db once with a schema administrator. Normal startup needs no CREATE/ALTER.'
        ) from None


def prepare_database(*, initialize=False):
    from telegram_code.database import _get_db_connection
    from telegram_code.rain_state_db import ensure_rain_state_schema
    from mysql.connector import Error
    deadline = time.monotonic() + 120
    while True:
        try:
            conn = _get_db_connection()
            break
        except Error as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Database unavailable after 120 seconds (MySQL code {exc.errno})') from None
            print(f'Waiting for database (MySQL code {exc.errno})', flush=True)
            time.sleep(5)
    try:
        cur = conn.cursor()
        try:
            if not initialize:
                validate_schema(cur)
                return
            cur.execute("""CREATE TABLE IF NOT EXISTS users (
                userid BIGINT NOT NULL PRIMARY KEY,
                mode VARCHAR(20) NOT NULL DEFAULT 'manual',
                location_flag BOOLEAN NOT NULL DEFAULT 0
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS user_location (
                userid BIGINT NOT NULL PRIMARY KEY,
                latitude DOUBLE NOT NULL, longitude DOUBLE NOT NULL
            )""")
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()
    ensure_rain_state_schema()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Check assets/model/imports without DB writes or polling')
    parser.add_argument('--init-db', action='store_true', help='Explicit one-off schema setup/migration; requires CREATE/ALTER')
    args = parser.parse_args()
    if args.init_db:
        prepare_database(initialize=True)
        print('Database schema initialized; no bot polling started.')
        return
    check_assets()
    from main import load_model, main as run_bot
    load_model()  # Fail before altering schema or starting network workers.
    if args.check:
        print('Bot imports, model and normalization contract passed; no polling started.')
        return
    prepare_database()
    run_bot()


if __name__ == '__main__':
    main()
