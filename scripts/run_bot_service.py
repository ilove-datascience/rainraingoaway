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


def prepare_database():
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
    args = parser.parse_args()
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
