"""Archive completed Singapore rainfall days independently of forecast inputs."""
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scraping.rainfall_schema import URL, SGT, parse_pages
from scraping.gov_api import OUTPUT_DIR, _get_json_with_retry

LOG = logging.getLogger(__name__)
ARCHIVE = OUTPUT_DIR / 'rainfall'


def fetch_day(day, folder=ARCHIVE):
    folder = Path(folder)
    path = folder / f'rainfall_{day:%Y%m%d}.json'
    # Existing frozen archives are immutable, including their retrieval metadata.
    if path.exists():
        return path
    pages, seen, token = [], set(), None
    while True:
        params = {'date': day.isoformat()}
        if token:
            params['paginationToken'] = token
        page = _get_json_with_retry(URL, params=params, timeout_seconds=30)
        if page.get('code') != 0:
            raise ValueError('Rainfall API failure')
        pages.append(page)
        token = page['data'].get('paginationToken')
        if not token:
            break
        if token in seen:
            raise ValueError('Repeated rainfall pagination token')
        seen.add(token)
        time.sleep(.5)
    _, observations, _ = parse_pages(pages, day.strftime('%Y%m%d'))
    if not observations:
        raise ValueError('No rainfall observations; will retry instead of marking complete')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(dict(url=URL, date=day.isoformat(),
            retrieved_utc=datetime.now(timezone.utc).isoformat(), pages=pages)), encoding='utf-8')
        # Exclusive publication prevents overwriting another collector's archive.
        import os
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info('Rainfall archived %s: %s station readings', day, len(observations))
    return path


def collect_missing_days(folder=ARCHIVE, now=None):
    now = now or datetime.now(SGT)
    if now.tzinfo is None:
        raise ValueError('Timezone-aware time required')
    today = now.astimezone(SGT).date()
    folder = Path(folder)
    dates = []
    for p in folder.glob('rainfall_????????.json'):
        try:
            dates.append(datetime.strptime(p.stem[9:], '%Y%m%d').date())
        except ValueError:
            continue
    # Repair recent holes as well as arbitrarily long downtime after last archive.
    start = min(max(dates) + timedelta(days=1), today - timedelta(days=14)) if dates else today - timedelta(days=14)
    failures = []
    day = start
    while day < today:
        try:
            fetch_day(day, folder)
        except Exception as exc:
            LOG.warning('Rainfall archive failed for %s (%s); will retry', day, type(exc).__name__)
            failures.append(day.isoformat())
        day += timedelta(days=1)
    return failures


def main():
    while True:
        try:
            collect_missing_days()
        except Exception:
            LOG.exception('Rainfall archive pass failed; will retry')
        time.sleep(3600)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
