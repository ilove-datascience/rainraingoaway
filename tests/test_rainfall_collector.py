import json
import sys
from pathlib import Path
from datetime import date, datetime
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scraping import rainfall


def page(token=None):
    return dict(code=0, data=dict(readingUnit='mm', readingType='TB1 Rainfall 5 Minute Total F',
        stations=[dict(id='S1', name='Station', location=dict(latitude=1.33, longitude=103.8))],
        readings=[dict(timestamp='2026-10-24T12:00:00+08:00', data=[dict(stationId='S1', value=.2)])],
        paginationToken=token))


def test_archive_paginates_and_preserves_existing(tmp_path, monkeypatch):
    get=Mock(side_effect=[page('next'), page()])
    monkeypatch.setattr(rainfall, '_get_json_with_retry', get)
    monkeypatch.setattr(rainfall.time, 'sleep', lambda _: None)
    path=rainfall.fetch_day(date(2026,10,24), tmp_path)
    assert len(json.loads(path.read_text())['pages'])==2
    assert get.call_args.kwargs['params']['paginationToken']=='next'
    content=path.read_bytes()
    rainfall.fetch_day(date(2026,10,24), tmp_path)
    assert get.call_count==2 and path.read_bytes()==content


def test_invalid_or_empty_response_never_published(tmp_path, monkeypatch):
    import pytest
    for response in (dict(code=1), dict(code=0,data=dict(page()['data'], readings=[]))):
        monkeypatch.setattr(rainfall, '_get_json_with_retry', Mock(return_value=response))
        with pytest.raises(ValueError):rainfall.fetch_day(date(2026,10,24), tmp_path)
        assert not list(tmp_path.iterdir())


def test_pagination_loop_never_published(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(rainfall, '_get_json_with_retry', Mock(return_value=page('repeat')))
    monkeypatch.setattr(rainfall.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='Repeated'):rainfall.fetch_day(date(2026,10,24), tmp_path)
    assert not list(tmp_path.iterdir())


def test_retry_holes_and_never_fetch_current_singapore_day(tmp_path, monkeypatch):
    (tmp_path/'rainfall_20261015.json').write_text('{}')
    attempts=[]
    def fetch(day, folder):
        attempts.append(day)
        if day==date(2026,10,16):raise OSError('offline')
    monkeypatch.setattr(rainfall, 'fetch_day', fetch)
    # UTC date is still Oct24; local Singapore date is Oct25.
    now=datetime.fromisoformat('2026-10-24T18:00:00+00:00')
    assert rainfall.collect_missing_days(tmp_path,now)==['2026-10-16']
    assert max(attempts)==date(2026,10,24)
    attempts.clear()
    rainfall.collect_missing_days(tmp_path,now)
    assert date(2026,10,16) in attempts


def test_bot_starts_independent_rainfall_worker():
    import ast
    source=ast.parse((Path(__file__).resolve().parents[1]/'src/main.py').read_text())
    calls=[n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Thread']
    workers=[{k.arg:k.value for k in c.keywords} for c in calls]
    rainfall_workers=[w for w in workers if isinstance(w.get('target'),ast.Name) and w['target'].id=='run_rainfall_scraper_forever']
    assert len(rainfall_workers)==1
    assert rainfall_workers[0]['daemon'].value is True
