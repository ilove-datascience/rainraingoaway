"""Verify the historical report survives publication without changing its lock."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'docs/evaluations/2026-09-15-forward'


def test_historical_lock_checksum_is_unchanged():
    expected = (ROOT / 'lock.sha256').read_text().strip()
    assert hashlib.sha256((ROOT / 'lock.json').read_bytes()).hexdigest() == expected
    assert json.loads((ROOT / 'results.json').read_text())['lock_sha256'] == expected


def test_published_day_and_run_counts_reconcile():
    aggregate = json.loads((ROOT / 'results.json').read_text())['aggregate']
    for filename in ('by_day.csv', 'by_continuous_run.csv'):
        with (ROOT / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        for total in aggregate:
            group = [r for r in rows if r['model'] == total['model'] and int(r['horizon']) == total['horizon']]
            for field in ('alerts', 'false_alerts', 'onsets', 'warned_onsets', 'location_hours', 'ap_samples', 'ap_positive'):
                assert math.isclose(sum(float(r[field]) for r in group), total[field], abs_tol=1e-8)
