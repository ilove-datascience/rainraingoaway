import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from locked_forward_evaluation import ap_stats, replay_details, runs
from replay_arrival_notifications import score_run


def keys(n, start=datetime(2026, 9, 11)):
    return [(start+timedelta(minutes=5*i)).strftime('%Y%m%d%H%M') for i in range(n)]


def test_details_match_existing_policy_and_additive_day_counts():
    rng = np.random.default_rng(17)
    ticks = keys(180, datetime(2026, 9, 10, 22)) + keys(100, datetime(2026, 9, 12))
    wet = rng.random((len(ticks), 3)) < .2
    probs = rng.random(wet.shape)
    for horizon in (15, 30, 60):
        total, daily, blocks, events = replay_details(ticks, wet, probs, horizon, .5)
        originals = [score_run(ticks[a:b], wet[a:b, j], probs[a:b, j], horizon, .5)
                     for a, b in runs(ticks) for j in range(3)]
        for field in ('alerts', 'true_alerts', 'false_alerts', 'onsets', 'warned_onsets', 'location_hours'):
            assert np.isclose(total[field], sum(r[field] for r in originals))
            assert np.isclose(total[field], sum(r[field] for r in daily.values()))
            assert np.isclose(total[field], sum(r[field] for r in blocks))
        assert len(events) == total['alerts']
        assert sorted(total['lead_minutes']) == sorted(x for r in daily.values() for x in r['lead_minutes'])


def test_midnight_warning_keeps_state_and_attributes_onset_to_next_day():
    ticks = keys(10, datetime(2026, 9, 10, 23, 40))
    wet = np.array([[False]]*4+[[True]]*6)
    probs = np.zeros((10, 1)); probs[2] = .9
    total, daily, _, _ = replay_details(ticks, wet, probs, 15, .5)
    assert total['warned_onsets'] == 1
    assert daily['20260910']['true_alerts'] == 1
    assert daily['20260910']['lead_minutes'] == [10]
    assert daily['20260911']['warned_onsets'] == 1


def test_missing_timestamp_splits_run_and_never_matches_across_gap():
    ticks = keys(5) + keys(5, datetime(2026, 9, 11, 1))
    wet = np.array([[False]]*5+[[True]]*5)
    total, _, blocks, _ = replay_details(ticks, wet, np.ones((10, 1)), 15, .5)
    assert len(blocks) == 2
    assert total['true_alerts'] == 0
    assert total['onsets'] == 0


def test_ap_uses_dry_anchors_and_ties():
    p = np.array([.5, .5, 1.])
    y = np.array([True, False, True])
    r = ap_stats(p, y, np.array([True, True, False]))
    assert r['AP'] == .5 and r['ap_samples'] == 2
    assert ap_stats(p, y, np.zeros(3, dtype=bool))['AP'] is None
