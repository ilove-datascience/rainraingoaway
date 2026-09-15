"""Freeze first, then score two saved arrival finalists without fitting anything."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from arrival.data import FrameStore, fingerprint, input_crop, shifted, tick_time
from arrival.calibration import project_cumulative
from arrival.ranking import precision_recall
from arrival.training import load_model
from replay_arrival_notifications import score_run
from audit_arrival_calibration import apply

OUT = ROOT / 'reports/locked_forward_20260826_20260914'
HORIZONS = (15, 30, 60)
START, END = '202608260000', '202609150000'
SOURCES = {
    'v5': ('models/arrival/runs/gru_t6_all_seed67_v5/best.pt',
           'reports/arrival_v5_calibration/projection_experiment.json',
           'reports/arrival_notification_replay'),
    'depth2_spatial': ('models/arrival/experiments_architecture_v1/gru_t6_depth2_spatial_seed67/best.pt',
                       'models/arrival/experiments_architecture_v1/gru_t6_depth2_spatial_seed67/calibrator.json',
                       'reports/arrival_replay_gru_t6_depth2_spatial_seed67'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze():
    if (OUT / 'lock.json').exists():
        raise FileExistsError('A lock already exists; use evaluate to honor it.')
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_json(ROOT / 'models/arrival/data_v1/prepared.json')
    records = read_json(ROOT / 'models/arrival/data_v1/evaluation_locations.json')
    selected = read_json(ROOT / 'reports/arrival_matched_burden/comparison.json')['selected']
    lock = dict(created_utc=datetime.now(timezone.utc).isoformat(), start=START, end_exclusive=END,
                false_warning_target_per_location_day=.25, models={}, config=data['config'], norm=data['norm'])
    for split in ('train', 'validation', 'calibration'):
        assert max(data['manifests'][split]) < START
    lock['old_split_ranges'] = {s: [min(k), max(k), len(k)] for s, k in data['manifests'].items() if k}
    for name, (checkpoint, calibration, replay) in SOURCES.items():
        cp, cal = ROOT / checkpoint, ROOT / calibration
        candidate = read_json(cal)
        old = read_json(ROOT / replay / 'results.json')['settings']
        assert digest(cp) == candidate['checkpoint_sha256'] == old['checkpoint']
        assert candidate['parameters'] == old['parameters']
        assert old['keys'] == data['manifests']['validation']
        if 'locations' in lock:
            assert lock['locations'] == old['locations']
        lock['locations'] = old['locations']
        model, state = load_model(cp, 'cpu')
        assert state['norm'] == data['norm']
        assert state['train_anchor_hash'] == fingerprint(data['manifests']['train'])
        assert state['validation_records_hash'] == fingerprint(records['validation'])
        assert state['data_config']['development_end'] == '2026-08-26'
        assert state['model_config']['history'] == 6 and list(state['channels']) == list(range(7))
        if 'calibration_records_hash' in candidate:
            assert candidate['calibration_records_hash'] == fingerprint(records['calibration'])
        frozen = OUT / 'frozen' / name
        frozen.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cp, frozen / 'best.pt')
        shutil.copyfile(cal, frozen / 'calibrator.json')
        rows = [r for r in selected if r['model'] == name and r['cap_false_per_day'] == .25]
        assert sorted(r['horizon'] for r in rows) == list(HORIZONS)
        lock['models'][name] = dict(checkpoint_source=checkpoint, calibration_source=calibration,
            checkpoint_sha256=digest(cp), calibrator_sha256=digest(cal),
            model_config=state['model_config'], parameter_count=sum(p.numel() for p in model.parameters()),
            thresholds={str(r['horizon']): r['threshold'] for r in rows}, old_validation_operating_points=rows)
        del model
    lock['definitions'] = {
        'location': 'Same nine fixed (y,x) radar pixels as old replay; centered 64x64 crops. No new location selection.',
        'wet': 'At least 5 of 25 pixels in the centered 5x5 patch have cleaned SOURCE radar intensity > 0.01.',
        'onset': 'A dry-to-wet transition at a fixed location within an uninterrupted eligible 5-minute run.',
        'eligibility': 'All 12 historical radar/environment frames and 12 future radar frames exist within the requested period; each historical environment frame passes the existing causal validity check. Same max_history=12 common eligibility as development.',
        'ap': 'Tie-aware average precision at currently dry eligible location/timestamps; positive if rain first occurs within 15/30/60 minutes. All eligible anchors, no sampling. Calibrated projected probabilities.',
        'warning': 'Existing score_run policy: >= frozen threshold, one confirmation, suppress until onset or horizon expiry; no clear notifications. Full forward horizon within eligible run required to issue warnings; onsets require full prior horizon within run.',
        'exposure': 'Same old-replay denominator: max(run_length - horizon_steps, 0) * 5 minutes per location, including wet times.',
        'day_attribution': 'Do not reset at midnight. Warnings/false warnings/lead time belong to issue day; arrivals warned/onsets belong to onset day; exposure and AP belong to anchor day.',
        'continuous_weather_run': 'Maximal uninterrupted 5-minute sequence of eligible input anchors. This is a data-continuity block, not a meteorologically independent storm.',
        'boundaries': 'Missing inputs split runs; no filling missing radar as dry. AP uses complete raw future radar even at eligible-run edges; warning/onset censoring follows old replay.',
    }
    lock['provenance'] = {
        'status': 'Forward relative to both finalists training/validation/calibration dates, but not a pristine independent holdout.',
        'prior_exposure': 'reports/heldout_misses/improvement_plan.md documents inspected 26 August-10 September targets and resulting development ideas. September 11-14 is reported separately, without claiming verified never-inspected status.',
        'publication_latency': 'Archived observation timestamps only; historical publication/ingestion timing and Telegram delivery latency unavailable.',
    }
    lock['decision_rule'] = {
        'advance_spatial_only_if': 'Every horizon false-warning rate <=0.25; macro onset-recall gain >=0.05; paired continuity-block bootstrap 95% lower bound for macro recall gain >0; no horizon recall loss >0.05; mean AP at least v5. Given prior data exposure, satisfying this is provisional support, not independent confirmation.',
        'otherwise': 'Keep v5; stop architecture tuning and collect additional months, reserving a genuinely untouched prospective period.',
        'bootstrap': '2000 paired continuity-run resamples, seed 20260915; diagnostic uncertainty, blocks need not be independent storms.',
    }
    # Inventory and hash files without inspecting outcomes. The exact data set is locked too.
    radar = sorted(p for p in (ROOT / 'data/70km/png').glob('*.png') if START <= p.stem < END)
    assert radar, 'No radar frames in requested period'
    files = {}
    for p in radar:
        files[str(p.relative_to(ROOT)).replace('\\', '/')] = digest(p)
        env = ROOT / 'data/environment' / f'weather_{p.stem}.csv'
        if env.exists():
            files[str(env.relative_to(ROOT)).replace('\\', '/')] = digest(env)
    lock['radar_keys'] = [p.stem for p in radar]
    lock['input_hashes'] = files
    sources = [Path(__file__), ROOT / 'scripts/replay_arrival_notifications.py', ROOT / 'scripts/audit_arrival_calibration.py',
               ROOT / 'src/data_processing/data_loading.py', ROOT / 'src/data_processing/radar_codec.py',
               ROOT / 'reports/arrival_matched_burden/comparison.json', ROOT / 'models/arrival/data_v1/prepared.json',
               ROOT / 'reports/heldout_misses/improvement_plan.md'] + sorted((ROOT / 'src/arrival').glob('*.py'))
    lock['source_hashes'] = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sources}
    write_json(OUT / 'lock.json', lock)
    (OUT / 'lock.sha256').write_text(digest(OUT / 'lock.json') + '\n')
    print('LOCK WRITTEN', digest(OUT / 'lock.json'), flush=True)
    print('Raw coverage', radar[0].stem, radar[-1].stem, len(radar), flush=True)
    for name, entry in lock['models'].items():
        print(name, entry['thresholds'], entry['parameter_count'], flush=True)


def verify(lock):
    assert digest(OUT / 'lock.json') == (OUT / 'lock.sha256').read_text().strip()
    for group in ('input_hashes', 'source_hashes'):
        for path, expected in lock[group].items():
            if digest(ROOT / path) != expected:
                raise ValueError(f'Frozen file changed: {path}')
    for name, entry in lock['models'].items():
        for file, key, source in [('best.pt', 'checkpoint_sha256', 'checkpoint_source'),
                                  ('calibrator.json', 'calibrator_sha256', 'calibration_source')]:
            assert digest(OUT / 'frozen' / name / file) == entry[key]
            assert digest(ROOT / entry[source]) == entry[key]


def runs(keys):
    edges = [0] + [i for i in range(1, len(keys)) if tick_time(keys[i])-tick_time(keys[i-1]) != timedelta(minutes=5)] + [len(keys)]
    return list(zip(edges[:-1], edges[1:])) if keys else []


def blank():
    return dict(alerts=0, true_alerts=0, false_alerts=0, onsets=0, warned_onsets=0,
                location_hours=0., lead_minutes=[])


def finish(row):
    row = dict(row)
    leads = row.pop('lead_minutes')
    row['median_lead_minutes'] = float(np.median(leads)) if leads else None
    row['onset_recall'] = row['warned_onsets']/row['onsets'] if row['onsets'] else None
    row['precision'] = row['true_alerts']/row['alerts'] if row['alerts'] else None
    row['false_warnings_per_location_day'] = row['false_alerts']/(row['location_hours']/24) if row['location_hours'] else None
    return row


def replay_details(keys, wet, probs, horizon, threshold):
    """Retain the old replay policy, adding additive day attribution and event records."""
    total, daily, blocks, events = blank(), defaultdict(blank), [], []
    h = horizon//5
    for run_id, (lo, hi) in enumerate(runs(keys)):
        block = blank()
        for j in range(wet.shape[1]):
            r = score_run(keys[lo:hi], wet[lo:hi, j], probs[lo:hi, j], horizon, threshold)
            for target in (total, block):
                for k in target:
                    if k == 'lead_minutes': target[k].extend(r[k])
                    else: target[k] += r[k]
            warned = set()
            for alert in r['alert_records']:
                idx = lo + alert['index']
                day = keys[idx][:8]
                daily[day]['alerts'] += 1
                daily[day]['true_alerts' if alert['hit'] else 'false_alerts'] += 1
                if alert['hit']:
                    warned.add(alert['index'] + alert['lead_minutes']//5)
                    daily[day]['lead_minutes'].append(alert['lead_minutes'])
                events.append(dict(run_id=run_id, location=j, **alert))
            for idx in range(lo, max(lo, hi-h)):
                daily[keys[idx][:8]]['location_hours'] += 1/12
            for idx in range(max(1, h), hi-lo):
                if wet[lo+idx, j] and not wet[lo+idx-1, j]:
                    daily[keys[lo+idx][:8]]['onsets'] += 1
                    daily[keys[lo+idx][:8]]['warned_onsets'] += int(idx in warned)
        blocks.append(dict(run_id=run_id, start=keys[lo], end=keys[hi-1], timestamps=hi-lo, **block))
    return total, dict(daily), blocks, events


def ap_stats(probs, truth, valid):
    scores, labels = probs[valid], truth[valid]
    return dict(AP=precision_recall(scores, labels)['average_precision'] if len(scores) else None,
                ap_samples=int(len(scores)), ap_positive=int(labels.sum()),
                prevalence=float(labels.mean()) if len(labels) else None)


def csv_write(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def bootstrap(run_rows):
    ids = sorted({r['run_id'] for r in run_rows})
    lookup = {(r['model'], r['horizon'], r['run_id']): r for r in run_rows}
    counts = {name: np.array([[[lookup[name, h, i]['warned_onsets'], lookup[name, h, i]['onsets']]
                              for h in HORIZONS] for i in ids], dtype=float) for name in SOURCES}
    rng = np.random.default_rng(20260915)
    deltas = []
    for _ in range(2000):
        indices = rng.integers(len(ids), size=len(ids))
        sums = {name: a[indices].sum(0) for name, a in counts.items()}
        if (sums['v5'][:, 1] == 0).any(): continue
        deltas.append(float(np.mean(sums['depth2_spatial'][:, 0]/sums['depth2_spatial'][:, 1]
                                    - sums['v5'][:, 0]/sums['v5'][:, 1])))
    return dict(valid_draws=len(deltas), macro_recall_gain_95_interval=np.quantile(deltas, [.025, .975]).tolist() if deltas else None)


def evaluate():
    lock = read_json(OUT / 'lock.json')
    verify(lock)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Completed results already exist; do not overwrite the locked test.')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    store = FrameStore(ROOT)
    key_set = set(lock['radar_keys'])
    # Ignore later-arriving input files; only files in the lock can qualify an anchor.
    keys, excluded = [], Counter()
    for key in lock['radar_keys']:
        history = [shifted(key, -5*i) for i in range(12)]
        future = [shifted(key, 5*i) for i in range(1, 13)]
        if not all(k in key_set for k in history + future):
            excluded['incomplete_radar_window_or_period_boundary'] += 1; continue
        if not all(f'data/environment/weather_{k}.csv' in lock['input_hashes'] and store.environment_valid(k) for k in history):
            excluded['incomplete_causal_environment_history'] += 1; continue
        keys.append(key)
    if not keys: raise ValueError('No complete common evaluation anchors')
    write_json(OUT / 'eligibility.json', dict(keys=keys, excluded=dict(excluded), by_day=dict(Counter(k[:8] for k in keys))))
    print('Eligible', len(keys), 'continuous runs', len(runs(keys)), 'excluded', dict(excluded), flush=True)
    locs = lock['locations']
    models = {}
    for name in SOURCES:
        model, state = load_model(OUT / 'frozen' / name / 'best.pt', device)
        models[name] = (model, read_json(OUT / 'frozen' / name / 'calibrator.json'))
    wet = np.zeros((len(keys), len(locs)), dtype=bool)
    truth = np.zeros((len(keys), len(locs), 3), dtype=bool)
    probs = {name: np.zeros(truth.shape, dtype=float) for name in models}
    raw_probs = {name: np.zeros(truth.shape, dtype=float) for name in models}
    cache_path = OUT / 'predictions.npz'
    if cache_path.exists():
        with np.load(cache_path) as cache:
            assert str(cache['lock_sha256']) == digest(OUT / 'lock.json')
            np.testing.assert_array_equal(cache['keys'], keys)
            wet, truth = cache['wet'], cache['truth']
            for name in models: probs[name], raw_probs[name] = cache[name], cache[name + '_raw']
        print('Using predictions from this exact lock', flush=True)
    else:
        batch, positions = [], []
        def flush():
            if not batch: return
            inputs = torch.stack(batch).to(device)
            with torch.inference_mode():
                for name, (model, cal) in models.items():
                    raw = model.log_probs(inputs).double().softmax(1).cpu().numpy().cumsum(1)[:, :12]
                    p = project_cumulative(np.column_stack([apply(raw[:, k], cal['parameters'][k]) for k in range(12)]))
                    for pos, values, raw_values in zip(positions, p[:, [2, 5, 11]], raw[:, [2, 5, 11]]):
                        probs[name][pos], raw_probs[name][pos] = values, raw_values
            batch.clear(); positions.clear()
        for i, key in enumerate(keys):
            radar = store.radar(key)
            for j, (y, x) in enumerate(locs):
                wet[i, j] = int((radar[y-2:y+3, x-2:x+3] > .01).sum()) >= 5
            future_wet = []
            for step in range(1, 13):
                radar = store.radar(shifted(key, 5*step))
                future_wet.append([int((radar[y-2:y+3, x-2:x+3] > .01).sum()) >= 5 for y, x in locs])
            for hi, horizon in enumerate(HORIZONS):
                truth[i, :, hi] = np.any(future_wet[:horizon//5], axis=0)
            for j in np.flatnonzero(~wet[i]):
                batch.append(input_crop(store, key, *locs[j], lock['norm']))
                positions.append((i, j))
                if len(batch) >= 32: flush()
            if i % 200 == 0: print('Forward input pass', i, '/', len(keys), flush=True)
        flush()
        np.savez_compressed(cache_path, keys=np.array(keys), lock_sha256=digest(OUT / 'lock.json'),
                            wet=wet, truth=truth, **probs, **{n+'_raw': p for n, p in raw_probs.items()})
    aggregate, day_rows, run_rows, event_rows = [], [], [], []
    days = np.array([k[:8] for k in keys])
    requested_days = [(datetime(2026, 8, 26)+timedelta(days=i)).strftime('%Y%m%d') for i in range(20)]
    for name in models:
        for hi, horizon in enumerate(HORIZONS):
            threshold = lock['models'][name]['thresholds'][str(horizon)]
            total, daily, blocks, events = replay_details(keys, wet, probs[name][:, :, hi], horizon, threshold)
            base = dict(model=name, horizon=horizon, threshold=threshold)
            aggregate.append(dict(**base, **finish(total), **ap_stats(probs[name][:, :, hi], truth[:, :, hi], ~wet),
                                  raw_AP=ap_stats(raw_probs[name][:, :, hi], truth[:, :, hi], ~wet)['AP']))
            for day in requested_days:
                mask = days == day
                day_rows.append(dict(**base, day=day, eligible_timestamps=int(mask.sum()), **finish(daily.get(day, blank())),
                    **ap_stats(probs[name][mask, :, hi], truth[mask, :, hi], ~wet[mask])))
            for block, (lo, up) in zip(blocks, runs(keys)):
                run_rows.append(dict(**base, **finish(block),
                    **ap_stats(probs[name][lo:up, :, hi], truth[lo:up, :, hi], ~wet[lo:up])))
            event_rows.extend(dict(**base, **event) for event in events)
    # Predeclared later-period breakout, using the same thresholds and censoring policy.
    subset = np.flatnonzero(days >= '20260911')
    later = []
    for name in models:
        for hi, horizon in enumerate(HORIZONS):
            threshold = lock['models'][name]['thresholds'][str(horizon)]
            subkeys = [keys[i] for i in subset]
            total, _, _, _ = replay_details(subkeys, wet[subset], probs[name][subset, :, hi], horizon, threshold)
            later.append(dict(model=name, horizon=horizon, threshold=threshold, **finish(total),
                              **ap_stats(probs[name][subset, :, hi], truth[subset, :, hi], ~wet[subset])))
    uncertainty = bootstrap(run_rows)
    lookup = {(r['model'], r['horizon']): r for r in aggregate}
    recall_delta = [lookup['depth2_spatial', h]['onset_recall'] - lookup['v5', h]['onset_recall'] for h in HORIZONS]
    ap_delta = [lookup['depth2_spatial', h]['AP'] - lookup['v5', h]['AP'] for h in HORIZONS]
    interval = uncertainty['macro_recall_gain_95_interval']
    advance = (all(lookup['depth2_spatial', h]['false_warnings_per_location_day'] <= .25 for h in HORIZONS)
               and np.mean(recall_delta) >= .05 and interval is not None and interval[0] > 0
               and min(recall_delta) >= -.05 and np.mean(ap_delta) >= 0)
    decision = ('Spatial meets the predeclared performance rule; provisional advancement only because the period was previously inspected.' if advance
                else 'Keep v5. Spatial does not clearly beat v5 under the frozen rule. Stop architecture tuning and collect more months with a truly reserved prospective test.')
    verify(lock)
    result = dict(lock_sha256=digest(OUT / 'lock.json'), completed_utc=datetime.now(timezone.utc).isoformat(),
        device=device, torch_version=str(torch.__version__), raw_frames=len(lock['radar_keys']),
        raw_first=lock['radar_keys'][0], raw_last=lock['radar_keys'][-1], eligible_anchors=len(keys),
        eligible_first=keys[0], eligible_last=keys[-1], continuous_runs=len(runs(keys)), excluded=dict(excluded),
        aggregate=aggregate, september_11_14=later, uncertainty=uncertainty, decision=decision,
        macro_recall_gain_spatial=float(np.mean(recall_delta)), mean_AP_gain_spatial=float(np.mean(ap_delta)),
        provenance=lock['provenance'])
    csv_write(OUT / 'by_day.csv', day_rows)
    csv_write(OUT / 'by_continuous_run.csv', run_rows)
    csv_write(OUT / 'warning_events.csv', event_rows)
    write_json(OUT / 'results.json', result)
    def fmt(value): return 'n/a' if value is None else f'{value:.3f}'
    def table(rows):
        lines = ['| Model | Minutes | Threshold | AP | Warned/onsets | False/location-day | Median lead (min) |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['model']} | {r['horizon']} | {r['threshold']:.3f} | {fmt(r['AP'])} | {r['warned_onsets']}/{r['onsets']} | {fmt(r['false_warnings_per_location_day'])} | {fmt(r['median_lead_minutes'])} |")
        return lines
    lines = ['# Locked forward comparison: 26 August–14 September 2026', '', decision, '',
             lock['provenance']['status'], lock['provenance']['prior_exposure'], '',
             f"Raw coverage: {result['raw_first']} to {result['raw_last']} SGT; {len(keys)} eligible anchors, {len(runs(keys))} continuity runs, nine fixed locations.",
             'Checkpoint and calibrator copies, old validation thresholds, input files, scoring code and decision rule were hashed before inference. All hashes were rechecked after scoring. No model fitting, calibration fitting or threshold tuning occurred.', '',
             '## Full requested period', ''] + table(aggregate) + ['', '## September 11–14 breakout',
             '', 'Shorter, less-exposed date range; never-inspected status is not established. Same frozen thresholds.', ''] + table(later) + [
             '', f"Spatial minus v5 macro onset-recall difference: {np.mean(recall_delta):.3f}; paired continuity-block bootstrap 95% interval: {interval}.",
             'The bootstrap is diagnostic: continuity blocks can belong to the same weather system, and three horizons are correlated.', '',
             '## Definitions and detailed results', '', *[f'- **{k}**: {v}' for k, v in lock['definitions'].items()], '',
             '- [By day](by_day.csv): includes zero-coverage dates; no midnight reset.',
             '- [By continuous weather run](by_continuous_run.csv).',
             '- [Warning events](warning_events.csv).',
             '- [Frozen protocol](lock.json).',
             '- [Machine-readable results](results.json).', '',
             lock['provenance']['publication_latency'],
             'The 0.25 rate is the old-validation selection target, not a guaranteed forward rate. Rates above it are reported without retuning. These are radar-pixel arrival warnings, not observed user experiences or delivered Telegram notifications.']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(table(aggregate)), flush=True)
    print(decision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'evaluate'])
    args = parser.parse_args()
    freeze() if args.phase == 'freeze' else evaluate()
