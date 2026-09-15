"""Generate the two furnished notebooks from readable, versionable cell sources."""
import json
from pathlib import Path
from textwrap import dedent

ROOT=Path(__file__).resolve().parents[1]


def md(text):return dict(cell_type='markdown',metadata={},source=dedent(text).strip().splitlines(keepends=True))
def code(text):return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=dedent(text).strip().splitlines(keepends=True))
def save(name,cells):
    for i,c in enumerate(cells):c['id']=f'cell-{i:03}'
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (.venv)',language='python',name='python3'),
        language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
    (ROOT/'src'/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')


setup=code('''
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/data_processing").exists())
if str(PROJECT_ROOT / "src") not in sys.path: sys.path.insert(0, str(PROJECT_ROOT / "src"))
from arrival.data import (DataConfig, FrameStore, LabelStore, ArrivalDataset, prepare,
                          fixed_locations, CLASSES, CHANNELS, fingerprint)
from arrival.models import ArrivalNet
from arrival.training import (seed_everything, fit, predict, load_model, metrics,
                              fit_calibrator, calibrated_probs, persistence, motion_baseline)
from arrival.plots import plot_history, plot_evaluation, plot_example
SEED = 67
seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| torch:", torch.__version__)
print("Channel order:", CHANNELS)
''')
data_config=code('''
# Shared across both notebooks. Use a NEW directory when changing label/split definitions.
DATA = DataConfig(crop=64, max_history=12, future=12, patch=5,
                  rain_threshold=0.01, coverage=0.20, stride=1,
                  development_end="2026-08-26", test_start=None, seed=SEED)
DATA_DIR = PROJECT_ROOT / "models/arrival/data_v1"
store = FrameStore(PROJECT_ROOT)
label_store = LabelStore(store, DATA)
# First run may take time: exact source decoding, labels and causal weather caches.
# No API or SQL access. No existing radar-frame checkpoints/caches are overwritten.
prepared = prepare(store, DATA, DATA_DIR, normalization_frames=512)
manifests, norm = prepared["manifests"], prepared["norm"]
print({split: len(anchors) for split, anchors in manifests.items()})
print(prepared["caveat"])
print("Manifest hash:", prepared["manifest_hash"])
''')
fixed=code('''
# Seeded uniform anchor subset and natural-frequency spatial sampling for quick evaluation.
# Set EVALUATION_ANCHORS=None for all eligible anchors. Keep this fixed across experiments.
EVALUATION_ANCHORS = 256
EVALUATION_LOCATIONS = 16
def evaluation_records(split):
    anchors = manifests[split]
    if EVALUATION_ANCHORS is not None and len(anchors) > EVALUATION_ANCHORS:
        rng = np.random.default_rng(SEED)
        anchors = sorted(rng.choice(anchors, EVALUATION_ANCHORS, replace=False).tolist())
    return fixed_locations(label_store, anchors, EVALUATION_LOCATIONS, SEED)
validation_records = evaluation_records("validation")
calibration_records = evaluation_records("calibration")
(DATA_DIR / "evaluation_locations.json").write_text(json.dumps({
    "validation": validation_records, "calibration": calibration_records}, indent=2))
print("Fixed validation/calibration examples:", len(validation_records), len(calibration_records))
def make_dataset(history=6, channels=tuple(range(7)), split="train"):
    records = None if split == "train" else (validation_records if split == "validation" else calibration_records)
    return ArrivalDataset(store, label_store, manifests[split], norm, history, channels,
                          per_anchor=32, seed=SEED, records=records)
''')
contract=md('''
## Data contract and leakage controls

- Source radar uses the exact 33-color decoder; values are ordinal, **not mm/hour**. The existing conservative radar cleaner remains; no blanket removal of 1–5-pixel targets.
- Crops are `[T, 7, 64, 64]`. The target is pixel `[32,32]`; boundary locations without a full crop are excluded. Channel order is radar, temperature, humidity, wind_u, wind_v, station mask, distance. No explicit coordinates are input, but environmental fields can still reveal geography.
- Currently dry means fewer than five of the center 25 pixels exceed 0.01. First future qualifying observation maps to class 0–11; class 12 requires **all twelve** future observations to remain dry. Missing future data never means no rain.
- For T=6, the required interval is t−25 to t+60 (18 frames, 85 minutes). Shared manifests conservatively reserve T=12 history: t−55 to t+60, keeping anchors identical across history ablations.
- Whole days are partitioned chronologically before constructing anchors; complete windows remain inside one partition. Development ends before the previous held-out period. Train/validation/calibration are separate; a **new** final-test period is disabled until explicitly configured.
- Weather uses the exact frame-time CSV and rejects observation timestamps later than that frame. Historical publication/ingestion timestamps are unavailable, so this is observation-time causality, not measured historical availability.
- Training selects one of four arrival groups with equal probability globally, then an eligible training anchor and location. Expected group frequencies are 25% each; individual batches vary. Validation/calibration locations are fixed and sampled without arrival balancing. Spatial/event correlation remains.
''')


main=[md('''
# Local rain-arrival model — 0 to 60 minutes

Predict a distribution over twelve five-minute arrival windows and “no rain within 60 minutes” at a currently dry location. This notebook prepares data, trains a small CNN + ConvGRU, evaluates baselines, calibrates probabilities and exports a self-describing checkpoint.

Select the project's `.venv` kernel. Run cells in order. **Training is opt-in** via `RUN_TRAINING`; no training or bot deployment occurs just by opening this file. The first data-preparation pass builds reusable local caches. Use the companion `rain_arrival_experiments.ipynb` for controlled comparisons.

This model does not generate a predicted radar image or predict rain ending. Those production capabilities remain separate.
'''),setup,contract,data_config,
code('''
counts = pd.DataFrame({k:v for k,v in prepared["class_counts"].items() if v is not None}, index=CLASSES)
display(counts)
counts.div(counts.sum()).plot.bar(figsize=(13,4), title="Natural eligible-location class frequencies (correlated samples)")
plt.ylabel("Fraction"); plt.tight_layout(); plt.show()
'''),fixed,
code('''
train_data = make_dataset()
validation_data = make_dataset(split="validation")
calibration_data = make_dataset(split="calibration")
x, y = train_data[0]
assert tuple(x.shape) == (6, 7, 64, 64)
print("Training examples/epoch:", len(train_data), "| example target:", CLASSES[y])
fig, axes = plt.subplots(1,6,figsize=(15,3))
for t, ax in enumerate(axes):
    ax.imshow(x[t,0],vmin=0,vmax=1,cmap="viridis"); ax.scatter(32,32,c="magenta",marker="+")
    ax.set_title(f"t{(t-5)*5:+} min"); ax.axis("off")
plt.show()
'''),md('''
## Baselines on the same natural-frequency subset

Persistence predicts no arrival because all eligible targets are currently dry. The motion baseline estimates one global translation from two full-domain frames; it is a transparent advection baseline, **not optical flow**. It cannot model growth/decay or nonuniform motion, and uses zero-filled domain boundaries. Use a fixed subset to bound runtime; compare learned predictions on this same subset later.
'''),code('''
BASELINE_ANCHORS = 64
unique_anchors = sorted({r[0] for r in validation_records})
rng = np.random.default_rng(SEED)
baseline_keys = set(rng.choice(unique_anchors, min(BASELINE_ANCHORS,len(unique_anchors)), replace=False))
baseline_records = [r for r in validation_records if r[0] in baseline_keys]
baseline_truth = np.array([label_store.get(k)[y,x] for k,y,x in baseline_records])
persistence_probs = persistence(baseline_records)
motion_probs, motion_shifts = motion_baseline(store, baseline_records, DATA)
baseline_results = {"persistence": metrics(persistence_probs,baseline_truth),
                    "global_translation": metrics(motion_probs,baseline_truth)}
display(pd.DataFrame({k:{m:v[m] for m in ["accuracy","macro_f1_all_13","missed_arrivals","false_arrivals"]}
                      for k,v in baseline_results.items()}).T)
(DATA_DIR / "baselines.json").write_text(json.dumps(baseline_results,indent=2))
'''),md('''
## Train or load the model

The default head is 13-way classification, trained with negative log likelihood (equivalent to cross-entropy on logits). Mean validation average precision across 15/30/60 minutes selects the checkpoint. The lowest-NLL checkpoint is saved separately as best_nll.pt. The v5 run uses up to 30,000 updates, stopping after 40 checks without ranking improvement. Training loss uses balanced sampling while validation uses natural frequencies, so their absolute levels need not match. Use a fresh run name; existing checkpoints are never overwritten by a new run.
'''),code('''
RUN_TRAINING = False  # Set True when ready for the full training run.
RUN_NAME = "gru_t6_all_seed67_v5"
RUN_DIR = PROJECT_ROOT / "models/arrival/runs" / RUN_NAME
UPDATE_BUDGET = 30000
model = None
if RUN_TRAINING:
    seed_everything(SEED)
    model = ArrivalNet(history=6, channels=7, kind="gru", output="categorical").to(DEVICE)
    history = fit(model, train_data, validation_data, RUN_DIR, budget=UPDATE_BUDGET,
                  eval_every=250, patience=40, batch=16, seed=SEED, selection="mean_ap")
    plot_history(history)
elif (RUN_DIR / "best.pt").exists():
    model, checkpoint = load_model(RUN_DIR / "best.pt", DEVICE)
    if checkpoint["validation_records_hash"] != fingerprint(validation_records):
        raise ValueError("Checkpoint uses different evaluation records")
    if checkpoint["norm"] != norm: raise ValueError("Checkpoint normalization differs from prepared data")
    plot_history(json.loads((RUN_DIR / "history.json").read_text()))
else:
    print("Data is ready. Set RUN_TRAINING = False above and rerun that cell.")
'''),code('''
if model is not None:
    validation_log_probs, validation_truth = predict(model, validation_data)
    raw_probs = calibrated_probs(validation_log_probs)
    raw_metrics = metrics(raw_probs, validation_truth)
    display(pd.DataFrame(raw_metrics["horizons"]).T)
    print("Macro F1:",raw_metrics["macro_f1_all_13"],"| conditional time MAE:",raw_metrics["arrival_mae_minutes_detected"])
    print("Missed / actual arrivals:",raw_metrics["missed_arrivals"],"/",raw_metrics["rainy_cases"])
    plot_evaluation(raw_probs,validation_truth,raw_metrics)
    plot_example(validation_data,0,raw_probs[0])
    subset = ArrivalDataset(store,label_store,manifests["validation"],norm,records=baseline_records)
    lp, yt = predict(model,subset)
    baseline_results["gru_same_subset"] = metrics(calibrated_probs(lp),yt)
    display(pd.DataFrame({k:{m:v[m] for m in ["accuracy","macro_f1_all_13","missed_arrivals","false_arrivals"]}
                          for k,v in baseline_results.items()}).T)

    from arrival.ranking import plot_ranking, ranking_report
    # Compute explicitly: a running kernel may still hold the older metrics().
    raw_metrics['ranking'] = ranking_report(raw_probs, validation_truth)
    baseline_results['persistence']['ranking'] = ranking_report(persistence_probs, baseline_truth)
    baseline_results['global_translation']['ranking'] = ranking_report(motion_probs, baseline_truth)
    baseline_results['gru_same_subset']['ranking'] = ranking_report(calibrated_probs(lp), yt)
    plot_ranking(raw_metrics['ranking'])
    # Constant probabilities estimated exclusively from natural training labels.
    prior = np.asarray(prepared['class_counts']['train'], dtype=float)
    prior /= prior.sum()
    baseline_results['constant_training_prior'] = metrics(np.tile(prior, (len(yt),1)), yt)
    baseline_results['constant_training_prior']['ranking'] = ranking_report(np.tile(prior, (len(yt),1)), yt)
    comparison = []
    for name, result in baseline_results.items():
        for horizon, ranking in result['ranking'].items():
            comparison.append(dict(model=name, horizon=int(horizon),
                average_precision=ranking['average_precision'], prevalence=ranking['prevalence'],
                brier=result['horizons'][int(horizon)]['brier']))
    display(pd.DataFrame(comparison))
    (RUN_DIR / 'validation_metrics.json').write_text(json.dumps(raw_metrics,indent=2))
    (RUN_DIR / 'matched_baselines.json').write_text(json.dumps(baseline_results,indent=2))
'''),md('''
## Calibrate probabilities on the separate natural-frequency partition

A shared increasing logistic map calibrates cumulative arrival probabilities using calibration data only. It preserves ranking at every horizon and ensures longer-horizon arrival probability never decreases. The previous class-bias calibration is retained as a comparison. This shared map is less flexible, so compare probability error as well as ranking. Calibration-fit scores are in-sample and are **not final performance claims**. Freeze the model/calibrator before evaluating a newly reserved test period. No-rain mistakes remain explicit; time MAE is only among true arrivals also predicted to arrive.
'''),code('''
calibrator = None
if model is not None:
    calibration_log_probs, calibration_truth = predict(model,calibration_data)
    calibrator = fit_calibrator(calibration_log_probs,calibration_truth)
    before = metrics(calibrated_probs(calibration_log_probs),calibration_truth)
    after_probs = calibrated_probs(calibration_log_probs,calibrator)
    after = metrics(after_probs,calibration_truth)
    display(pd.DataFrame({"before":before["horizons"],"after (fit partition)":after["horizons"]}))
    plot_evaluation(after_probs,calibration_truth,after)
    (RUN_DIR / "calibrator.json").write_text(json.dumps(calibrator,indent=2))
    (RUN_DIR / "calibration_fit_metrics.json").write_text(json.dumps({"before":before,"after":after},indent=2))
    probabilities = calibrated_probs(validation_log_probs[:1],calibrator)[0]
    display(pd.Series(probabilities,index=CLASSES,name="Example arrival distribution"))
    print({f"within_{minutes}_min":float(probabilities[:minutes//5].sum()) for minutes in (15,30,60)})

    # Full fixed validation sample, not just the 64-anchor baseline subset.
    # These are development scores: checkpoint selection already used validation.
    full_motion, _ = motion_baseline(store, validation_records, DATA)
    prior = np.asarray(prepared['class_counts']['train'], dtype=float)
    prior /= prior.sum()
    full_results = {
        'raw_gru': metrics(raw_probs, validation_truth),
        'calibrated_gru': metrics(calibrated_probs(validation_log_probs,calibrator), validation_truth),
        'motion': metrics(full_motion, validation_truth),
        'constant_training_prior': metrics(np.tile(prior,(len(validation_truth),1)),validation_truth),
    }
    rows=[]
    for name,result in full_results.items():
        for horizon,ranking in result['ranking'].items():
            scores=result['horizons'][int(horizon)]
            rows.append(dict(model=name,horizon=int(horizon),AP=ranking['average_precision'],
                             prevalence=ranking['prevalence'],**scores))
    display(pd.DataFrame(rows))
    (RUN_DIR/'full_validation_comparison.json').write_text(json.dumps(full_results,indent=2))
    plot_ranking(full_results['calibrated_gru']['ranking'])

    # Compare the previous calibration on the same development examples.
    from arrival.training import fit_legacy_calibrator
    legacy = fit_legacy_calibrator(calibration_log_probs, calibration_truth)
    legacy_result = metrics(calibrated_probs(validation_log_probs,legacy),validation_truth)
    (RUN_DIR/'legacy_calibration_comparison.json').write_text(json.dumps(legacy_result,indent=2))
    print('Legacy calibration AP:', {h:r['average_precision'] for h,r in legacy_result['ranking'].items()})
    # Threshold tradeoffs are development diagnostics, not chosen deployment settings.
    threshold_rows=[]
    calibrated_validation=calibrated_probs(validation_log_probs,calibrator)
    truth_array=np.asarray(validation_truth)
    for minutes in (15,30,60):
        probability=calibrated_validation[:,:minutes//5].sum(1)
        truth=truth_array<minutes//5
        for threshold in (.01,.02,.05,.10,.20,.30,.50):
            decision=probability>=threshold
            tp=int((decision & truth).sum()); fp=int((decision & ~truth).sum())
            threshold_rows.append(dict(horizon=minutes,threshold=threshold,
                precision=tp/max(tp+fp,1),recall=tp/max(int(truth.sum()),1),
                true_alerts=tp,false_alerts=fp,missed=int(truth.sum())-tp))
    display(pd.DataFrame(threshold_rows))
    (RUN_DIR/'validation_threshold_tradeoffs.json').write_text(json.dumps(threshold_rows,indent=2))
'''),md('''
## Historical-only inference

The helper below reads only the anchor and its preceding history. It does not read arrival labels or future images. An already-raining target returns a separate status rather than an arrival forecast. The example reuses a known validation location for convenience; its inputs are still historical-only.
'''),code('''
from arrival.inference import predict_location
if model is not None:
    anchor, target_y, target_x = validation_records[0]
    forecast = predict_location(model,store,anchor,target_y,target_x,norm,DATA,calibrator)
    print(json.dumps(forecast,indent=2))
'''),md('''
## Explicit final evaluation and export

`test_start=None` intentionally leaves the final test empty. Configure a genuinely new untouched period, prepare a **new data artifact directory**, keep development splits unchanged, and freeze all choices before enabling the test below. Do not reuse the older August/September test data already inspected in this project. An arrival model is not automatically connected to Telegram.
'''),code('''
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    if model is None or calibrator is None or not DATA.test_start or not manifests["test"]:
        raise RuntimeError("Require frozen model/calibrator and a new complete test partition")
    test_records = fixed_locations(label_store,manifests["test"],EVALUATION_LOCATIONS,SEED)
    test_data = ArrivalDataset(store,label_store,manifests["test"],norm,records=test_records)
    lp, yt = predict(model,test_data)
    test_probs = calibrated_probs(lp,calibrator)
    result = metrics(test_probs,yt)
    (RUN_DIR / "final_test_metrics.json").write_text(json.dumps(result,indent=2))
    plot_evaluation(test_probs,yt,result)
if model is not None and calibrator is not None:
    export = dict(model_config=model.config,normalization=norm,data_config=vars(DATA),
                  channels=CHANNELS,classes=CLASSES,calibrator=calibrator,
                  manifest_hash=prepared["manifest_hash"],task="arrival_at_currently_dry_location")
    (RUN_DIR / "inference_config.json").write_text(json.dumps(export,indent=2))
    print("Saved best.pt, calibrator.json and inference_config.json in",RUN_DIR)
''')]

experiments=[md('''
# Rain-arrival experiments

Controlled comparisons of stacked CNN, ConvGRU, ConvLSTM, history length, input channels and categorical versus hazard output. Uses the **same data contract and complete-window anchors** as `rain_arrival_model.ipynb`.

Select the project `.venv` kernel. Run setup/preparation once. The experiment sweep is opt-in, and each run has a separate directory. No radar-frame model, bot settings or old held-out data are modified.
'''),setup,contract,data_config,fixed,
md('''
## Experiment matrix and controls

All runs use the same source normalization, balanced training sampler, fixed natural-frequency validation locations, update budget and model-selection criterion. `max_history=12` keeps anchors eligible for every history condition. Compare compute time as well as validation skill; fewer recurrent parameters do not guarantee faster or better forecasts.

Start with CNN versus GRU. Enable one family of ablations at a time. Repeat promising comparisons across seeds before treating small differences as meaningful. A cropped local model may lack upstream context for long-range arrivals; crop-size/multiscale context is a follow-on study, requiring a new shared dataset definition.
'''),code('''
EXPERIMENTS = [
    dict(name="gru_recovery_baseline",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.0),
    dict(name="gru_recovery_hard10",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.1),
    dict(name="gru_recovery_hard10_curriculum",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.1,hard_negative_warmup=5000,hard_negative_ramp=5000),
    {'name': 'gru_t6_depth2_spatial_baseline', 'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'recurrent_layers': 2, 'pooling': 'spatial', 'hard_negative_fraction': 0.0},
    {'name': 'gru_t6_depth2_spatial_hardnegative', 'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'recurrent_layers': 2, 'pooling': 'spatial', 'hard_negative_fraction': 0.5},
    {'name': 'gru_t6_depth2_spatial_change_baseline', 'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'recurrent_layers': 2, 'pooling': 'spatial_change', 'hard_negative_fraction': 0.0},
    {'name': 'gru_t6_depth2_spatial_change_hardnegative', 'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'recurrent_layers': 2, 'pooling': 'spatial_change', 'hard_negative_fraction': 0.5},
    dict(name="gru_t6_negative_baseline",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.0),
    dict(name="gru_t6_hardnegative",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.5),
    {'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'pooling': 'global', 'recurrent_layers': 2, 'encoder_extra_layers': 0, 'name': 'gru_t6_depth2'},
    {'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'pooling': 'global', 'recurrent_layers': 3, 'encoder_extra_layers': 0, 'name': 'gru_t6_depth3'},
    {'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'pooling': 'global', 'recurrent_layers': 1, 'encoder_extra_layers': 2, 'name': 'gru_t6_encoder_deep'},
    {'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 96, 'pooling': 'spatial', 'recurrent_layers': 1, 'encoder_extra_layers': 0, 'name': 'gru_t6_wide96_spatial'},
    {'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 64, 'pooling': 'spatial', 'recurrent_layers': 2, 'encoder_extra_layers': 0, 'name': 'gru_t6_depth2_spatial'},
    {'kind': 'gru', 'history': 6, 'channels': (0, 1, 2, 3, 4, 5, 6), 'output': 'categorical', 'hidden_dim': 96, 'pooling': 'spatial', 'recurrent_layers': 2, 'encoder_extra_layers': 0, 'name': 'gru_t6_wide96_depth2_spatial'},
    dict(name="gru_t6_wide96",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hidden_dim=96,pooling="global"),
    dict(name="gru_t6_spatial",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hidden_dim=64,pooling="spatial"),
    dict(name="cnn_t6_all",kind="cnn",history=6,channels=tuple(range(7)),output="categorical"),
    dict(name="gru_t6_all",kind="gru",history=6,channels=tuple(range(7)),output="categorical"),
    dict(name="gru_t3_all",kind="gru",history=3,channels=tuple(range(7)),output="categorical"),
    dict(name="gru_t9_all",kind="gru",history=9,channels=tuple(range(7)),output="categorical"),
    dict(name="gru_t12_all",kind="gru",history=12,channels=tuple(range(7)),output="categorical"),
    dict(name="gru_t6_radar",kind="gru",history=6,channels=(0,),output="categorical"),
    dict(name="gru_t6_wind",kind="gru",history=6,channels=(0,3,4),output="categorical"),
    dict(name="gru_t6_physical",kind="gru",history=6,channels=(0,1,2,3,4),output="categorical"),
    dict(name="lstm_t6_all",kind="lstm",history=6,channels=tuple(range(7)),output="categorical"),
    dict(name="gru_t6_hazard",kind="gru",history=6,channels=tuple(range(7)),output="hazard"),
]
display(pd.DataFrame(EXPERIMENTS))
RUN_EXPERIMENTS = False
ENABLED = {"gru_recovery_baseline","gru_recovery_hard10","gru_recovery_hard10_curriculum"}
SEEDS = [67]  # Extend for repeatability after the first paired comparison.
UPDATE_BUDGET = 30000
SWEEP_DIR = PROJECT_ROOT / "models/arrival/experiments_hardnegative_recovery_v1"
'''),code('''
if RUN_EXPERIMENTS:
    SWEEP_DIR.mkdir(parents=True,exist_ok=True)
    for experiment in EXPERIMENTS:
        if experiment["name"] not in ENABLED: continue
        for seed in SEEDS:
            from arrival.run_paths import sweep_run_path, write_completed_result
            path = sweep_run_path(SWEEP_DIR / f"{experiment['name']}_seed{seed}")
            if (path / "result.json").exists():
                completed = json.loads((path / "result.json").read_text())
                if (completed["manifest_hash"] != prepared["manifest_hash"] or
                    completed["validation_records_hash"] != fingerprint(validation_records) or
                    json.dumps(completed["experiment"],sort_keys=True) != json.dumps(experiment,sort_keys=True)):
                    raise ValueError(f"Completed run configuration differs: {path}")
                print("Skipping completed run:", path.name)
                continue
            seed_everything(seed)
            training = make_dataset(experiment["history"],experiment["channels"])
            training.seed = seed
            training.hard_negative_fraction = experiment.get("hard_negative_fraction",0.0)
            training.hard_negative_warmup = experiment.get("hard_negative_warmup",0)
            training.hard_negative_ramp = experiment.get("hard_negative_ramp",0)
            validation = make_dataset(experiment["history"],experiment["channels"],"validation")
            candidate = ArrivalNet(history=experiment["history"],channels=len(experiment["channels"]),
                kind=experiment["kind"],output=experiment["output"],
                hidden_dim=experiment.get("hidden_dim",64),pooling=experiment.get("pooling","global"),
                recurrent_layers=experiment.get("recurrent_layers",1),
                encoder_extra_layers=experiment.get("encoder_extra_layers",0)).to(DEVICE)
            parameter_count = sum(p.numel() for p in candidate.parameters())
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            history = fit(candidate,training,validation,path,budget=UPDATE_BUDGET,
                          eval_every=250,patience=40,batch=16,seed=seed,selection="mean_ap")
            log_probs, truth = predict(candidate,validation)
            result = metrics(calibrated_probs(log_probs),truth)
            payload = dict(experiment=experiment,seed=seed,parameters=parameter_count,
                sample_counts=training.sample_counts,
                peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
                actual_updates=history[-1]["step"],elapsed_seconds=history[-1]["elapsed_seconds"],
                validation_metrics=result,manifest_hash=prepared["manifest_hash"],
                validation_records_hash=fingerprint(validation_records))
            np.savez_compressed(path / "validation_predictions.npz",probabilities=calibrated_probs(log_probs),labels=truth.numpy())
            write_completed_result(path, payload)
            del candidate
            if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print("Enable an experiment family and set RUN_EXPERIMENTS = False when ready.")
'''),md('''
## Compare results without reopening the final test

Compare mean validation average precision and 15/30/60-minute AP, alongside NLL, Brier scores, parameter count, elapsed time and peak GPU memory. The enabled comparisons change one factor: baseline 64-channel GRU with global pooling, 96-channel GRU with global pooling, or 64-channel GRU with a spatial 4x4 pooled head. All use six frames, seven channels, seed 67 and up to 30,000 updates. Training runs sequentially. These are fresh runs, not resumptions of v5. Calibration and notification replay are separate follow-up evaluations after selecting a candidate. The same maximum update budget is used; early stopping may yield different actual updates, which are reported. Validation examples share anchors/events; use day-level uncertainty below rather than pretending every crop is independent.
'''),code('''
from arrival.run_paths import completed_run
results = [json.loads(p.read_text()) for p in sorted(SWEEP_DIR.glob("*/result.json")) if completed_run(p.parent)]
rows = []
for r in results:
    if r["manifest_hash"] != prepared["manifest_hash"] or r["validation_records_hash"] != fingerprint(validation_records):
        raise ValueError("Experiment result uses a different comparison dataset")
    m = r["validation_metrics"]
    rows.append(dict(name=r["experiment"]["name"],seed=r["seed"],parameters=r["parameters"],
        updates=r["actual_updates"],seconds=r["elapsed_seconds"],NLL=m["nll"],macro_F1=m["macro_f1_all_13"],
        peak_gpu_mb=r.get("peak_gpu_allocated_mb"),
        mean_AP=np.mean([m["ranking"][str(h)]["average_precision"] for h in (15,30,60)]),
        **{f"AP_{h}":m["ranking"][str(h)]["average_precision"] for h in (15,30,60)},
        missed_arrivals=m["missed_arrivals"],false_arrivals=m["false_arrivals"],
        **{f"Brier_{h}":m["horizons"][str(h)]["brier"] for h in (15,30,60)}))
comparison = pd.DataFrame(rows)
if len(comparison):
    display(comparison.sort_values("mean_AP",ascending=False))
    comparison.to_csv(SWEEP_DIR / "comparison.csv",index=False)
    comparison.set_index("name")[["Brier_15","Brier_30","Brier_60"]].plot.bar(figsize=(12,4))
    plt.ylabel("Brier score (lower is better)"); plt.tight_layout(); plt.show()
else:
    print("No completed experiment results yet.")
'''),code('''
# Optional paired day-block bootstrap for two completed runs; no future labels used.
PAIR = ("gru_recovery_baseline_seed67", "gru_recovery_hard10_curriculum_seed67")
from arrival.run_paths import find_completed_run
paired_runs = [find_completed_run(SWEEP_DIR / name) for name in PAIR]
paths = [p / "validation_predictions.npz" for p in paired_runs if p is not None]
if len(paths)==2 and all(p.exists() for p in paths):
    predictions = [np.load(p) for p in paths]
    assert np.array_equal(predictions[0]["labels"],predictions[1]["labels"])
    days = np.array([r[0][:8] for r in validation_records]); unique_days = np.unique(days)
    y = predictions[0]["labels"] < 6
    errors = [(p["probabilities"][:,:6].sum(1)-y)**2 for p in predictions]
    daily_difference = np.array([(errors[1][days==day]-errors[0][days==day]).mean() for day in unique_days])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(daily_difference,(2000,len(daily_difference)),replace=True).mean(1)
    print("Equal-day mean Brier-30 difference (second minus first):",daily_difference.mean())
    print("Day-bootstrap 95% interval:",np.quantile(draws,[.025,.975]))
    print("Few days and storms spanning multiple days limit this interval; inspect event groups too.")
'''),md('''
## Calibrate the selected candidate only after comparisons

Select a candidate using validation. Fit its temperature/class bias on the separate natural-frequency calibration records. Do not rank models by calibration-fit performance. Hazard probabilities are constructed from survival products; both output types yield 13 probabilities summing to one. Use the main notebook's explicit new-test workflow or the frozen export below for final testing.
'''),code('''
SELECTED_RUN = "gru_recovery_baseline_seed67"  # Compare results before treating this as a winner.
from arrival.run_paths import find_completed_run
selected_path = find_completed_run(SWEEP_DIR / SELECTED_RUN)
RUN_CALIBRATION = False  # Enable only after choosing a successful candidate.
selected = None
calibrator = None
if RUN_CALIBRATION and selected_path is not None and (selected_path / "best.pt").exists():
    selected, state = load_model(selected_path / "best.pt",DEVICE)
    if state["validation_records_hash"] != fingerprint(validation_records) or state["norm"] != norm:
        raise ValueError("Selected checkpoint data contract differs")
    calibration = make_dataset(state["model_config"]["history"],tuple(state["channels"]),"calibration")
    log_probs, truth = predict(selected,calibration)
    from arrival.calibration import fit_projected_calibrator
    calibrator = fit_projected_calibrator(log_probs,truth)
    probabilities = calibrated_probs(log_probs,calibrator)
    result = metrics(probabilities,truth)
    plot_evaluation(probabilities,truth,result)
    (selected_path / "calibrator.json").write_text(json.dumps(calibrator,indent=2))

    import hashlib
    calibrator['checkpoint_sha256'] = hashlib.sha256((selected_path/'best.pt').read_bytes()).hexdigest()
    calibrator['calibration_records_hash'] = fingerprint(calibration_records)
    (selected_path/'calibrator.json').write_text(json.dumps(calibrator,indent=2))
    validation = make_dataset(state['model_config']['history'],tuple(state['channels']),'validation')
    validation_lp, validation_y = predict(selected,validation)
    comparison = {'raw':metrics(calibrated_probs(validation_lp),validation_y),
                  'projected':metrics(calibrated_probs(validation_lp,calibrator),validation_y)}
    (selected_path/'calibration_validation_metrics.json').write_text(json.dumps(comparison,indent=2))
    np.savez_compressed(selected_path/'calibration_predictions.npz',log_probs=log_probs.numpy(),labels=truth.numpy())
    display(pd.DataFrame([dict(method=name,horizon=h,AP=r['average_precision'],
        brier=m['horizons'][int(h)]['brier']) for name,m in comparison.items() for h,r in m['ranking'].items()]))
    plot_evaluation(calibrated_probs(validation_lp,calibrator),validation_y,comparison['projected'])
    print('Winner calibration ready for offline notification replay:',selected_path)

else:
    print("Complete/select a run first.")
'''),code('''
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    if selected is None or calibrator is None or not DATA.test_start or not manifests["test"]:
        raise RuntimeError("Require a frozen selected model, calibrator and new untouched test period")
    test_records = fixed_locations(label_store,manifests["test"],EVALUATION_LOCATIONS,SEED)
    test_data = ArrivalDataset(store,label_store,manifests["test"],norm,
        history=state["model_config"]["history"],channels=tuple(state["channels"]),records=test_records)
    lp, yt = predict(selected,test_data)
    final_probabilities = calibrated_probs(lp,calibrator)
    final_metrics = metrics(final_probabilities,yt)
    (selected_path / "final_test_metrics.json").write_text(json.dumps(final_metrics,indent=2))
    plot_evaluation(final_probabilities,yt,final_metrics)
if selected is not None and calibrator is not None:
    export = dict(model_config=state["model_config"],channels=state["channels"],normalization=norm,
        data_config=vars(DATA),classes=CLASSES,calibrator=calibrator,manifest_hash=prepared["manifest_hash"])
    (selected_path / "inference_config.json").write_text(json.dumps(export,indent=2))
'''),md('''
## Explicitly deferred

No architecture/loss result is assumed superior in advance. Neither notebook deploys to Telegram, generates predicted radar images, changes rain-ending rules, or uses the old inspected test set to select a model. Dense optical flow and crop-size/multiscale context are follow-on experiments if the global translation baseline and history ablations identify a need.
''')]

main.extend([md('## Calibration experiment: independent curves with consistent horizons\n\nFits twelve cumulative logistic curves on calibration data, then applies least-squares isotonic projection across horizons. Tests probability error and ranking on held-out calibration dates and development validation. Projection prevents inconsistent horizons but may alter ranking. Uses the saved v5 checkpoint prediction caches; no training or deployment. Results are written to `reports/arrival_v5_calibration/projection_report.md`.'), code('# Standalone v5 experiment: uses cached predictions, never retrains the model.\nimport sys, subprocess\nfrom pathlib import Path\nPROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/arrival").exists())\nresult = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/experiment_arrival_projection.py")],\n                        cwd=PROJECT_ROOT, capture_output=True, text=True)\nprint(result.stdout)\nif result.returncode:\n    raise RuntimeError(result.stderr)\n')])

experiments[0]["source"] += ['\n', '\n', 'Expanded sweep: nine enabled models, including 2/3 recurrent layers, two extra stride-1 encoder convolutions, and combined depth/width/spatial heads. All retain six inputs, seven channels, batch 16 and a 30,000-update maximum. Completed matching runs are skipped; incomplete checkpoints are protected and require a new run directory or explicit recovery. Run models sequentially with other GPU notebook kernels shut down. Depth increases runtime and memory; parameter counts and peak allocated memory are recorded. No automatic batch-size changes are made, preserving comparisons.']

experiments.insert(next(i for i,c in enumerate(experiments) if "SELECTED_RUN =" in "".join(c["source"]))+1,code("# Run after the winner calibration cell; offline only, no notifications sent.\nRUN_NOTIFICATION_REPLAY = False  # Set True to measure the new winner's alert performance.\nif RUN_NOTIFICATION_REPLAY:\n    import subprocess, sys\n    if globals().get('selected_path') is None:\n        raise RuntimeError('Run the selection/calibration cell first and select a completed run.')\n    if not (selected_path / 'calibrator.json').is_file():\n        raise RuntimeError(f'No saved calibration for {selected_path.name}. Set RUN_CALIBRATION = False in the preceding cell, run it to completion, then retry replay. No retraining is needed.')\n    subprocess.run([sys.executable,str(PROJECT_ROOT/'scripts/replay_arrival_notifications.py'),\n                    '--run',str(selected_path),'--calibration',str(selected_path/'calibrator.json'),\n                    '--output',str(PROJECT_ROOT/'reports'/f'arrival_replay_{selected_path.name}')],\n                   cwd=PROJECT_ROOT,check=True)\n"))


# Deterministic paired sweep configuration.
for c in experiments:
    source="".join(c["source"])
    if source.startswith("EXPERIMENTS = ["):
        c["source"]=['EXPERIMENTS = [\n', '    dict(name="gru_deterministic_baseline",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.0),\n', '    dict(name="gru_deterministic_hard10",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.1),\n', ']\n', 'display(pd.DataFrame(EXPERIMENTS))\n', 'RUN_EXPERIMENTS = False\n', 'ENABLED = {e["name"] for e in EXPERIMENTS}\n', 'SEEDS = [67, 71, 79]\n', 'UPDATE_BUDGET = 40000\n', 'SWEEP_DIR = PROJECT_ROOT / "models/arrival/experiments_deterministic_v1"\n']
    if 'import sys, json' in source:
        c["source"]=['import os\n', 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n', 'import sys, json\n', 'from pathlib import Path\n', 'from datetime import datetime\n', 'import numpy as np\n', 'import pandas as pd\n', 'import torch\n', 'import matplotlib.pyplot as plt\n', 'PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/data_processing").exists())\n', 'if str(PROJECT_ROOT / "src") not in sys.path: sys.path.insert(0, str(PROJECT_ROOT / "src"))\n', 'from arrival.data import (DataConfig, FrameStore, LabelStore, ArrivalDataset, prepare,\n', '                          fixed_locations, CLASSES, CHANNELS, fingerprint)\n', 'from arrival.models import ArrivalNet\n', 'from arrival.training import (seed_everything, fit, predict, load_model, metrics,\n', '                              fit_calibrator, calibrated_probs, persistence, motion_baseline)\n', 'from arrival.plots import plot_history, plot_evaluation, plot_example\n', 'SEED = 67\n', 'seed_everything(SEED,deterministic=True)\n', 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n', 'print("Device:", DEVICE, "| torch:", torch.__version__)\n', 'print("Channel order:", CHANNELS)']
    if 'if RUN_EXPERIMENTS:' in source:
        c["source"]=['if RUN_EXPERIMENTS:\n', '    SWEEP_DIR.mkdir(parents=True,exist_ok=True)\n', '    for experiment in EXPERIMENTS:\n', '        if experiment["name"] not in ENABLED: continue\n', '        for seed in SEEDS:\n', '            from arrival.run_paths import sweep_run_path, write_completed_result\n', '            path = sweep_run_path(SWEEP_DIR / f"{experiment[\'name\']}_seed{seed}")\n', '            if (path / "result.json").exists():\n', '                completed = json.loads((path / "result.json").read_text())\n', '                if (not completed.get("deterministic",False) or completed.get("update_budget") != UPDATE_BUDGET or\n', '                    completed["manifest_hash"] != prepared["manifest_hash"] or\n', '                    completed["validation_records_hash"] != fingerprint(validation_records) or\n', '                    json.dumps(completed["experiment"],sort_keys=True) != json.dumps(experiment,sort_keys=True)):\n', '                    raise ValueError(f"Completed run configuration differs: {path}")\n', '                print("Skipping completed run:", path.name)\n', '                continue\n', '            seed_everything(seed,deterministic=True)\n', '            training = make_dataset(experiment["history"],experiment["channels"])\n', '            training.seed = seed\n', '            training.hard_negative_fraction = experiment.get("hard_negative_fraction",0.0)\n', '            training.hard_negative_warmup = experiment.get("hard_negative_warmup",0)\n', '            training.hard_negative_ramp = experiment.get("hard_negative_ramp",0)\n', '            validation = make_dataset(experiment["history"],experiment["channels"],"validation")\n', '            candidate = ArrivalNet(history=experiment["history"],channels=len(experiment["channels"]),\n', '                kind=experiment["kind"],output=experiment["output"],\n', '                hidden_dim=experiment.get("hidden_dim",64),pooling=experiment.get("pooling","global"),\n', '                recurrent_layers=experiment.get("recurrent_layers",1),\n', '                encoder_extra_layers=experiment.get("encoder_extra_layers",0)).to(DEVICE)\n', '            parameter_count = sum(p.numel() for p in candidate.parameters())\n', '            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()\n', '            history = fit(candidate,training,validation,path,budget=UPDATE_BUDGET,\n', '                          eval_every=250,patience=40,batch=16,seed=seed,selection="mean_ap",deterministic=True)\n', '            log_probs, truth = predict(candidate,validation)\n', '            result = metrics(calibrated_probs(log_probs),truth)\n', '            payload = dict(experiment=experiment,seed=seed,parameters=parameter_count,\n', '                sample_counts=training.sample_counts,deterministic=True,update_budget=UPDATE_BUDGET,\n', '                peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,\n', '                actual_updates=history[-1]["step"],elapsed_seconds=history[-1]["elapsed_seconds"],\n', '                validation_metrics=result,manifest_hash=prepared["manifest_hash"],\n', '                validation_records_hash=fingerprint(validation_records))\n', '            np.savez_compressed(path / "validation_predictions.npz",probabilities=calibrated_probs(log_probs),labels=truth.numpy())\n', '            write_completed_result(path, payload)\n', '            del candidate\n', '            if torch.cuda.is_available(): torch.cuda.empty_cache()\n', 'else:\n', '    print("Enable an experiment family and set RUN_EXPERIMENTS = False when ready.")']
    if 'PAIR =' in source:
        c["source"]=['# Optional paired day-block bootstrap for two completed runs; no future labels used.\n', 'PAIR = ("gru_deterministic_baseline_seed67", "gru_deterministic_hard10_seed67")\n', 'from arrival.run_paths import find_completed_run\n', 'paired_runs = [find_completed_run(SWEEP_DIR / name) for name in PAIR]\n', 'paths = [p / "validation_predictions.npz" for p in paired_runs if p is not None]\n', 'if len(paths)==2 and all(p.exists() for p in paths):\n', '    predictions = [np.load(p) for p in paths]\n', '    assert np.array_equal(predictions[0]["labels"],predictions[1]["labels"])\n', '    days = np.array([r[0][:8] for r in validation_records]); unique_days = np.unique(days)\n', '    y = predictions[0]["labels"] < 6\n', '    errors = [(p["probabilities"][:,:6].sum(1)-y)**2 for p in predictions]\n', '    daily_difference = np.array([(errors[1][days==day]-errors[0][days==day]).mean() for day in unique_days])\n', '    rng = np.random.default_rng(SEED)\n', '    draws = rng.choice(daily_difference,(2000,len(daily_difference)),replace=True).mean(1)\n', '    print("Equal-day mean Brier-30 difference (second minus first):",daily_difference.mean())\n', '    print("Day-bootstrap 95% interval:",np.quantile(draws,[.025,.975]))\n', '    print("Few days and storms spanning multiple days limit this interval; inspect event groups too.")']
    if 'SELECTED_RUN =' in source:
        c["source"]=['SELECTED_RUN = "gru_deterministic_baseline_seed67"  # Compare results before treating this as a winner.\n', 'from arrival.run_paths import find_completed_run\n', 'selected_path = find_completed_run(SWEEP_DIR / SELECTED_RUN)\n', 'RUN_CALIBRATION = False  # Enable only after choosing a successful candidate.\n', 'selected = None\n', 'calibrator = None\n', 'if RUN_CALIBRATION and selected_path is not None and (selected_path / "best.pt").exists():\n', '    selected, state = load_model(selected_path / "best.pt",DEVICE)\n', '    if state["validation_records_hash"] != fingerprint(validation_records) or state["norm"] != norm:\n', '        raise ValueError("Selected checkpoint data contract differs")\n', '    calibration = make_dataset(state["model_config"]["history"],tuple(state["channels"]),"calibration")\n', '    log_probs, truth = predict(selected,calibration)\n', '    from arrival.calibration import fit_projected_calibrator\n', '    calibrator = fit_projected_calibrator(log_probs,truth)\n', '    probabilities = calibrated_probs(log_probs,calibrator)\n', '    result = metrics(probabilities,truth)\n', '    plot_evaluation(probabilities,truth,result)\n', '    (selected_path / "calibrator.json").write_text(json.dumps(calibrator,indent=2))\n', '\n', '    import hashlib\n', "    calibrator['checkpoint_sha256'] = hashlib.sha256((selected_path/'best.pt').read_bytes()).hexdigest()\n", "    calibrator['calibration_records_hash'] = fingerprint(calibration_records)\n", "    (selected_path/'calibrator.json').write_text(json.dumps(calibrator,indent=2))\n", "    validation = make_dataset(state['model_config']['history'],tuple(state['channels']),'validation')\n", '    validation_lp, validation_y = predict(selected,validation)\n', "    comparison = {'raw':metrics(calibrated_probs(validation_lp),validation_y),\n", "                  'projected':metrics(calibrated_probs(validation_lp,calibrator),validation_y)}\n", "    (selected_path/'calibration_validation_metrics.json').write_text(json.dumps(comparison,indent=2))\n", "    np.savez_compressed(selected_path/'calibration_predictions.npz',log_probs=log_probs.numpy(),labels=truth.numpy())\n", "    display(pd.DataFrame([dict(method=name,horizon=h,AP=r['average_precision'],\n", "        brier=m['horizons'][int(h)]['brier']) for name,m in comparison.items() for h,r in m['ranking'].items()]))\n", "    plot_evaluation(calibrated_probs(validation_lp,calibrator),validation_y,comparison['projected'])\n", "    print('Winner calibration ready for offline notification replay:',selected_path)\n", '\n', 'else:\n', '    print("Complete/select a run first.")']
experiments.insert(next(i for i,c in enumerate(experiments) if "comparison = pd.DataFrame(rows)" in "".join(c["source"]))+1,code('# Compare paired seeds; do not choose a method from its single best seed.\nif len(comparison):\n    score_columns = ["mean_AP","AP_15","AP_30","AP_60"]\n    display(comparison.groupby("name")[score_columns].agg(["count","mean","std"]))\n    paired = comparison.pivot(index="seed",columns="name",values="mean_AP")\n    if {"gru_deterministic_baseline","gru_deterministic_hard10"}.issubset(paired.columns):\n        paired["hard10_minus_baseline"] = paired["gru_deterministic_hard10"]-paired["gru_deterministic_baseline"]\n        display(paired)\n        paired.to_csv(SWEEP_DIR/"paired_seed_comparison.csv")\n        print("Complete pairs:",paired.dropna().shape[0],"of",len(SEEDS))\n    print("Calibration/replay still needed before notification conclusions. Three seeds are not a final independent test.")\n'))
experiments[0]["source"]=['# Deterministic baseline versus 10% hard-negative sampling\n\nRestart the kernel, then Run All. Only two configurations are enabled, each with seeds 67, 71 and 79 (six fresh runs total). Training is disabled by default; explicitly enable it only when ready. Both use six frames, seven channels, the original 64-channel GRU, batch 16, and up to 40,000 updates with the same early stopping. Prior experiments remain in their old directories. Completed matching new runs are skipped; interrupted runs are preserved in retry folders.\n\nStrict deterministic execution is enabled before model initialization and inside training. Unsupported deterministic operations will raise rather than silently fall back. This improves same-device repeatability, not guarantees across hardware or framework versions. Calibration and replay remain disabled until the paired results have been reviewed.\n']

if __name__=='__main__':
    save('rain_arrival_model.ipynb',main)
    save('rain_arrival_experiments.ipynb',experiments)
    print('Wrote both arrival notebooks.')
