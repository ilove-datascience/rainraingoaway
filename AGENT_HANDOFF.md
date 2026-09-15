# RainRainGoAway — agent handoff

Updated 2026-09-14, Asia/Singapore. This document summarizes the long development conversation and local artifacts. It is context, not authorization to train, deploy, push, send notifications, or overwrite files. Inspect current code and saved artifacts before acting; the user may run notebooks between turns.

## Immediate next task / current state

The user asked for a clear next experiment after hard-negative sampling gave mixed results. **`src/rain_arrival_experiments.ipynb` is now configured for six fresh deterministic runs:**

- `gru_deterministic_baseline`, seeds 67, 71, 79.
- `gru_deterministic_hard10`, seeds 67, 71, 79.
- Original 64-channel single-layer ConvGRU, global pooling, six frames, seven channels, batch 16.
- Maximum 40,000 updates; validation every 250; early stopping after 40 checks without improvement in mean 15/30/60-minute AP.
- `RUN_EXPERIMENTS = True`; strict deterministic settings before initialization and inside training.
- Output: `models/arrival/experiments_deterministic_v1/`.
- Comparison includes individual results, mean/std across seeds and paired hard10-minus-baseline mean AP.
- Calibration and notification replay are disabled pending review of results.
- Last instruction to user: restart kernel and Run All in the **experiments** notebook. We have configured, not launched, these six runs. Check disk before assuming they remain unrun.

The most recent edit was by `scripts/configure_deterministic_arrival_sweep.py`. Notebook and generator Python syntax were checked. Last full regression suite before this notebook-only configuration: **82 tests passed**. Do not imply six full trainings were tested.

## User preferences and recurring confusion

- User wants action and simple, direct explanations, not repeated permission requests or endless proposed experiments.
- They usually run long training themselves in VS Code and say “ran” or “done.” Read saved artifacts rather than guessing.
- Explain exactly which notebook/cell to run, whether training is needed, and which flags to change. They repeatedly opened the main model notebook when intending the experiments notebook.
- Do not rerun all old architectures when enabling a small experiment. Preserve completed runs and model files.
- Kernel restart clears imported code/memory, **not checkpoints**. Stale imports caused old sampler/metrics code to run despite edited files.
- Only one GPU notebook kernel should train at a time. Shutting down the other kernel releases its retained models.
- Updates are optimizer steps, each processing batch 16. Dataset has 10,400 train anchors ×32 samples =332,800 nominal samples, ~20,800 steps per epoch. Epoch zero is expected for shorter runs; sampled “epochs” do not exhaust every possible spatial location.
- Calibration adjusts probabilities; notification thresholds are the gate. Calibration does not retrain model weights.
- Better AP does not automatically mean better notification performance. Always distinguish sampled prediction scores, event replay and independent testing.
- User believes 1–5-pixel echoes are often noise. Never silently delete all such target events or count nearby rain as a correct exact-location warning.

## Environment / practical tooling

- Workspace: `C:\Users\Jacobs laptop\rainraingoaway`; shell PowerShell.
- `.venv\Scripts\python.exe`: PyTorch 2.12.0+cu126, CUDA available during this work; numpy/scipy/pandas/PIL/matplotlib available. In this session executable launch generally needed `require_escalated` because sandbox launch failed.
- Bundled Python usable for stdlib JSON/AST/file edits without escalation:
  `C:\Users\Jacobs laptop\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
- `nbformat` was missing in project venv. Notebook validation used JSON + AST (exclude IPython magic for old ConvLSTM notebook), plus functional tests/smokes.
- Git has many modified/untracked notebooks, checkpoints, reports and scripts. Some model/notebook modifications are user-generated training outputs. Do not reset, clean or overwrite them. No current sweep work has been pushed in this part of the conversation.
- No AGENTS.md was found by recent repo search. Do not assume this handoff replaces other applicable instructions.
- VS Code had “Could not register service worker / invalid document state” opening notebooks. File JSON was valid. Suggested closing VS Code, ending remaining Code.exe after saving/not training, and renaming `%APPDATA%\Code\Service Worker` if present. Do not conflate that webview issue with notebook/model errors.

## Two separate model workstreams

### Existing radar-frame +5/+10 model

- Notebook: `src/test_multimodal_convlstm_long.ipynb`.
- ConvLSTM produces future radar intensity/maps, using three input frames. +10 is autoregressive rollout with persisted latest environmental channels.
- Earlier work fixed loss averaging, calibration/evaluation issues, added +10 metrics and both-horizon visualizations. Loss includes BCE +0.5 Dice for rain head, Smooth L1 rain intensity and dry terms. A positive-class weight tensor of 20 existed but should not be blindly activated.
- +10 auxiliary loss was 0.25; proposed comparisons were not automatically implemented.
- User had trained to about epoch 30, with a best checkpoint around epoch29 in earlier diagnostics. Recheck current checkpoint rather than assuming unchanged.
- Later implemented **overlapping training windows** in this actual notebook: one five-minute stride, whole-date 75/15/10 partitions before windows, no windows across missing-frame gaps. Validation/test remain nonoverlapping. Day split differs from the previous sample-index split; old scores cannot be treated as a matched baseline.
- `create_samples(..., stride=None)` retains old nonoverlap default. Dataset `overlapping_training=True` selects new behavior, exposes `split_indices`/`frame_manifest`; notebook saves `models/overlapping_long_manifest.json` when run.
- Fixed shared-frame input cleanup to clone on write, preventing overlap windows from mutating other samples or future targets.
- More windows increase epoch duration. Compare training by optimizer updates, not epochs.
- Existing production bot forecast remains +5. Arrival work has NOT replaced it, generated radar images, or changed rain-ending behavior.

### New local rain-arrival model

- Main notebook: `src/rain_arrival_model.ipynb` (v5 training plus calibration experiments).
- Experiment notebook: `src/rain_arrival_experiments.ipynb` (currently deterministic paired sweep above).
- Shared modules: `src/arrival/data.py`, `models.py`, `training.py`, `calibration.py`, `ranking.py`, `plots.py`, `inference.py`, `run_paths.py`.
- Predicts first arrival at a **currently dry** centre within 60 minutes, in twelve 5-minute bins plus no-arrival (13 classes).
- Main model: CNN encoder (16 then32 channels, stride2 each), ConvGRU hidden64, pooling/head to13 classes. Optional hazard head outputs12 hazards then converts to13 probabilities.
- Inputs: 6×7×64×64, target at [32,32], history t−25 through t.
- Channel order is **radar, temperature, humidity, wind_u, wind_v, station_mask, distance**. No land-use input in current arrival runs.
- Rain at target: ≥5 of 25 centre-neighbourhood pixels exceed .01. Currently wet targets excluded. Every one of 12 future radar observations must exist; missing future is not no-arrival.
- Radar exact-source palette values are ordinal, not mm/hour. Existing conservative cleaning remains.
- Shared eligible anchors reserve max history12 for consistent history comparisons: complete t−55 through t+60 inside one day-based partition, exact5min continuity.
- Default development end is exclusive `2026-08-26`, test_start=None. Chronological 75/15/10 of available dates for train/validation/calibration. Train anchors10,400; validation1,998; calibration1,968 in saved data_v1.
- Normalization fits training-only weather frames (not validation); source/channel metadata checked. First preparation caches labels and weather.
- Weather CSV must be exact frame time with observation timestamps no later than input frame. Historical publication/ingestion times unavailable; do not claim proven real-time availability.
- `models/arrival/data_v1/prepared.json`, `evaluation_locations.json`, `baselines.json` hold shared data artifacts. Validation/calibration evaluation each samples256 fixed anchors ×16 locations =4,096 records, naturally sampled.
- Natural validation has185 within-hour arrivals /4,096; calibration77 /4,096. Do not rebalance these evaluations to manufacture accuracy.
- Training sampler chooses equally among four global groups: 0–15,15–30,30–60,no-arrival, then an eligible anchor and location. Earlier v1/v2 sampler balanced only within each timestamp and was flawed.
- Dynamic sampling is deterministic by seed/epoch/index; num_workers=0 preserves per-update curriculum state.

## Versioned radar and production-related earlier work

- `src/data_processing/radar_codec.py`: legacy_nearest30_v1 preserves old decoder by default; nea_exact33_v1 uses33 exact official colours, transparent zero, rejects unknown opaque colours. Exact values are rank-like ceil((index+1)/33*100)/100, not physical rain rates.
- Source category helper returns Light, Light to Moderate, Moderate, Moderate to Heavy, Heavy.
- `data_loading.py` has versioned cache separation and optional `build_env_data(...,as_of=...)`; old callers retain behavior when omitted.
- `model_contract.py` sidecars record decoder/preprocessing/checkpoint and normalization hashes. Never feed source-exact values to legacy-trained weights without appropriate retraining/versioning.
- Production mask in `src/telegram_code/forecast_mask.py`: probability .55, min component25 at last inspection; some older notebook candidates differ.
- Actual radar requests include user location and intensity. Forecast notifications include predicted radar/intensity. Text was standardized to distinguish observed versus forecast information.
- Notification requests: one predicted-rain start per episode; follow-up on predicted clearing or observed ending; no15-minute cooldown or extra confirmation/intensity-change automatic alerts. Preserve existing policy unless explicitly changing it.
- Startup CREATE TABLE was explicitly removed after MySQL user weathercat lacked CREATE privilege. Do not reintroduce automatic schema creation. SQL provisioning is separate.
- Prior scraper work skips fallback downloads/cache construction to reduce delays; forecast timesteps have freshness/fallback logic. Inspect current code for exact current implementation, as earlier conversation details are abbreviated here.
- Existing radar notification replay: `reports/notification_replay/report.md`, 3,262 ticks25 locations, 1,844 missing inputs, Aug9–26. It diagnosed weak onset warnings and premature clearing; +10 replay hypothetical. Do not confuse this with arrival-model replay.
- Plans and older diagnostics: `STEP4_MODEL_IMPROVEMENTS.md`, `reports/source_check`, `reports/heldout_misses`, `reports/small_echo_audit`. The step4 document’s early “planned” status is stale relative to implemented arrival notebooks. User’s supplied arrival proposal was incorporated there.

## Training, checkpoints and interruption handling

- `fit` uses AdamW lr.001, decay1e-4, gradient norm clip1, batch16, update budget, validation every250. Sample-weighted training loss.
- Recent selection: mean AP across cumulative15/30/60, not NLL. Saves `best.pt` and separate `best_nll.pt`, plus `history.json`.
- Training checkpoints include config/norm/seed/manifest hashes, sampling details. Old checkpoints lack optimizer and data-loader state. **An interrupted old run cannot be exactly resumed.**
- `run_paths.py` preserves interrupted runs in original folders and finds fresh `_attempt2`, `_attempt3`, etc. Completed runs are reused only with result.json, best.pt and validation_predictions.npz plus basic metadata. Searches numbered attempts even if earlier attempt numbers are absent.
- `write_completed_result` publishes result.json atomically only after checkpoint/predictions exist. Notebook validates data/config before skipping. A malformed or partial result is not silently counted complete.
- Comparison and calibration use completed retry resolution. A fresh retry restarts from scratch; do not tell user it resumes optimizer steps.
- Result files: `result.json` (metrics/config/counts/runtime/GPU memory), `history.json`, `best.pt`, `best_nll.pt`, `validation_predictions.npz`. Calibrator/export files added only when those cells run.
- Strict reproducibility added to `seed_everything(...,deterministic=True)` and `fit(...,deterministic=True)`. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization. cuDNN benchmark false; deterministic flag true; torch deterministic algorithms true. Unsupported ops raise. Does not promise identical results across devices/framework versions.
- Short fixed-synthetic-input GPU audit: default settings repeated40 updates with max weight difference2.08e-6, loss difference4.77e-7. Strict mode bit-identical. Evidence of nondeterminism, not proof of all historical divergence causes. Report: `reports/arrival_recovery_audit/repeatability.json`.

## Calibration methods and artifacts

1. Original temperature +13-class bias: improved probability error but degraded arrival ranking. Retained as `fit_legacy_calibrator`.
2. Shared increasing cumulative logistic map: preserves ranking and horizon consistency but markedly underconfident at15/30min. `fit_calibrator` still denotes this shared method; check call sites, do not assume it means newest method.
3. Independent positive-slope logistic map for each of12 cumulative horizons, followed by equal-weight least-squares isotonic projection across horizons. Supported in `src/arrival/calibration.py` (`fit_projected_calibrator`, `apply_projected_calibrator`, `project_cumulative`), dispatched by `calibrated_probs`.
   - Method marker: independent_12_cumulative_logistic_then_isotonic_v1.
   - Returns coherent13-class probabilities by differencing cumulative probabilities and adding final survival.
   - Projection may alter ranking; it is not strictly rank-preserving like the shared map.
   - Fit on calibration partition only; compare validation and day-out-of-fold calibration error. Do not present calibration-fit scores as independent performance.
- Ordered-intercept/shared-slope alternative was tested and gave no useful improvement; do not promote it.
- Scripts: `audit_arrival_calibration.py`, `experiment_arrival_projection.py`, `add_run_projection_cell.py`.
- Reports: `reports/arrival_v5_calibration/` contains audit.json, report.md, NEXT_STEPS.md, projection_experiment.json, projection_report.md, cached predictions and projected validation probabilities.
- V5 shared curve: top validation risk decile predicted1.0% vs observed12.0% at15min;4.0% vs18.3% at30;22.1% vs21.5% at60.
- Projected v5 validation AP49.6/46.6/32.3%, Brier .00988/.01858/.03719 at15/30/60; projection improved day-held-out calibration Brier too. Still development data.
- Calibration is distinct from notification threshold selection. Never treat .5 as automatically appropriate or threshold .3 as reliably “30% incident rate.”

## Completed run history and conclusions

### Main arrival runs: models/arrival/runs/

- v1/v2 `gru_t6_all_seed67_v1/v2`: 3k updates, best~1750. v2 checkpoint lacked sampler marker, indicating old training imports; do not consider it proof corrected sampler was active.
- v3: confirmed global_groups_v2 sampler. 3k updates, best2500. Validation AP17.2/15.9/10.8%. Better short-range ranking but weak.
- v4:10k updates, best9500, ~36.7min. Raw AP49.2/44.2/18.4%. Better than motion at15/30, weaker60.
- v5:30k updates, best27000, ~114min. Raw AP49.6/46.4/25.4%; projected AP49.6/46.6/32.3%. This remains the quieter long-range notification reference. Do not replace based on frame AP alone.

### Architecture sweep: models/arrival/experiments_architecture_v1/

All nine completed; original wide96 incomplete folder preserved, completed result in `_attempt2`. Same manifest/eval records. User increased max budget to40k. Best checkpoints selected by mean AP, with early stopping.

| Architecture | AP15 | AP30 | AP60 |
|---|---:|---:|---:|
| gru_t6_all | .505 | .452 | .233 |
| gru_t6_wide96 (attempt2) | .475 | .410 | .238 |
| gru_t6_depth2 | .540 | .436 | .181 |
| gru_t6_depth3 | .554 | .482 | .277 |
| gru_t6_encoder_deep | .471 | .438 | .264 |
| gru_t6_spatial | .541 | .534 | .334 |
| gru_t6_wide96_spatial | .473 | .523 | .353 |
| gru_t6_depth2_spatial | .634 | .582 | .361 |
| gru_t6_wide96_depth2_spatial | .498 | .531 | .363 |

Two-layer64 spatial head was best overall ranking (best24750,426,349 params,~2.31h run). Global pooling discards location information; spatial pooling retains a4×4 grid. Added depth/width alone was inconsistent.

Winner calibrated/projected AP .637/.578/.397, but notification precision at a fixed .3 threshold was worse than v5. Winner reports: `reports/arrival_replay_gru_t6_depth2_spatial_seed67/`.

### Hard-negative architecture sweep: models/arrival/experiments_hardnegative_v1/

Six completed, same data. Each of original GRU, two-layer spatial, two-layer spatial_change had baseline and50% hard negatives within no-arrival group. The new spatial_change head concatenates final recurrent features and final-minus-first features, then spatial pools; it is not optical flow or measured velocity.

| Architecture / sampling | AP15 | AP30 | AP60 |
|---|---:|---:|---:|
| Original / baseline | .539 | .458 | .163 |
| Original / hard50 | .089 | .077 | .031 |
| Spatial / baseline | .530 | .558 | .369 |
| Spatial / hard50 | .053 | .055 | .051 |
| Spatial change / baseline | .512 | .523 | .356 |
| Spatial change / hard50 | .050 | .046 | .064 |

**50% intervention failed across all three.** Counters verified intended group proportions; sampled32 hard-negative labels verified against actual future radar with no contradictions. Final losses2.36–2.38 near approximate constant group-prior entropy2.3835, consistent with weak discrimination, not definitive proof of one root cause. The automatically calibrated hard50 run is bad and must not be selected.

### Recovery sweep: models/arrival/experiments_hardnegative_recovery_v1/

| Run | AP15 | AP30 | AP60 | Updates / best |
|---|---:|---:|---:|---|
| gru_recovery_baseline_seed67 | .473 | .394 | .230 |14000 /4000|
| gru_recovery_hard10_seed67 | .484 | .458 | .219 |30250 /20250|
| gru_recovery_hard10_curriculum_seed67 | .525 | .469 | .191 |34750 /24750|

Warmup supposedly identical but diverged before hard negatives began. Nondeterministic execution is a confirmed possible contributor; other historical uncontrolled kernel state not reconstructable. Avoid causal claims that all gains came from sampling.

Recovery mechanisms implemented:
- Hard fraction .1 rather than .5.
- Curriculum optional: zero until5000updates, linear ramp to.1 by10000.
- Independent branch RNG avoids perturbing general-negative sample selection when hard branch is not chosen. Warmup samples match baseline in unit tests.
- Global arrival groups remain equally likely. Hard fraction .1 of no-arrival group ≈2.5% of all draws.
- Log effective fraction and counts; schedule saved in checkpoints.
- Hard candidate: currently near qualifying rain (Euclidean ≤5pixels to qualifying5×5 patch centre), complete future label12, eligible centred crop. General negative branch retained and can overlap hard pool.

All recovery runs were calibrated and replayed by `scripts/evaluate_arrival_recovery.py`. Artifacts are in **reports/arrival_recovery_audit/<run>/**, deliberately separate from model run folders (calibrator.json, calibrated_validation.json, forecasts.npz, results.json, report.md). Do not assume no calibration just because training folder lacks a calibrator.

## Notification replay semantics and findings

Main scripts:
- `replay_arrival_notifications.py --run PATH --calibration JSON --output DIR`
- `compare_arrival_alert_burden.py` (v5 vs spatial), `--recovery` (v5 +3 recovery runs)
- `audit_arrival_warning_consistency.py`
- `diagnose_arrival_spatial_misses.py`, `audit_arrival_negative_sampling.py`

Replay uses1,998 validation anchor times Aug1–16 in37 uninterrupted runs, nine synthetic crop-valid radar-pixel locations. Observed rain uses same cleaned-source five-of25 criterion. Forecasts only for currently dry points. Missing inputs split episodes, not dry weather. Onsets need full preceding horizon and warnings need full future horizon; exposure excludes right-censored intervals. Some correct alerts precede boundary onsets excluded from recall, so true-alert count can differ from warned-onset count. No scraping/delivery latency.

Proposed offline policy: one outstanding start warning until observed onset or horizon expiry; later alerts can reoccur after expiry. No predicted-clear notifications in arrival replay. This is **not production bot policy** and must not silently replace user-authorized episode rules.

Models are compared at fixed caps on false warnings per24h usable location coverage. Threshold grid is tuned on same development replay; frontier is optimistic. Equal caps do not guarantee exactly equal achieved false rates. Dense grid is descriptive, not certified global optimum. Onsets and locations are correlated. Do not claim production-user incident rates.

At60min and cap .25 false/location/day:

| Model | Warned onsets | Actual false/day | Median lead |
|---|---:|---:|---:|
| v5 |26/52|.237|15min|
| Two-layer spatial |18/52|.197|10min|
| Fresh recovery baseline |20/52|.237|10min|
| Immediate hard10 |23/52|.237|15min|
| Gradual hard10 |25/52|.237|15min|

At cap .5: v5 34/52; immediatehard10 33/52; gradual31/52. Immediatehard10 at cap .1 warns19 vs v5’s18, too small to establish superiority. Recommendation remained **keep v5 as reference**, do not deploy new candidates based on current evidence.

Spatial model helped shorter15min warnings under some moderate caps, but not a clear general winner. Reports: `reports/arrival_matched_burden/`, `reports/arrival_recovery_matched_burden/`.

### False-warning diagnosis

At60min/.3 threshold: v5 false12, spatial false23. Qualifying rain came within5pixels during horizon in8/12 and17/23; already that close at issue time in7/12 and15/23. Within10pixels counts10/12 and21/23. Do not treat these as correct forecasts or equate pixels with kilometres without verifying geometry.

Late rain is not main explanation: through15min after expiry, v5 11/12 still dry plus1unobserved; spatial21/23 still dry,1late,1unobserved.

Training audit256 sampledanchors: under current no-arrival sampling,2.32% of negatives near5pixels,5.92% near10;65.49% of sampled negative anchors have near5 candidates. This motivated hard-negative sampling, but failed50% runs show the intervention was not validated simply by motivation. Nearby rain is common; counts alone do not establish enrichment against matched correct warnings.

Two consecutive forecast confirmation was tested. v5 immediate threshold.3:26/52 onsets,12false,15minlead. Two-confirmation threshold.2:25/52,12false,12.5minlead. Mandatory confirmation not recommended from this test. Production unchanged.

Reports: `reports/arrival_warning_consistency/`, `reports/arrival_spatial_diagnosis/decision.md`, `reports/hardnegative_failure/diagnosis.json`, `reports/arrival_recovery_audit/summary.md`.

## Notebook maintenance / known limitations

- `scripts/build_arrival_notebooks.py` originally generated both notebooks but now contains accumulated update hooks. Many one-off patch scripts exist (e.g. prepare_winner_calibration, add_hard_negative_experiments, prepare_hardnegative_recovery, configure_deterministic_arrival_sweep).
- **Do not blindly run the generator or old patch scripts**: they can replace notebook outputs or reintroduce stale prose/settings. Inspect current notebook source and preserve user outputs. Prefer targeted cell edits; synchronize generator carefully if maintaining it.
- Some notebook explanatory prose from earlier sweeps is stale even when active settings are correct. Current code/flags/run metadata are authoritative. In particular, calibration targets were previously preset to candidates before results existed; current RUN_CALIBRATION=False prevents accidentally promoting them.
- Training artifact cache assumes source data unchanged; some prediction caches key checkpoint and records but not full source hashes. Invalidate/rebuild appropriately if input radar/weather files change.
- Existing per-instance lru_cache methods and frame-loading work can make first passes slow; avoid claiming a memory leak without measurements.
- For replay comparisons, load npz arrays into memory once, not inside nested threshold loops. A previous version repeatedly decompressed arrays and was fixed.
- A shared input pass with batching was used for recovery replays to avoid three redundant data passes.
- Deterministic mode is meaningful only on a fresh kernel with environment configured before CUDA; it does not retroactively make old artifacts deterministic.
- Full new untouched weather evaluation has not been done. Prior held-out periods were inspected repeatedly, so do not label them independent anymore. Keep final-test gates off until a genuinely reserved period and frozen settings are chosen.

## Verification and useful commands

```
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe scripts/check_arrival_repeatability.py
.venv\Scripts\python.exe scripts/evaluate_arrival_recovery.py
.venv\Scripts\python.exe scripts/compare_arrival_alert_burden.py --recovery
```

Be careful: the recovery evaluation script refits calibration and reruns inference; only run if needed. The user’s main training runs should normally be launched from the intended notebook after confirming flags. Do not execute the whole main notebook merely to inspect results.

Tests cover decoder contracts, notification behavior, overlapping windows, model variants, hazard probabilities, sampling fractions/labels, curriculum/warmup identity, calibration coherence, replay suppression/censoring and interrupted-run path handling. Latest full suite passed82tests. This is implementation verification, not proof of forecasting skill.

## What to do when user next says “ran”

1. Inspect `models/arrival/experiments_deterministic_v1/*/result.json`, history, checkpoint reproducibility metadata and paired comparison outputs.
2. Verify both models ×3seeds completed, same data hashes/budgets and deterministic settings. Count retries correctly; do not mix incomplete artifacts.
3. Report mean/variation and per-seed differences, not just the best seed/model. Distinguish early-stopped actual update counts from maximum budget.
4. Decide whether any gain is consistent enough to justify calibration/replay. Keep v5 reference and avoid another uncontrolled broad sweep.
5. Clearly state what actually ran and what is only configured. Never imply a calibrated model was deployed or that validation threshold tuning is a final test.
