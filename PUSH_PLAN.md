# Separate, reviewable pushes

Prepared from the current working tree. This is a plan, not a record of completed pushes.
Current branch: `main`. HEAD: `c9f2b5b`. Local HEAD matches the locally cached
`origin/main`; fetch before executing because the remote has not been refreshed.

## Execution approach

- Use a separate `codex/` topic branch and PR for each batch below. Do not push the
  entire working tree to `main`.
- Build each branch in an isolated checkout from the correct base, using explicit
  files or hunks. Keep this working tree intact, including user notebook changes.
- Merge prerequisite PRs first, then base dependent branches on updated `main`.
  Independent cute mode can proceed separately. This avoids cumulative PR diffs.
- Each commit must contain its own implementation and relevant tests. Use one
  commit per small batch; split larger batches into independently valid commits.
- Never use blanket staging. Inspect the staged name list and diff before every
  commit, and push only the named topic branch. No force pushes.
- Do not launch training, replay live notifications, or run notebook generators
  merely to prepare a push. Generators can overwrite notebooks.

## 1. Repository hygiene

Branch: `codex/repo-hygiene`
Commit: `chore: ignore local data and generated experiment artifacts`

Include `.gitignore`. Keep data, caches, logs, generated model directories,
notebook output images, and bulk reports excluded. Verify source files and tests
remain visible. Ignoring models does not untrack already committed checkpoints.

## 2. Radar decoding and data correctness

Branch: `codex/radar-data-contracts`
Commits:

1. `fix: version radar decoding and validate checkpoint contracts`
2. `feat: support overlapping windows with chronological frame splits`

Include:

- `src/data_processing/radar_codec.py`
- `src/data_processing/model_contract.py`
- `src/data_processing/data_loading.py`
- `src/data_processing/multimodal_radar_dataset.py`
- `src/main.py`
- `tests/test_radar_codec.py`
- `tests/test_overlapping_windows.py`
- The source-radar decoding, observed intensity categories, and error-handling
  hunks in `src/telegram_code/methods.py` and `notification_text.py`.

Split the shared data-loading files by hunk where needed; a whole-file stage
would otherwise mix decoder, windowing, and causal-weather changes. Include the
causal `as_of` weather filtering with its data contract. Preserve compatibility
with existing legacy checkpoints. Keep the forecast-mask extraction for batch 4.

Validation: decoder/contract/window tests, existing forecast fallback and policy
tests, and an offline model-load contract check against the intended artifacts.

## 3. Hidden cat message mode — independent feature

Branch: `codex/hidden-cute-mode`
Commit: `feat: add persistent hidden cat message mode`

Include only:

- `src/telegram_code/cute_mode.py`
- `src/telegram_code/telegram_bot.py`
- `tests/test_cute_mode.py`

Validate per-user toggling, restart persistence, captions and notification
delivery, and the absence of new buttons/menu entries. Keep private preference
databases ignored. Targeted cute-mode, delivery, and forecast-policy tests passed
16 tests during implementation; rerun the relevant checks on the isolated branch.

## 4. Existing forecast diagnostics and offline replay

Branch: `codex/forecast-diagnostics`
Commit: `feat: add offline forecast and rain-event diagnostics`
Prerequisite: batch 2.

Include:

- `src/telegram_code/forecast_mask.py` and only its extraction/import hunks in
  `src/telegram_code/methods.py`.
- `src/evaluation/`.
- `tests/test_notification_replay.py`.
- `scripts/run_heldout_test.py`, `diagnose_heldout_misses.py`,
  `replay_validation_notifications.py`, `summarize_notification_replay.py`,
  `audit_small_echoes.py`, and `check_radar_cleaning_solutions.py`.
- Source changes in `src/test_multimodal_convlstm_long.ipynb`, reviewed after
  removing outputs and execution counts in the isolated checkout.

Replace hard-coded personal Windows paths with repository-relative paths or CLI
arguments before staging. Separate genuine notebook code changes from thousands
of output-only diff lines. Do not publish diagnostic outputs as an independent
holdout result: the earlier held-out period has already informed development.

## 5. Arrival model framework

Branch: `codex/arrival-framework`
Commit: `feat: add causal local rain-arrival modelling framework`
Prerequisite: batch 2.

Include `src/arrival/` and framework tests:

- `test_arrival.py`, `test_arrival_architectures.py`,
  `test_arrival_hard_negatives.py`, `test_arrival_run_paths.py`,
  `test_arrival_sampling.py`.
- `scripts/smoke_arrival_pipeline.py`.

Keep model alternatives already implemented in this framework; committing code
does not authorize new architecture training. Validate the label boundaries,
causal inputs, model interfaces, sampling, calibration, and run recovery with
bounded tests. This does not switch the production bot to an arrival model.

## 6. Arrival experiment notebooks and replay tools

Branch: `codex/arrival-replay-tools`
Commits:

1. `feat: add reproducible arrival experiment notebooks`
2. `feat: compare calibrated arrival warnings at fixed operating targets`

Prerequisite: batch 5.

Include:

- `src/rain_arrival_model.ipynb`, `src/rain_arrival_experiments.ipynb`, and
  `scripts/build_arrival_notebooks.py`.
- `scripts/replay_arrival_notifications.py`, `compare_arrival_alert_burden.py`,
  `audit_arrival_calibration.py`, `experiment_arrival_projection.py`,
  `audit_arrival_warning_consistency.py`, `audit_arrival_negative_sampling.py`,
  `check_arrival_repeatability.py`, `diagnose_arrival_spatial_misses.py`,
  `diagnose_hard_negative_failure.py`, and `evaluate_arrival_recovery.py`.
- `tests/test_arrival_replay.py`.

Keep these imports together: the replay script imports the projection script,
which imports the calibration audit. Clean notebook outputs without executing
the notebooks. Reconcile generator/notebook defaults in the publication copies;
make expensive training and recalibration explicit opt-ins, while preserving
the user's active local notebook settings.

One-time notebook rewriting scripts (`add_*`, `fix_*`, `update_*`, `prepare_*`,
`expand_arrival_depth_sweep.py`, `harden_sweep_recovery.py`, and
`configure_deterministic_arrival_sweep.py`) need their own historical-tooling
commit if retained. Document their invocation order and mutation behavior, or
retain them locally as superseded helpers. Do not quietly include them as
supported entry points. In particular, some overwrite notebooks at import time.

## 7. Locked forward evaluation and its decision record

Branch: `codex/locked-forward-evaluation`
Commit: `feat: add locked forward comparison of frozen arrival finalists`
Prerequisite: batch 6.

Include:

- `scripts/locked_forward_evaluation.py`
- `tests/test_locked_forward_evaluation.py`
- A curated copy of the completed report, aggregate results, daily/run CSVs,
  verification record, and frozen protocol under a versioned documentation path
  such as `docs/evaluations/2026-09-15-forward/`.

Copy the original protocol byte-for-byte with its checksum; do not edit or
recreate the completed lock. Keep the original report and run directory intact.
If documentation links need adjusting, distinguish the publication copy from
the original. Exclude checkpoint copies, input caches, prediction arrays,
personal data, and temporary cache-warming helpers from the code PR.

Document how to obtain required model/calibration assets and old-validation
inputs. The evaluator currently depends on ignored local files and refuses to
overwrite completed results; a clean clone needs explicit reproduction setup.
Any later portability changes must be identified as later revisions, not the
exact hashed evaluator that produced this run.

The decision record must retain: keep v5, no threshold retuning, 1,406 eligible
anchors, 86 continuity runs, prior exposure of August 26–September 10, and only
46 eligible anchors in the September 11–14 breakout. Eight evaluator/replay
tests passed; daily/run/event totals reconciled after the completed run.

## 8. Separate full-archive v5 training workflow

Branch: `codex/v5-full-archive-workflow`
Commit: `feat: add explicit resumable full-archive v5 training workflow`
Prerequisite: batch 5.

Include `scripts/build_final_v5_full_data_notebook.py` and
`src/rain_arrival_v5_full_data.ipynb` together. This new notebook is present in
the working tree and must not be folded into the completed evaluation PR.

Review its code, resume guarantees, cutoff, and execution defaults separately.
Make clear that full-archive fitting creates new weights: old calibration and
the locked comparison do not validate that new checkpoint. Do not execute it
as part of push preparation, and do not promote its artifacts automatically.

## Files needing disposition before “everything” is published

- `AGENT_HANDOFF.md` and `STEP4_MODEL_IMPROVEMENTS.md`: review as historical
  notes, correct stale status statements in publication copies, and include in
  a separate documentation commit if useful. They are not current run commands.
- Already tracked `models/model_best_long.pkl` and
  `models/normalization_stats.json` are modified. Keep them out of code batches.
  Verify checkpoint, normalization, decoder and metadata together before any
  dedicated model-artifact release. Do not silently discard or untrack them.
- All new model/calibration artifacts and generated reports remain local by
  default, apart from the explicitly curated evaluation documentation above.
  If distributing the evaluated finalists, package their exact frozen weights,
  calibrators, normalization, thresholds and hashes as a separate asset bundle;
  copying only a checkpoint is insufficient.
- This plan can accompany the final documentation commit or remain local.

## Final checks for each push

1. Fetch and inspect the intended base and remote; reconcile concurrent edits.
2. Stage an explicit manifest; inspect `git diff --cached --stat`, the full
   staged diff, and `git diff --cached --check`.
3. Verify no credentials, local databases, training output, model binaries,
   personal paths, or unintended notebook results are staged.
4. Run the relevant tests from that branch, without local-only source files
   masking missing imports. Run the combined regression suite after integration.
5. Confirm the commit scope, write a concise behavior-and-validation PR
   description, and push only that branch.
6. Compare the remaining working-tree inventory with this plan so every file is
   either assigned, deliberately retained locally, or explicitly released.
