# Locked forward comparison: 26 August–14 September 2026

Keep v5. Spatial does not clearly beat v5 under the frozen rule. Stop architecture tuning and collect more months with a truly reserved prospective test.

Forward relative to both finalists training/validation/calibration dates, but not a pristine independent holdout.
reports/heldout_misses/improvement_plan.md documents inspected 26 August-10 September targets and resulting development ideas. September 11-14 is reported separately, without claiming verified never-inspected status.

Raw coverage: 202608260000 to 202609141025 SGT; 1406 eligible anchors, 86 continuity runs, nine fixed locations.
Checkpoint and calibrator copies, old validation thresholds, input files, scoring code and decision rule were hashed before inference. All hashes were rechecked after scoring. No model fitting, calibration fitting or threshold tuning occurred.

## Full requested period

| Model | Minutes | Threshold | AP | Warned/onsets | False/location-day | Median lead (min) |
|---|---:|---:|---:|---:|---:|---:|
| v5 | 15 | 0.315 | 0.338 | 13/59 | 0.218 | 10.000 |
| v5 | 30 | 0.361 | 0.337 | 12/51 | 0.159 | 10.000 |
| v5 | 60 | 0.301 | 0.309 | 15/43 | 0.129 | 15.000 |
| depth2_spatial | 15 | 0.453 | 0.321 | 19/59 | 0.245 | 5.000 |
| depth2_spatial | 30 | 0.499 | 0.303 | 16/51 | 0.190 | 10.000 |
| depth2_spatial | 60 | 0.563 | 0.267 | 11/43 | 0.215 | 5.000 |

## September 11–14 breakout

Shorter, less-exposed date range; never-inspected status is not established. Same frozen thresholds.

| Model | Minutes | Threshold | AP | Warned/onsets | False/location-day | Median lead (min) |
|---|---:|---:|---:|---:|---:|---:|
| v5 | 15 | 0.315 | 0.428 | 1/3 | 3.556 | 5.000 |
| v5 | 30 | 0.361 | 0.424 | 0/0 | 0.000 | n/a |
| v5 | 60 | 0.301 | 0.486 | 0/0 | n/a | n/a |
| depth2_spatial | 15 | 0.453 | 0.334 | 1/3 | 1.778 | 5.000 |
| depth2_spatial | 30 | 0.499 | 0.351 | 0/0 | 0.000 | n/a |
| depth2_spatial | 60 | 0.563 | 0.335 | 0/0 | n/a | n/a |

Spatial minus v5 macro onset-recall difference: 0.029; paired continuity-block bootstrap 95% interval: [-0.0316820112242342, 0.12848525983820086].
The bootstrap is diagnostic: continuity blocks can belong to the same weather system, and three horizons are correlated.

## Definitions and detailed results

- **location**: Same nine fixed (y,x) radar pixels as old replay; centered 64x64 crops. No new location selection.
- **wet**: At least 5 of 25 pixels in the centered 5x5 patch have cleaned SOURCE radar intensity > 0.01.
- **onset**: A dry-to-wet transition at a fixed location within an uninterrupted eligible 5-minute run.
- **eligibility**: All 12 historical radar/environment frames and 12 future radar frames exist within the requested period; each historical environment frame passes the existing causal validity check. Same max_history=12 common eligibility as development.
- **ap**: Tie-aware average precision at currently dry eligible location/timestamps; positive if rain first occurs within 15/30/60 minutes. All eligible anchors, no sampling. Calibrated projected probabilities.
- **warning**: Existing score_run policy: >= frozen threshold, one confirmation, suppress until onset or horizon expiry; no clear notifications. Full forward horizon within eligible run required to issue warnings; onsets require full prior horizon within run.
- **exposure**: Same old-replay denominator: max(run_length - horizon_steps, 0) * 5 minutes per location, including wet times.
- **day_attribution**: Do not reset at midnight. Warnings/false warnings/lead time belong to issue day; arrivals warned/onsets belong to onset day; exposure and AP belong to anchor day.
- **continuous_weather_run**: Maximal uninterrupted 5-minute sequence of eligible input anchors. This is a data-continuity block, not a meteorologically independent storm.
- **boundaries**: Missing inputs split runs; no filling missing radar as dry. AP uses complete raw future radar even at eligible-run edges; warning/onset censoring follows old replay.

- [By day](by_day.csv): includes zero-coverage dates; no midnight reset.
- [By continuous weather run](by_continuous_run.csv).
- [Warning events](warning_events.csv).
- [Frozen protocol](lock.json).
- [Machine-readable results](results.json).

Archived observation timestamps only; historical publication/ingestion timing and Telegram delivery latency unavailable.
The 0.25 rate is the old-validation selection target, not a guaranteed forward rate. Rates above it are reported without retuning. These are radar-pixel arrival warnings, not observed user experiences or delivered Telegram notifications.
