# Step 4: local rain-arrival prediction and movement/growth experiments

Status: planned, not implemented. Keep current training, checkpoints and notification behavior unchanged until an experiment is explicitly started.

## Added objective: local rain arrival within 60 minutes

Develop a separate location-centered model that predicts a distribution over twelve five-minute rain-arrival windows plus **no rain within 60 minutes**. This is a new output task, not a direct replacement of the current radar-frame model. Preserve the original +5/+10 experiments below as a separate workstream; their pixel metrics and losses do not transfer directly to arrival classification.

The complete supplied proposal is included later in this file. The implementation clarifications below take precedence where its illustrative details conflict with the existing pipeline or evaluation safeguards.

### Initial experiment configuration

- Six five-minute history frames, target-centered 64 × 64 crops, seven channels.
- Approximately 16–32 sampled locations per anchor, resampled during training.
- Currently dry anchors only; first qualifying future observation defines the arrival bin.
- Initial arrival criterion: at least five of the 25 center-neighborhood pixels exceed 0.01. Treat this as a candidate label definition, not proven noise rejection.
- Start with 13-class cross-entropy and a stacked-frame CNN baseline, then CNN + ConvGRU. Compare ConvLSTM subsequently.
- Keep the hazard/time-to-event formulation, history-length and environmental-channel ablations as staged experiments.
- No training or bot integration is authorized merely by adding this plan.

### Implementation clarifications and safeguards

1. **Channel order:** the existing pipeline is `[radar, temperature, humidity, wind_u, wind_v, station_mask, distance]`. The supplied proposal lists humidity before temperature. Preserve the existing order or explicitly version/reorder it; never swap channels silently.
2. **Temporal coverage:** T=6 requires history from t−25 through t and future observations through t+60: **18 frames spanning 85 minutes**, not a 70-minute sequence. Missing future observations are not “no rain.” Initially exclude incomplete windows; a later hazard model may explicitly handle right censoring.
3. **Partitioning:** assign entire days/events to partitions before creating anchors or locations. Require each complete history-plus-future interval to remain inside its partition. For T=12 comparisons, account for the full t−55 to t+60 interval and use the same eligible anchors across history ablations.
4. **Availability:** input weather must have been available by anchor time. Do not reuse future-CSV fallback logic from the operational loader when constructing a leakage-free arrival dataset. Future radar is for labels only.
5. **Balanced training is not calibrated deployment probability:** use the proposed four equally sampled arrival groups for training only. Preserve natural frequencies in validation/test. Fit calibration on representative validation data, accounting for sampling changes, before presenting percentages to users. Keep held-out evaluation locations fixed for reproducibility.
6. **Label scope:** a five-pixel coverage requirement may still admit clustered noise. Compare candidate label definitions on training/validation, document their spatial meaning, and preserve original-radar diagnostics. Already-raining locations need a separate state/clearing path.
7. **Coordinates and jitter:** omit explicit latitude/longitude initially, but do not assume this eliminates location information—station masks and distance fields can still encode geography. If the target is jittered, recenter the crop and regenerate its label at the new target. Define which pixel is the target in an even-sized 64 × 64 crop and how boundary crops are handled.
8. **Context size:** evaluate crop coverage for 30–60-minute arrivals; rain can enter from outside a local crop. Test crop size or additional coarse context if that limits skill. Keep input geometry/version metadata explicit.
9. **Baselines:** for a currently dry target, pure persistence predicts no arrival. Include it as a floor but use event metrics and the motion baseline so class imbalance does not make it appear sufficient. Treat any ConvGRU compute/accuracy advantage as a hypothesis to measure.
10. **Probability outputs:** a softmax already gives a distribution. A hazard model is an alternative to test, not an assumed improvement. For hazards h_k, arrival probability is `h_k × product(1−h_j, j<k)` and no-arrival probability is `product(1−h_j, all j)`. Verify probabilities sum to one and cumulative 15/30/60-minute probabilities are monotonic.
11. **Arrival error:** observations identify a five-minute interval, not an exact onset instant. Predefine bin-midpoint or endpoint scoring. Report missed arrivals and false alarms alongside time MAE; do not silently drop predictions of no rain from the error report.
12. **Operational integration:** an arrival-only model does not produce forecast radar images or rain-ending predictions. Keep those existing capabilities separate until an explicit integration design is selected. Evaluate probability calibration and repeated-alert behavior before rollout.

### Arrival-model experiment order

1. Implement and test the location-centered label generator and complete-window manifests.
2. Measure natural label frequencies, missing-window rates and label sensitivity.
3. Add the balanced training sampler and fixed representative validation locations.
4. Evaluate persistence and motion/advection baselines.
5. Train the stacked-frame CNN classifier.
6. Train CNN + ConvGRU under a comparable optimization budget.
7. Compare T=3, T=6 and T=9, then T=12 if justified.
8. Compare radar only, radar + wind, physical weather channels, and all seven channels.
9. Compare ConvGRU with ConvLSTM.
10. Compare hazard output with the categorical baseline and calibrate the selected model.
11. Freeze settings and evaluate a new untouched weather period.
12. Design and test Telegram integration, preserving radar images and clearing behavior as separate requirements.

Report macro F1, per-class precision/recall, confusion matrices, time error with miss rates, and calibrated rain-within-15/30/60-minute probabilities. Break down results by arrival horizon and independent weather events. Accuracy alone is insufficient.

## Existing radar-frame workstream

Improve forecasts where rain moves into, or develops at, previously dry pixels. The existing held-out diagnosis found much stronger recall at already-wet pixels than at previously-dry pixels. Small-echo noise is a separate uncertainty; do not optimize exclusively for those echoes.

## Prerequisites

- Choose and record one decoder/preprocessing version for each comparison. Do not feed exact-source values into a legacy-trained checkpoint.
- Save the chronological frame manifests, normalization statistics, seeds and model configuration.
- Fit normalization on training data only. Select thresholds and model settings on validation only.
- Use the continuous notification replay and region-size diagnostics alongside full-radar scores.
- Reserve a new untouched later period for final evaluation: the previous held-out set has already informed these development ideas.

## Experiment 1: denser training windows

Compare the current non-overlapping windows with overlapping windows advanced by one five-minute frame. Keep three input frames and both +5/+10 targets available for every experiment.

Split raw chronological frames before constructing windows. No source frame may appear in more than one partition, and windows must not cross missing-frame gaps. Keep validation examples identical between candidates.

Keep the architecture, loss and optimizer fixed. Compare under the same optimizer-update budget, recording elapsed time; epoch counts are not comparable when window counts change. Retain dry, clearing and developing-rain examples rather than sampling only wet scenes.

Decision: retain denser windows only if validation shows useful improvement in movement/growth or notification outcomes without an unacceptable false-alarm increase.

## Experiment 2: motion-extrapolation baseline

Estimate movement from the existing input radar frames and extrapolate to +5 and +10. Compare with simple persistence and the ConvLSTM on identical validation timestamps and preprocessing.

Use this to distinguish movement errors from growth/decay and noise problems. Do not assume motion extrapolation can predict newly developing rain. Handle image boundaries explicitly and report performance near boundaries separately.

Decision: consider motion features or a motion-aware model only if the baseline demonstrates a useful advantage on the relevant validation cases.

## Experiment 3: auxiliary +10 objective

Compare:

| Run | Objective |
|---|---|
| A | +5 loss only; auxiliary weight 0 |
| B | +5 loss plus the current 0.25-weighted +10 rollout loss |

Use the same two-target sample manifest for both runs. Do not change the dataset to a one-target configuration for Run A, because that changes sample grouping and invalidates the comparison.

Keep initialization seed, architecture, optimizer and update budget matched. Record compute time as well as forecasting quality. Select checkpoints using the same validation criterion. Assess whether Run B improves +5, rather than treating usable +10 output as proof that it helps +5.

Decision: retain the auxiliary objective if its validation benefit justifies its extra computation. Keep operational +10 disabled until its notification behavior is acceptable.

## Evaluation for every experiment

- Full-radar precision, recall, CSI, F1 and rainy/dry intensity errors for both horizons.
- Recall at previously-wet and previously-dry target pixels.
- Region-size diagnostics, with 1–5-pixel echoes reported separately and original targets retained.
- Continuous location-level onset warnings, premature predicted clearing, false start forecasts and lead time.
- Results by day/event, not only pooled pixels; repeat promising comparisons across seeds before drawing a strong conclusion.

Set acceptable notification tradeoffs on validation before selecting a winner. Do not tune settings to the previous test results.

## Follow-on work

Consider a controlled rain-head loss experiment only after these comparisons establish the remaining failure mode. Change one factor at a time; do not blindly activate the unused positive-class weight of 20.

Store experimental checkpoints, normalization files and reports separately from production artifacts. Freeze the winning checkpoint, preprocessing and calibration together before evaluating the new untouched period.

## Existing references

- `reports/source_check/implementation.md`: implemented decoder and reporting changes.
- `reports/notification_replay/report.md`: continuous validation replay results and limitations.
- `reports/heldout_misses/improvement_plan.md`: missed-rain evidence.
- `src/test_multimodal_convlstm_long.ipynb`: current training/evaluation notebook.

---

# Supplied arrival-time proposal

The following is preserved as supplied. Apply the integration clarifications above during implementation.

## Objective

Build a model that takes recent radar + environmental data around a user's location and predicts **when rain is expected to begin at that location within the next 60 minutes**.

The model should output a probability distribution over 5-minute arrival windows rather than generate an entire future radar frame.

---

## 1. Define the Prediction Task

Use a target-centered local crop.

### Input

For each target location:

* Radar + environmental channels
* Local crop around the target, initially `64 × 64`
* Target location always placed at the center
* Start with 6 timesteps at 5-minute intervals

So:

```text
Input shape:
[B, T=6, C=7, H=64, W=64]

History:
t-25
t-20
t-15
t-10
t-5
t
```

Initial channels:

```text
0 radar
1 humidity
2 temperature
3 wind_u
4 wind_v
5 station mask
6 distance to nearest station
```

Do not initially include latitude/longitude. This prevents the model from simply learning location-specific rain patterns.

### Output

Predict rain arrival during:

```text
0–5 min
5–10 min
10–15 min
...
55–60 min
No rain within 60 min
```

This gives 13 possible outcomes.

---

## 2. Define "Rain Arrival"

Do not use one individual radar pixel to determine whether rain has arrived.

For each target location, examine a small neighborhood around the center, for example:

```text
5 × 5 pixels
```

Define rain arrival using both:

* radar intensity threshold
* minimum spatial coverage

For example:

```python
rain_pixels = local_patch > rain_threshold

is_raining = rain_pixels.mean() >= coverage_threshold
```

Initial values to test:

```text
rain_threshold = 0.01
coverage_threshold = 0.20
```

This should reduce false labels caused by isolated radar noise or single-pixel echoes.

---

## 3. Generate Labels

For every valid anchor time `t`, require future observations:

```text
t+5
t+10
...
t+60
```

For each sampled location:

1. Confirm it is not currently raining.
2. Check future frames sequentially.
3. Record the first timestep where rain satisfies the arrival criterion.
4. If rain never arrives, label it as `"no rain within 60 min"`.

Example:

```text
t      dry
t+5    dry
t+10   dry
t+15   rain

Label = 10–15 min
```

---

## 4. Build the Dataset Around Anchor Times

Do not treat every location as requiring its own independent 70-minute sequence.

Instead:

```text
one valid temporal window
        ↓
one anchor time
        ↓
many target locations
```

For each anchor time, sample perhaps:

```text
16–32 locations
```

This lets one temporal sequence produce many training examples.

Resample locations every epoch rather than permanently storing every crop.

---

## 5. Balance the Sampling

Uniform location sampling will produce too many `"no rain"` examples.

Create candidate pools based on future arrival time:

```text
Group A: rain in 0–15 min
Group B: rain in 15–30 min
Group C: rain in 30–60 min
Group D: no rain within 60 min
```

Initially sample roughly equally:

```text
25% A
25% B
25% C
25% D
```

You can later adjust this distribution to better match real-world frequency.

---

## 6. Prevent Data Leakage

Split the dataset **before generating location samples**.

Do not randomly split crops.

Use entire days or preferably entire continuous weather/storm periods:

```text
Training events
Validation events
Test events
```

No overlapping radar sequence should appear across those sets.

This is critical because adjacent target locations and timestamps are highly correlated.

---

## 7. Establish Baselines First

Before building a large neural model, establish several baselines.

### Baseline 1 — Persistence

Predict that rain conditions remain roughly unchanged.

This provides a minimum performance floor.

### Baseline 2 — Motion / Optical Flow

Estimate radar-cell movement from recent frames and extrapolate toward the target.

Conceptually:

```text
rain motion
+
distance to target
        ↓
estimated arrival time
```

This is particularly important because short-term rain arrival may be dominated by advection.

### Baseline 3 — CNN Without Recurrence

Stack the six radar/environment frames together:

```text
[6 × 7 = 42 channels]
        ↓
CNN
        ↓
arrival-time output
```

This tests whether recurrent processing actually provides meaningful benefit.

---

## 8. Main Model

Start with:

```text
CNN encoder
    ↓
ConvGRU
    ↓
Global Average Pool
    ↓
MLP
    ↓
arrival-time output
```

Example structure:

```text
Input
6 × 7 × 64 × 64

↓ CNN

16 channels
32 × 32

↓ CNN

32 channels
16 × 16

↓ ConvGRU

64 hidden channels
16 × 16

↓ Global Average Pool

64 features

↓ MLP

64 → 32 → output
```

Use ConvGRU first because it has fewer parameters than ConvLSTM and your temporal sequence is relatively short.

Then compare against ConvLSTM.

---

## 9. Start With Simple Classification

For the first working version, use:

```text
13-way classification
```

with cross-entropy loss.

This gives you a straightforward baseline.

Once the complete pipeline works, replace this with a more appropriate probabilistic formulation.

---

## 10. Upgrade to a Hazard / Time-to-Event Model

The final model should ideally treat rain arrival as a time-to-event problem.

Predict:

```text
P(rain begins in 0–5 | not already raining)
P(rain begins in 5–10 | not arrived before)
P(rain begins in 10–15 | not arrived before)
...
```

From those probabilities, derive:

```text
Most likely arrival window
Probability rain arrives within 15 min
Probability rain arrives within 30 min
Probability rain arrives within 60 min
```

Example bot output:

```text
Rain likely in approximately 20–25 minutes.

Chance of rain:
within 15 min: 18%
within 30 min: 72%
within 60 min: 88%
```

This should ultimately be preferable to a single hard classification.

---

## 11. Test How Much History Is Needed

Run an ablation experiment using the same model and dataset:

```text
T=3  → 10 min history
T=6  → 25 min history
T=9  → 40 min history
T=12 → 55 min history
```

Measure performance separately by forecast range:

```text
0–15 min
15–30 min
30–60 min
```

Do not assume longer input automatically improves performance.

The main hypothesis is:

```text
T=3  may be sufficient for short-term arrival
T=6  may give the best accuracy/compute compromise
T=9+ may help 30–60 minute prediction but with diminishing returns
```

---

## 12. Test Whether Environmental Data Actually Helps

Run:

```text
Experiment A:
radar only

Experiment B:
radar + wind

Experiment C:
radar + wind + humidity + temperature

Experiment D:
all 7 channels
```

This will tell you whether the station-derived environmental channels are contributing information or just increasing model complexity.

---

## 13. Evaluation Metrics

Do not evaluate only overall accuracy.

Measure:

### Classification performance

```text
Accuracy
Macro F1
Per-class precision/recall
Confusion matrix
```

### Arrival-time error

For cases where rain does occur:

```text
MAE in minutes
```

For example:

```text
Actual arrival: +20 min
Predicted: +30 min

Error = 10 min
```

### Operational metrics

Evaluate:

```text
Rain within 15 min
Rain within 30 min
Rain within 60 min
```

using:

```text
Precision
Recall
F1
Brier score / calibration
```

Calibration is especially important if the Telegram bot will display probabilities.

---

## 14. Examine Performance by Forecast Horizon

Report performance separately for:

```text
0–15 minutes
15–30 minutes
30–60 minutes
```

Expect accuracy to degrade with lead time.

This is useful information rather than necessarily a model failure.

---

## 15. Overfitting Controls

Use:

```text
small model initially
dropout
weight decay
early stopping
storm/day-based validation split
dynamic location sampling
random spatial jitter
```

Target-center jitter can be approximately:

```text
±1–3 pixels
```

where appropriate.

Most importantly, judge overfitting based on completely unseen weather events, not randomly held-out crops.

---

## 16. Recommended Experiment Order

Run the project in this order:

```text
1. Build clean arrival-time label generator

2. Analyze label distribution

3. Implement balanced target-location sampler

4. Create persistence baseline

5. Create motion/advection baseline

6. Train simple CNN classifier

7. Train CNN + ConvGRU

8. Compare T=3 vs T=6 vs T=9

9. Compare radar-only vs multimodal

10. Compare ConvGRU vs ConvLSTM

11. Replace softmax output with hazard model

12. Calibrate probabilities

13. Test on completely unseen storm days

14. Integrate best model into Telegram bot
```

---

## Initial Configuration

The first serious model I would train is:

```text
History:             6 frames / 25 minutes
Forecast horizon:    60 minutes
Temporal resolution: 5 minutes

Input:
[B, 6, 7, 64, 64]

Model:
small CNN
→ ConvGRU
→ global pooling
→ MLP

Output:
13 arrival classes

Locations/anchor:
~32

Sampling:
balanced across
0–15 / 15–30 / 30–60 / no-rain

Validation:
completely separate days/storm events
```

Once that baseline works, the most valuable next upgrades are likely to be the **hazard output**, the **optical-flow baseline comparison**, and the **input-history ablation**.
