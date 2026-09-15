# Frozen forward evaluation publication

The decision is **keep v5**. See [the report](report.md), [daily results](by_day.csv),
[continuous-run results](by_continuous_run.csv), and [verification](verification.json).
The original lock is published byte-for-byte with its SHA-256 checksum.

No models, calibrators, input data, prediction arrays, or user locations are
distributed here. The nine evaluated locations are synthetic radar pixels.

## Reproduction prerequisites

Restore the exact checkpoint and calibration files named in `lock.json`, the
`models/arrival/data_v1/prepared.json` and evaluation-location records, the old
validation replay bundles and matched-burden comparison, and the input radar and
weather files specified by the lock. Obtain these from the retained local
experiment archive; they are not downloadable from this repository. Verify their
hashes against the lock. The complete prior archive is required, not just model weights.

The evaluator imports the replay, projection, and calibration audit scripts.
Install the repository dependencies and use `scripts/locked_forward_evaluation.py`.
For a *new* run, `freeze` records a protocol before `evaluate` can score it. For
the historical run, restore its original lock and frozen artifact directory to
`reports/locked_forward_20260826_20260914/` and use `evaluate` only if completed
results are not already present. It deliberately refuses to overwrite results.

Exact historical source bytes also matter: the lock includes code hashes, and
Git line-ending conversion can change them. Restore matching source files from
the original archive if hashes differ; do not regenerate the old lock to bypass
verification. Later changes to code or dependencies are not the historical run.

This was not a pristine independent holdout: August 26–September 10 had already
informed earlier error analysis. September 11–14 had just 46 eligible timestamps.
No thresholds were adjusted after seeing forward results.
