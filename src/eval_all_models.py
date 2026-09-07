"""Evaluate every saved model checkpoint against a single held-out eval set and rank them.

Metrics used and why
---------------------
Rain nowcasting is dominated by mostly-dry pixels, so plain pixel-wise MAE/RMSE mostly
measures how well a model predicts "no rain" and can look good even when it fails at the
one thing that matters: correctly forecasting where/how much it rains. We therefore score
models on a mix of classification-quality and intensity-quality metrics, all computed on
the *same* eval dataset for a fair comparison:

  - CSI (Critical Success Index): pred_rain & target_rain / pred_rain | target_rain.
    Standard meteorological verification metric; a single number that punishes both
    misses and false alarms, unlike precision or recall alone.
  - Recall (POD, probability of detection): fraction of actual rain pixels correctly
    flagged. Weighted heavily because earlier checkpoints collapsed to near-zero recall
    (predicting "dry" everywhere) while still posting a deceptively low overall MAE.
  - Precision: fraction of predicted-rain pixels that were actually rain. Without this,
    a model could inflate recall by flagging rain everywhere.
  - rain_mae: mean absolute error restricted to actually-rainy pixels. Captures intensity
    accuracy where it matters, ignoring the large dry background that dominates plain MAE.
  - SSIM: structural similarity between predicted and target fields. Flags models that
    minimize loss by outputting a blurry/mean field instead of a spatially coherent one.
  - dry_false_alarm_rate (dry_pred > 0.01): fraction of truly-dry pixels the model predicts
    as wet. Guards against a model that overreacts (raises recall by hallucinating rain).

Composite score = weighted, direction-corrected, min-max normalized combination of the
above (see WEIGHTS below). Plain MAE/RMSE/loss/accuracy are still reported for reference
but are not part of the ranking, since they're easily dominated by the dry background.
"""
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.multimodal_radar_dataset import radar_dataset_multimodal
from models.multi_modal_convlstm import ConvLSTM_MM
from eval_model import (
    EVAL_DATA_ROOT,
    RAIN_THRESHOLD,
    find_normalization_stats_file,
    radar_intensity_loss,
    structural_similarity,
)

SEED = 67
SAMPLE_SIZE = 4
BATCH_SIZE = 16
TOP_N = 5

# Composite score weights; each metric is min-max normalized across candidate models
# (direction-corrected so "higher normalized value = better") before weighting.
WEIGHTS = {
    "csi": 0.30,
    "recall": 0.20,
    "precision": 0.15,
    "rain_mae": 0.15,   # lower is better -> inverted before weighting
    "ssim": 0.10,
    "dry_false_alarm_rate": 0.10,  # lower is better -> inverted before weighting
}


def find_all_checkpoints(project_root: Path) -> list[Path]:
    models_dir = project_root / "models"
    checkpoints = list(models_dir.glob("model_best_*.pkl"))
    checkpoints.extend(models_dir.glob("*/model_*.pkl"))
    return sorted(set(checkpoints))


def load_model(checkpoint_path: Path, device: torch.device) -> ConvLSTM_MM:
    model = ConvLSTM_MM(
        input_dim=7,
        hidden_dim=[32, 64],
        kernel_size=[(3, 3), (3, 3)],
        num_layers=2,
        batch_first=True,
        bias=True,
        land_use_channels=33,
        land_use_feature_dim=8,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, loader, device):
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_pixels = 0
    total_accuracy = 0.0
    total_ssim = 0.0
    ssim_batches = 0

    rain_mae_sum = 0.0
    rain_pixel_count = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    dry_false_alarms = 0
    dry_pixel_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs).squeeze(1)
            batch_loss = radar_intensity_loss(predictions, targets)
            diff = predictions - targets

            batch_pixels = targets.numel()
            total_loss += batch_loss.item()
            total_mae += diff.abs().sum().item()
            total_mse += diff.pow(2).sum().item()
            total_pixels += batch_pixels

            predicted_rain = predictions > RAIN_THRESHOLD
            target_rain = targets > RAIN_THRESHOLD
            total_accuracy += (predicted_rain == target_rain).float().mean().item()
            total_ssim += structural_similarity(predictions, targets, data_range=1.0)
            ssim_batches += 1

            rain_mask = target_rain
            dry_mask = ~target_rain

            if rain_mask.any():
                rain_mae_sum += diff.abs()[rain_mask].sum().item()
                rain_pixel_count += rain_mask.sum().item()

            true_positive += (predicted_rain & target_rain).sum().item()
            false_positive += (predicted_rain & ~target_rain).sum().item()
            false_negative += (~predicted_rain & target_rain).sum().item()

            if dry_mask.any():
                dry_false_alarms += (predicted_rain & dry_mask).sum().item()
                dry_pixel_count += dry_mask.sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    csi = true_positive / max(true_positive + false_positive + false_negative, 1)

    return {
        "loss": total_loss / max(len(loader), 1),
        "mae": total_mae / max(total_pixels, 1),
        "rmse": math.sqrt(total_mse / max(total_pixels, 1)),
        "accuracy": total_accuracy / max(len(loader), 1),
        "ssim": total_ssim / max(ssim_batches, 1),
        "rain_mae": rain_mae_sum / max(rain_pixel_count, 1),
        "precision": precision,
        "recall": recall,
        "csi": csi,
        "dry_false_alarm_rate": dry_false_alarms / max(dry_pixel_count, 1),
    }


def compute_composite_scores(results: list[dict]) -> None:
    """Add a 'composite_score' key to each result dict, mutating in place."""
    lower_is_better = {"rain_mae", "dry_false_alarm_rate"}

    for metric in WEIGHTS:
        values = [r["metrics"][metric] for r in results]
        lo, hi = min(values), max(values)
        span = hi - lo

        for r in results:
            value = r["metrics"][metric]
            normalized = 0.5 if span == 0 else (value - lo) / span
            if metric in lower_is_better:
                normalized = 1.0 - normalized
            r.setdefault("normalized", {})[metric] = normalized

    for r in results:
        r["composite_score"] = sum(
            WEIGHTS[metric] * r["normalized"][metric] for metric in WEIGHTS
        )


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(__file__).resolve().parents[1]
    checkpoints = find_all_checkpoints(project_root)
    print(f"Found {len(checkpoints)} checkpoints to evaluate\n")

    dataset = radar_dataset_multimodal(
        str(project_root / EVAL_DATA_ROOT / "png"),
        str(project_root / EVAL_DATA_ROOT / "environment"),
        list_length=SAMPLE_SIZE,
        total=None,
        num_workers=8,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Eval samples: {len(dataset)} | batches: {len(loader)}\n")

    results = []
    for i, checkpoint_path in enumerate(checkpoints, start=1):
        rel_path = checkpoint_path.relative_to(project_root)
        print(f"[{i}/{len(checkpoints)}] {rel_path}")
        try:
            model = load_model(checkpoint_path, device)
        except Exception as exc:
            print(f"  skipped (failed to load, likely incompatible architecture): {exc}")
            continue

        norm_stats_path = find_normalization_stats_file(checkpoint_path)
        if norm_stats_path is not None:
            norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
            dataset.set_normalization_stats(norm_stats)
        else:
            dataset.set_normalization_stats(None)

        try:
            metrics = evaluate_model(model, loader, device)
        except Exception as exc:
            print(f"  skipped (evaluation failed): {exc}")
            continue

        print(
            f"  csi={metrics['csi']:.3f} recall={metrics['recall']:.3f} "
            f"precision={metrics['precision']:.3f} rain_mae={metrics['rain_mae']:.4f} "
            f"ssim={metrics['ssim']:.3f} dry_fp={metrics['dry_false_alarm_rate']:.3f}"
        )
        results.append({"checkpoint": str(rel_path), "metrics": metrics})

    if not results:
        print("\nNo checkpoints could be evaluated.")
        return

    compute_composite_scores(results)
    results.sort(key=lambda r: r["composite_score"], reverse=True)

    print("\n" + "=" * 100)
    print(f"Top {min(TOP_N, len(results))} models by composite score")
    print("=" * 100)

    top_results = results[:TOP_N]
    rows = []
    for rank, r in enumerate(top_results, start=1):
        m = r["metrics"]
        print(
            f"\n#{rank} {r['checkpoint']}  (composite score: {r['composite_score']:.4f})\n"
            f"    csi={m['csi']:.4f}  recall={m['recall']:.4f}  precision={m['precision']:.4f}\n"
            f"    rain_mae={m['rain_mae']:.4f}  ssim={m['ssim']:.4f}  dry_false_alarm_rate={m['dry_false_alarm_rate']:.4f}\n"
            f"    (reference only) loss={m['loss']:.4f}  mae={m['mae']:.4f}  rmse={m['rmse']:.4f}  accuracy={m['accuracy']:.4f}"
        )
        rows.append({"rank": rank, "checkpoint": r["checkpoint"], "composite_score": r["composite_score"], **m})

    output_path = project_root / "model_leaderboard.csv"
    all_rows = []
    for rank, r in enumerate(results, start=1):
        all_rows.append({"rank": rank, "checkpoint": r["checkpoint"], "composite_score": r["composite_score"], **r["metrics"]})
    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    print(f"\nSaved full leaderboard ({len(results)} models) to {output_path}")


if __name__ == "__main__":
    main()
