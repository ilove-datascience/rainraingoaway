import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.multimodal_radar_dataset import radar_dataset_multimodal
from models.multi_modal_convlstm import ConvLSTM_MM


SEED = 67
SAMPLE_SIZE = 4
BATCH_SIZE = 16
PREVIEW_COUNT = 3
RAIN_THRESHOLD = 0.01
SSIM_WINDOW_SIZE = 7
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2
EVAL_DATA_ROOT = Path("data") / "eval" / "20260709_20260710"


def radar_intensity_loss(
	pred,
	target,
	rain_threshold=RAIN_THRESHOLD,
	background_weight=0.2,
	rain_weight=2.5,
	intensity_weight=2.0,
	beta=0.05,
):
	loss = F.smooth_l1_loss(
		pred,
		target,
		reduction="none",
		beta=beta,
	)

	rain_mask = (target > rain_threshold).float()
	weights = background_weight + rain_mask * rain_weight + target * intensity_weight

	return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def structural_similarity(predictions, targets, data_range=1.0, window_size=SSIM_WINDOW_SIZE):
	predictions_4d = predictions.unsqueeze(1)
	targets_4d = targets.unsqueeze(1)

	padding = window_size // 2
	mu_pred = F.avg_pool2d(predictions_4d, kernel_size=window_size, stride=1, padding=padding)
	mu_target = F.avg_pool2d(targets_4d, kernel_size=window_size, stride=1, padding=padding)

	mu_pred_sq = mu_pred.pow(2)
	mu_target_sq = mu_target.pow(2)
	mu_pred_target = mu_pred * mu_target

	sigma_pred_sq = F.avg_pool2d(predictions_4d * predictions_4d, kernel_size=window_size, stride=1, padding=padding) - mu_pred_sq
	sigma_target_sq = F.avg_pool2d(targets_4d * targets_4d, kernel_size=window_size, stride=1, padding=padding) - mu_target_sq
	sigma_pred_target = F.avg_pool2d(predictions_4d * targets_4d, kernel_size=window_size, stride=1, padding=padding) - mu_pred_target

	c1 = (SSIM_C1 * data_range * data_range)
	c2 = (SSIM_C2 * data_range * data_range)

	numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
	denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

	ssim_map = numerator / denominator.clamp_min(1e-12)
	return ssim_map.mean().item()


def find_latest_checkpoint(project_root: Path) -> Path:
	checkpoint_candidates = list(project_root.glob("models/*/model_*.pkl"))
	checkpoint_candidates.extend(project_root.glob("models/model_*.pkl"))
	checkpoint_candidates.extend(project_root.glob("models/model_best_*.pkl"))

	if not checkpoint_candidates:
		raise FileNotFoundError(
			"No model checkpoint found. Pass --checkpoint or train the model first."
		)

	return max(checkpoint_candidates, key=lambda path: path.stat().st_mtime)


def load_dataset(project_root: Path, sample_size: int, radar_png_dir: Path | None = None, env_dir: Path | None = None):
	if radar_png_dir is None:
		radar_png_dir = project_root / EVAL_DATA_ROOT / "png"

	if env_dir is None:
		env_dir = project_root / EVAL_DATA_ROOT / "environment"

	dataset = radar_dataset_multimodal(
		str(radar_png_dir),
		str(env_dir),
		list_length=sample_size,
		total=None,
		num_workers=8,
	)

	return dataset


def find_normalization_stats_file(checkpoint_path: Path) -> Path | None:
	candidate = checkpoint_path.parent / "normalization_stats.json"
	if candidate.exists():
		return candidate
	return None


def find_latest_losses_csv(checkpoint_dir: Path) -> Path | None:
	losses_candidates = list(checkpoint_dir.glob("multimodal_convlstm_losses_*.csv"))
	if not losses_candidates:
		return None

	return max(losses_candidates, key=lambda path: path.stat().st_mtime)


def plot_loss_curve(losses_path: Path, output_path: Path):
	losses_df = pd.read_csv(losses_path)

	fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
	axis.plot(losses_df["epoch"], losses_df["train_loss"], label="Train loss", linewidth=2)
	axis.plot(losses_df["epoch"], losses_df["val_loss"], label="Validation loss", linewidth=2)
	axis.set_title("Training loss curve")
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Loss")
	axis.grid(True, alpha=0.3)
	axis.legend()

	fig.savefig(output_path, dpi=160)
	plt.close(fig)


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


def evaluate(model, loader, device):
	total_loss = 0.0
	total_mae = 0.0
	total_mse = 0.0
	total_pixels = 0
	total_accuracy = 0.0
	total_ssim = 0.0
	ssim_batches = 0
	rain_mae_sum = 0.0
	train_pixels = 0
	preview_candidates = []
	fallback_preview = None

	with torch.no_grad():
		for batch_index, (inputs, targets) in enumerate(loader):
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

			rain_mask = targets > RAIN_THRESHOLD
			if rain_mask.any():
				rain_mae_sum += diff.abs()[rain_mask].sum().item()
				train_pixels += rain_mask.sum().item()

			sample_rain_mask = rain_mask.view(rain_mask.shape[0], -1)
			sample_rain_pixels = sample_rain_mask.sum(dim=1)
			sample_rain_intensity = (targets * rain_mask.float()).view(targets.shape[0], -1).sum(dim=1)

			if fallback_preview is None:
				fallback_preview = {
					"inputs": inputs[0].detach().cpu(),
					"target": targets[0].detach().cpu(),
					"prediction": predictions[0].detach().cpu(),
				}

			for sample_index in range(targets.shape[0]):
				rain_pixels = int(sample_rain_pixels[sample_index].item())
				if rain_pixels == 0:
					continue

				rain_intensity = float(sample_rain_intensity[sample_index].item())
				preview_candidates.append(
					{
						"score": (rain_pixels, rain_intensity),
						"inputs": inputs[sample_index].detach().cpu(),
						"target": targets[sample_index].detach().cpu(),
						"prediction": predictions[sample_index].detach().cpu(),
					},
				)

	if preview_candidates:
		preview_candidates.sort(key=lambda item: item["score"], reverse=True)
		previews = preview_candidates[:PREVIEW_COUNT]
	else:
		previews = [fallback_preview] if fallback_preview is not None else []

	metrics = {
		"loss": total_loss / max(len(loader), 1),
		"mae": total_mae / max(total_pixels, 1),
		"rmse": math.sqrt(total_mse / max(total_pixels, 1)),
		"accuracy": total_accuracy / max(len(loader), 1),
		"ssim": total_ssim / max(ssim_batches, 1),
		"rain_mae": rain_mae_sum / max(train_pixels, 1),
	}

	return metrics, previews


def save_preview(preview, output_path: Path):
	inputs = preview["inputs"]
	target = preview["target"]
	prediction = preview["prediction"]
	latest_radar_frame = inputs[-1, 0].numpy()
	residual = prediction.numpy() - target.numpy()

	fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
	plots = [
		(latest_radar_frame, "Latest input radar"),
		(target.numpy(), "Target next radar"),
		(prediction.numpy(), "Predicted next radar"),
		(residual, "Prediction error"),
	]

	for axis, (image, title) in zip(axes, plots):
		image_plot = axis.imshow(image, aspect="auto")
		axis.set_title(title)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image_plot, ax=axis, fraction=0.046, pad=0.04)

	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def save_preview_set(previews, output_path: Path):
	if not previews:
		return []

	if len(previews) == 1:
		save_preview(previews[0], output_path)
		return [output_path]

	output_paths = []
	for index, preview in enumerate(previews, start=1):
		preview_path = output_path.with_name(f"{output_path.stem}_{index}{output_path.suffix}")
		save_preview(preview, preview_path)
		output_paths.append(preview_path)

	return output_paths


def main():
	parser = argparse.ArgumentParser(description="Evaluate a trained multimodal ConvLSTM model")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=None,
		help="Path to a saved model_*.pkl checkpoint. Defaults to the latest checkpoint in models/.",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=SAMPLE_SIZE,
		help="Number of past radar frames used per sample.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=BATCH_SIZE,
		help="Evaluation batch size.",
	)
	parser.add_argument(
		"--radar-dir",
		type=str,
		default=None,
		help="Optional path to the radar PNG folder for evaluation samples.",
	)
	parser.add_argument(
		"--env-dir",
		type=str,
		default=None,
		help="Optional path to the environment CSV folder for evaluation samples.",
	)
	parser.add_argument(
		"--preview-path",
		type=str,
		default=None,
		help="Optional path for a PNG preview of the first evaluation sample.",
	)
	args = parser.parse_args()

	torch.manual_seed(SEED)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	project_root = Path(__file__).resolve().parents[1]
	checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(project_root)
	print(f"Checkpoint: {checkpoint_path}")

	radar_dir = Path(args.radar_dir) if args.radar_dir else None
	env_dir = Path(args.env_dir) if args.env_dir else None
	test_dataset = load_dataset(project_root, args.sample_size, radar_dir, env_dir)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	print(f"Test samples: {len(test_dataset)}")
	print(f"Test batches: {len(test_loader)}")

	model = load_model(checkpoint_path, device)
	norm_stats_path = find_normalization_stats_file(checkpoint_path)
	if norm_stats_path is not None:
		norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
		test_dataset.set_normalization_stats(norm_stats)
		print(f"Loaded normalization stats: {norm_stats_path}")
	else:
		print("Warning: normalization_stats.json not found next to checkpoint; using raw env scales.")
	metrics, previews = evaluate(model, test_loader, device)

	print("Evaluation metrics:")
	for name, value in metrics.items():
		print(f"  {name}: {value}")

	checkpoint_dir = checkpoint_path.parent
	metrics_path = checkpoint_dir / f"eval_metrics_{checkpoint_path.stem}.json"
	with metrics_path.open("w", encoding="utf-8") as metrics_file:
		json.dump(metrics, metrics_file, indent=2)
	print(f"Saved metrics: {metrics_path}")

	losses_path = find_latest_losses_csv(checkpoint_dir)
	if losses_path is not None:
		loss_curve_path = checkpoint_dir / f"loss_curve_{checkpoint_path.stem}.png"
		plot_loss_curve(losses_path, loss_curve_path)
		print(f"Saved loss curve: {loss_curve_path}")

		losses_df = pd.read_csv(losses_path)
		last_row = losses_df.iloc[-1]
		print(
			"Last recorded training losses: "
			f"train={last_row['train_loss']} val={last_row['val_loss']}"
		)

	preview_path = Path(args.preview_path) if args.preview_path else checkpoint_dir / f"eval_preview_{checkpoint_path.stem}.png"
	preview_paths = save_preview_set(previews, preview_path)
	for path in preview_paths:
		print(f"Saved preview: {path}")


if __name__ == "__main__":
	main()