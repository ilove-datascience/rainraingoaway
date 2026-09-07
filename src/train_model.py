import os
import json
import copy
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.multimodal_radar_dataset import radar_dataset_multimodal
from models.multi_modal_convlstm import ConvLSTM_MM


SEED = 67
SAMPLE_SIZE = 3
BATCH_SIZE = 16
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 50


def radar_intensity_loss(
	pred,
	target,
	rain_threshold=0.01,
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


def dual_head_loss(
	pred,
	target,
	rain_logits,
	rain_threshold=0.01,
	classification_weight=1.0,
	rain_intensity_weight=1.0,
	dry_weight=0.25,
	beta=0.05,
	pos_weight=None,
	return_components=False,
):
	rain_target = (target > rain_threshold).float()
	rain_mask = rain_target.bool()
	dry_mask = ~rain_mask

	rain_loss = F.binary_cross_entropy_with_logits(
		rain_logits,
		rain_target,
		pos_weight=pos_weight,
	)

	if rain_mask.any():
		intensity_loss = F.smooth_l1_loss(
			pred[rain_mask],
			target[rain_mask],
			beta=beta,
		)
	else:
		intensity_loss = pred.new_tensor(0.0)

	if dry_mask.any():
		dry_loss = F.smooth_l1_loss(
			pred[dry_mask],
			torch.zeros_like(pred[dry_mask]),
			beta=beta,
		)
	else:
		dry_loss = pred.new_tensor(0.0)

	total_loss = (
		classification_weight * rain_loss
		+ rain_intensity_weight * intensity_loss
		+ dry_weight * dry_loss
	)
	if return_components:
		return total_loss, rain_loss, intensity_loss, dry_loss
	return total_loss


def train_one_epoch(epoch_index, optimizer, model, loss_fn, train_loader, device):
	running_loss = 0.0
	last_loss = 0.0

	for i, data in enumerate(train_loader):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs, _, _, rain_logits = model(inputs, return_logits=True)
		outputs = outputs.squeeze(1)
		rain_logits = rain_logits.squeeze(1)

		loss, rain_loss, intensity_loss, dry_loss = loss_fn(
			outputs,
			labels,
			rain_logits,
			return_components=True,
		)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 10 == 9:
			last_loss = running_loss / 10
			print(f"  batch {i + 1} loss: {last_loss}")
			print(
				f"  rain={rain_loss.item():.4f} "
				f"intensity={intensity_loss.item():.4f} "
				f"dry={dry_loss.item():.4f} "
				f"weighted_dry={0.25 * dry_loss.item():.4f}"
			)
			running_loss = 0.0

	if len(train_loader) > 0 and last_loss == 0.0:
		last_loss = running_loss / len(train_loader)

	return last_loss


def main():
	torch.manual_seed(SEED)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	project_root = Path(__file__).resolve().parents[1]
	radar_png_dir = project_root / "data" / "70km" / "png"
	env_dir = project_root / "data" / "environment"

	dataset = radar_dataset_multimodal(
		str(radar_png_dir),
		str(env_dir),
		list_length=SAMPLE_SIZE,
		total=None,
		num_workers=8,
	)

	split_idx = int(0.8 * len(dataset))
	train_indices = list(range(0, split_idx))
	dataset.fit_normalization(train_indices)
	normalization_stats = dataset.get_normalization_stats()
	print("Applied train-split normalization stats for env channels and distance")
	train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
	test_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	print(f"Train samples: {len(train_dataset)}")
	print(f"Test samples: {len(test_dataset)}")
	print(f"Train batches: {len(train_loader)}")
	print(f"Test batches: {len(test_loader)}")

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

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
	loss_fn = radar_intensity_loss
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = project_root / "models" / timestamp
	run_dir.mkdir(parents=True, exist_ok=True)
	model_path = project_root / "models" / "model_best_latest.pkl"
	losses_path = run_dir / f"multimodal_convlstm_losses_{timestamp}.csv"
	norm_stats_path = project_root / "models" / "normalization_stats.json"
	norm_stats_path.write_text(json.dumps(normalization_stats, indent=2), encoding="utf-8")
	print(f"Saved production normalization stats: {norm_stats_path}")

	epoch_number = 0
	train_losses = []
	val_losses = []
	best_vloss = float("inf")
	best_state_dict = None
	epochs_without_improvement = 0

	for _ in range(EPOCHS):
		print(f"EPOCH {epoch_number + 1}:")

		model.train(True)
		avg_train_loss = train_one_epoch(
			epoch_number,
			optimizer,
			model,
			loss_fn,
			train_loader,
			device,
		)

		running_vloss = 0.0
		model.eval()

		with torch.no_grad():
			for i, vdata in enumerate(test_loader):
				vinputs, vlabels = vdata
				vinputs = vinputs.to(device)
				vlabels = vlabels.to(device)

				voutputs, _, _, v_rain_logits = model(vinputs, return_logits=True)
				voutputs = voutputs.squeeze(1)
				v_rain_logits = v_rain_logits.squeeze(1)
				vloss = dual_head_loss(voutputs, vlabels.float(), v_rain_logits)
				running_vloss += vloss.item()

		avg_val_loss = running_vloss / max(len(test_loader), 1)
		print(f"LOSS train {avg_train_loss} valid {avg_val_loss}")

		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)

		if avg_val_loss < best_vloss:
			best_vloss = avg_val_loss
			best_state_dict = copy.deepcopy(model.state_dict())
			epochs_without_improvement = 0
			torch.save(model.state_dict(), model_path)
			print(f"Saved best model to production checkpoint: {model_path}")
		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
			print(
				f"Early stopping triggered after {epoch_number + 1} epochs "
				f"with no validation improvement for {EARLY_STOPPING_PATIENCE} epochs."
			)
			break

		epoch_number += 1

	if best_state_dict is not None:
		model.load_state_dict(best_state_dict)
		print("Restored best model weights in memory.")

	losses_df = pd.DataFrame(
		{
			"epoch": list(range(1, len(train_losses) + 1)),
			"train_loss": train_losses,
			"val_loss": val_losses,
		}
	)
	losses_df.to_csv(losses_path, index=False)
	print(f"Saved best model: {model_path}")
	print(f"Saved losses CSV: {losses_path}")


if __name__ == "__main__":
	os.environ.pop("MPLBACKEND", None)
	main()
