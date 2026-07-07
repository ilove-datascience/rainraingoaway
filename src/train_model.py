import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.multimodal_radar_dataset import radar_dataset_multimodal
from models.convlstm import ConvLSTM


SEED = 67
SAMPLE_SIZE = 4
BATCH_SIZE = 16
EPOCHS = 200


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


def train_one_epoch(epoch_index, optimizer, model, loss_fn, train_loader, device):
	running_loss = 0.0
	last_loss = 0.0

	for i, data in enumerate(train_loader):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		outputs = outputs.squeeze(1)

		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 10 == 9:
			last_loss = running_loss / 10
			print(f"  batch {i + 1} loss: {last_loss}")
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
	)

	split_idx = int(0.8 * len(dataset))
	train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
	test_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	print(f"Train samples: {len(train_dataset)}")
	print(f"Test samples: {len(test_dataset)}")
	print(f"Train batches: {len(train_loader)}")
	print(f"Test batches: {len(test_loader)}")

	model = ConvLSTM(
		input_dim=5,
		hidden_dim=64,
		kernel_size=(3, 3),
		num_layers=2,
		batch_first=True,
		bias=True,
	).to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr=0.007)
	loss_fn = radar_intensity_loss
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = project_root / "models" / timestamp
	run_dir.mkdir(parents=True, exist_ok=True)
	model_path = run_dir / f"model_{timestamp}.pkl"
	losses_path = run_dir / f"multimodal_convlstm_losses_{timestamp}.csv"

	epoch_number = 0
	train_losses = []
	val_losses = []
	best_vloss = float("inf")

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

				voutputs = model(vinputs).squeeze(1)
				vloss = loss_fn(voutputs, vlabels.float())
				running_vloss += vloss.item()

		avg_val_loss = running_vloss / max(len(test_loader), 1)
		print(f"LOSS train {avg_train_loss} valid {avg_val_loss}")

		train_losses.append(avg_train_loss)
		val_losses.append(avg_val_loss)

		if avg_val_loss < best_vloss:
			best_vloss = avg_val_loss
			torch.save(model.state_dict(), model_path)

		epoch_number += 1

	losses_df = pd.DataFrame(
		{
			"epoch": list(range(1, EPOCHS + 1)),
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
