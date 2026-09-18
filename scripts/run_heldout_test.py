from pathlib import Path
_repo_root = Path(__file__).resolve().parents[1]

import os, sys, json
from pathlib import Path
import torch, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models.multi_modal_convlstm import ConvLSTM_MM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device,flush=True)
import importlib
import json
from pathlib import Path
import data_processing.multimodal_radar_dataset as multimodal_dataset_module

importlib.reload(multimodal_dataset_module)
radar_dataset_multimodal = multimodal_dataset_module.radar_dataset_multimodal

# T=3: three input radar/environment frames predict t+5, then t+5 is rolled back into
# the same model (with persisted t environment) to predict t+10.
sequence_length = 3
num_target_steps = 2
project_root = Path(str(_repo_root))
land_use_path = project_root / "data" / "land_use_masks.npy"
dataset = radar_dataset_multimodal(
    str(project_root / "data" / "70km" / "png"),
    str(project_root / "data" / "environment"),
    list_length=sequence_length,
    total=None,
    num_workers=14,
    land_use_path=str(land_use_path),
    use_land_use=False,
    num_target_steps=num_target_steps,
)
split_idx = int(0.75 * len(dataset))
val_split_idx = int(0.90 * len(dataset))
train_indices = list(range(split_idx))
val_indices = list(range(split_idx, val_split_idx))
test_indices = list(range(val_split_idx, len(dataset)))
assert len(dataset) == 4330, f'Dataset changed: {len(dataset)} samples (expected 4330)'
dataset.set_normalization_stats(json.loads((project_root / 'models' / 'normalization_stats_long.json').read_text()))
# Chronological 75/15/10 split: val is used for early stopping/checkpoint selection,
# test is only ever touched once at the very end.
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
LAMBDA_SECOND_STEP = 0.25


def build_rollout_inputs(inputs, pred_full):
    """Shift the input window forward, replacing the oldest frame with the model's
    own +5 radar prediction plus the latest known (persisted) environment channels.
    """
    latest_env = inputs[:, -1, 1:, :, :]
    next_frame = torch.cat([pred_full, latest_env], dim=1)
    return torch.cat([inputs[:, 1:, :, :, :], next_frame.unsqueeze(1)], dim=1)



model2 = ConvLSTM_MM(input_dim=7,
                    hidden_dim=[32, 64],
                    kernel_size=[(3, 3), (3, 3)],
                    num_layers=2,
                    batch_first=True,
                    bias=True,
                    use_land_use=False)

model2 = model2.to(device)
model2.load_state_dict(torch.load(str(_repo_root / 'models/model_best_long.pkl'), map_location=device))
print("Loaded production model_best_long.pkl into model2")
# HELD-OUT TEST: frozen p=0.40, minimum size=40 for both leads
# Uses model2 and the existing chronological test_loader; no threshold tuning.
from scipy import ndimage

selected_threshold_test = 0.40
selected_component_size_test = 40

def horizon_clean_mask(probability, threshold, minimum_size):
    cleaned = []
    for sample in probability.numpy():
        components, _ = ndimage.label(sample >= threshold, structure=np.ones((3, 3)))
        keep = np.bincount(components.ravel()) >= minimum_size
        keep[0] = False
        cleaned.append(torch.from_numpy(keep[components]))
    return torch.stack(cleaned)


def horizon_accumulate(stats, mask, prediction, target):
    rain = target > 0.01
    dry = ~rain
    stats['tp'] += (mask & rain).sum().item()
    stats['fp'] += (mask & dry).sum().item()
    stats['fn'] += (~mask & rain).sum().item()
    stats['rain_n'] += rain.sum().item()
    stats['dry_n'] += dry.sum().item()
    error = (prediction - target).abs()
    stats['rain_error'] += error[rain].sum().item()
    stats['dry_error'] += error[dry].sum().item()
    stats['dry_wet'] += (prediction[dry] > 0.01).sum().item()


def horizon_ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


horizon_totals = {
    (lead, method): dict.fromkeys(
        ['tp', 'fp', 'fn', 'rain_n', 'dry_n', 'rain_error', 'dry_error', 'dry_wet'], 0
    )
    for lead in (5, 10)
    for method in ('Model mask', 'Model intensity > .01', 'Persistence')
}
model2.eval()
heldout_sample_count = 0
with torch.no_grad():
    for test_batch, (metric_inputs, metric_targets) in enumerate(test_loader, 1):
        if metric_targets.ndim != 4 or metric_targets.shape[1] != 2:
            raise ValueError('Expected targets [batch, 2, height, width].')
        metric_inputs = metric_inputs.to(device)
        raw5, _, _, logits5 = model2(metric_inputs, return_logits=True)
        # Raw, ungated +5 goes into the +10 rollout, exactly as in training.
        raw10, _, _, logits10 = model2(
            build_rollout_inputs(metric_inputs, raw5), return_logits=True
        )
        persistence = metric_inputs[:, -1, 0].cpu().clamp_min(0)
        for index, (lead, raw, logits) in enumerate(((5, raw5, logits5), (10, raw10, logits10))):
            target = metric_targets[:, index].cpu()
            probability = logits.sigmoid().squeeze(1).cpu()
            raw = raw.squeeze(1).cpu().clamp_min(0)
            mask = horizon_clean_mask(probability, selected_threshold_test, selected_component_size_test)
            gated = raw * mask
            horizon_accumulate(horizon_totals[lead, 'Model mask'], mask, gated, target)
            horizon_accumulate(horizon_totals[lead, 'Model intensity > .01'], gated > 0.01, gated, target)
            horizon_accumulate(horizon_totals[lead, 'Persistence'], persistence > 0.01, persistence, target)
        heldout_sample_count += len(metric_targets)
        if test_batch % 5 == 0:
            print(f'Test progress: {test_batch}/{len(test_loader)} batches', flush=True)

if not heldout_sample_count:
    raise RuntimeError('Test loader is empty.')

heldout_test_results = []
for (lead, method), stats in horizon_totals.items():
    tp, fp, fn = (stats[key] for key in ('tp', 'fp', 'fn'))
    heldout_test_results.append({
        'lead_min': lead, 'method': method,
        'precision': horizon_ratio(tp, tp + fp),
        'recall': horizon_ratio(tp, tp + fn),
        'CSI': horizon_ratio(tp, tp + fp + fn),
        'F1': horizon_ratio(2 * tp, 2 * tp + fp + fn),
        'rainy_MAE': horizon_ratio(stats['rain_error'], stats['rain_n']),
        'dry_MAE': horizon_ratio(stats['dry_error'], stats['dry_n']),
        'dry_gt_001': horizon_ratio(stats['dry_wet'], stats['dry_n']),
    })
print(f'HELD-OUT TEST: {heldout_sample_count} samples | loaded best checkpoint (model2)')
print(f'Both leads: probability >= {selected_threshold_test:.2f}, component size >= {selected_component_size_test}')
print('Pixel-pooled metrics; MAE uses normalized radar intensity, not mm/hr.')
print('Model mask scores the cleaned probability mask; Model intensity scores the final gated radar.')
print('Persistence repeats the last preprocessed input radar at both horizons.')
print(pd.DataFrame(heldout_test_results).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

Path('heldout_test_results.json').write_text(json.dumps(heldout_test_results, indent=2))
