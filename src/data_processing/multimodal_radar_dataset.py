from pathlib import Path
import math

import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import label

from .data_loading import load_data_multimodal, create_samples

class radar_dataset_multimodal(Dataset):
    def __init__(self, folderloc_radar, folderloc_env, total, list_length=10,
                 min_list_length=10, num_workers=None, land_use_path=None):
        min_list_length = list_length

        self.channel_map = {
            "radar": 0,
            "temperature": 1,
            "humidity": 2,
            "wind_u": 3,
            "wind_v": 4,
            "station_mask": 5,
            "distance": 6,
        }
        self.zscore_channels = [
            self.channel_map["temperature"],
            self.channel_map["humidity"],
            self.channel_map["wind_u"],
            self.channel_map["wind_v"],
        ]
        self._zscore_mean = None
        self._zscore_std = None
        self._distance_scale = None
        self._zscore_clip = 4.0
        self._remove_persistent_echoes = True
        self._persistent_rain_threshold = 0.01
        self._persistent_min_pixels = 4
        self._persistent_strong_threshold = 0.10

        if land_use_path is None:
            land_use_path = Path(__file__).resolve().parents[2] / "data" / "land_use_masks.npy"
        self.land_use_masks = torch.from_numpy(
            np.load(land_use_path).astype(np.float32)
        )
        if self.land_use_masks.ndim != 3:
            raise ValueError("land_use_masks must have shape [classes, H, W]")
        
        self.data = load_data_multimodal(
            folder_path_radar=folderloc_radar,
            folder_path_env=folderloc_env,
            total=total,
            num_workers=num_workers,
        )
        self.x, self.y = create_samples(self.data,  list_length=list_length, min_list_length = min_list_length)
        if self._remove_persistent_echoes:
            self._remove_persistent_input_echoes()
        if len(self.x) > 0:
            first_sample = torch.stack(self.x[0]).float()
            height, width = first_sample.shape[-2], first_sample.shape[-1]
            self._distance_scale = float(math.sqrt((height - 1) ** 2 + (width - 1) ** 2))

    def _remove_persistent_input_echoes(self):
        """Remove weak components of at least four pixels with identical masks."""
        structure = np.ones((3, 3), dtype=np.uint8)
        removed_components = 0

        for sample_index, frames in enumerate(self.x):
            radar = torch.stack(frames).float()[:, self.channel_map["radar"]].numpy()
            rain_masks = radar > self._persistent_rain_threshold
            first_labels, component_count = label(rain_masks[0], structure=structure)

            for component_id in range(1, component_count + 1):
                component = first_labels == component_id
                if int(component.sum()) < self._persistent_min_pixels:
                    continue

                same_component = True
                for timestep in range(1, radar.shape[0]):
                    timestep_labels, _ = label(rain_masks[timestep], structure=structure)
                    point_y, point_x = np.argwhere(component)[0]
                    timestep_component_id = timestep_labels[point_y, point_x]
                    if timestep_component_id == 0 or not np.array_equal(
                        component,
                        timestep_labels == timestep_component_id,
                    ):
                        same_component = False
                        break

                if not same_component:
                    continue
                if max(float(radar[timestep][component].max()) for timestep in range(radar.shape[0])) >= self._persistent_strong_threshold:
                    continue

                for frame in frames:
                    frame[self.channel_map["radar"]][component] = 0.0
                removed_components += 1

        if removed_components:
            print(
                f"Removed {removed_components} persistent weak input components "
                f"with {self._persistent_min_pixels}+ pixels"
            )

    def fit_normalization(self, sample_indices):
        if len(self.x) == 0:
            raise ValueError("Cannot fit normalization on an empty dataset")
        if not sample_indices:
            raise ValueError("sample_indices must not be empty")

        channel_sum = torch.zeros(len(self.zscore_channels), dtype=torch.float64)
        channel_sq_sum = torch.zeros(len(self.zscore_channels), dtype=torch.float64)
        total_count = 0

        for idx in sample_indices:
            sample = torch.stack(self.x[idx]).float()
            if sample.shape[1] < 7:
                raise ValueError("Expected at least 7 dynamic channels before land-use concatenation")

            values = sample[:, self.zscore_channels, :, :].reshape(len(self.zscore_channels), -1).double()
            channel_sum += values.sum(dim=1)
            channel_sq_sum += (values * values).sum(dim=1)
            total_count += values.shape[1]

        if total_count == 0:
            raise ValueError("No values found while fitting normalization")

        means = channel_sum / total_count
        variances = (channel_sq_sum / total_count) - (means * means)
        stds = torch.sqrt(torch.clamp(variances, min=1e-8))

        first_sample = torch.stack(self.x[sample_indices[0]]).float()
        height, width = first_sample.shape[-2], first_sample.shape[-1]
        distance_scale = math.sqrt((height - 1) ** 2 + (width - 1) ** 2)

        self._zscore_mean = means.float()
        self._zscore_std = stds.float()
        self._distance_scale = float(distance_scale)

    def set_normalization_stats(self, stats):
        if not stats:
            self._zscore_mean = None
            self._zscore_std = None
            self._distance_scale = None
            return

        ordered_names = ["temperature", "humidity", "wind_u", "wind_v"]
        means = [float(stats[name]["mean"]) for name in ordered_names]
        stds = [max(float(stats[name]["std"]), 1e-4) for name in ordered_names]

        self._zscore_mean = torch.tensor(means, dtype=torch.float32)
        self._zscore_std = torch.tensor(stds, dtype=torch.float32)
        self._distance_scale = float(stats["distance"]["scale"])

    def get_normalization_stats(self):
        if self._zscore_mean is None or self._zscore_std is None or self._distance_scale is None:
            raise ValueError("Normalization stats are not initialized")

        return {
            "temperature": {
                "mean": float(self._zscore_mean[0].item()),
                "std": float(self._zscore_std[0].item()),
            },
            "humidity": {
                "mean": float(self._zscore_mean[1].item()),
                "std": float(self._zscore_std[1].item()),
            },
            "wind_u": {
                "mean": float(self._zscore_mean[2].item()),
                "std": float(self._zscore_std[2].item()),
            },
            "wind_v": {
                "mean": float(self._zscore_mean[3].item()),
                "std": float(self._zscore_std[3].item()),
            },
            "distance": {
                "scale": float(self._distance_scale),
            },
            "zscore_clip": float(self._zscore_clip),
        }

    def _normalize_dynamic_channels(self, sample):
        # sample: [T, 7, H, W]
        distance_idx = self.channel_map["distance"]

        if self._zscore_mean is not None and self._zscore_std is not None:
            mean = self._zscore_mean.view(1, -1, 1, 1)
            std = self._zscore_std.view(1, -1, 1, 1)
            sample[:, self.zscore_channels, :, :] = (
                sample[:, self.zscore_channels, :, :] - mean
            ) / std
            sample[:, self.zscore_channels, :, :] = torch.clamp(
                sample[:, self.zscore_channels, :, :],
                min=-self._zscore_clip,
                max=self._zscore_clip,
            )

        if self._distance_scale is not None and self._distance_scale > 0:
            sample[:, distance_idx, :, :] = sample[:, distance_idx, :, :] / self._distance_scale
            sample[:, distance_idx, :, :] = torch.clamp(sample[:, distance_idx, :, :], min=0.0, max=1.0)

        return sample
        
        
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        target = self.y[idx]

        # sample: [T, 7, H, W] before adding static land-use features
        sample = torch.stack(sample).float()
        sample = self._normalize_dynamic_channels(sample)
        if sample.shape[-2:] != self.land_use_masks.shape[-2:]:
            raise ValueError(
                "Radar/environment and land-use grids must have identical spatial dimensions"
            )
        land_use = self.land_use_masks.unsqueeze(0).expand(sample.shape[0], -1, -1, -1)
        sample = torch.cat([sample, land_use], dim=1)

        # Predict next radar frame only (channel 0), shape [H, W]
        target = target[0].float()
        return sample, target