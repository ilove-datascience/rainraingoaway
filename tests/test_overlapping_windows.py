import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_processing.data_loading import create_samples, chronological_frame_partitions
from data_processing.multimodal_radar_dataset import radar_dataset_multimodal


class OverlappingTests(unittest.TestCase):
    def test_stride_and_gap(self):
        start = datetime(2026, 1, 1)
        data = {(start + timedelta(minutes=5*i)).strftime('%Y%m%d%H%M'): np.full((7, 3, 3), i, dtype=np.float32)
                for i in list(range(8)) + list(range(9, 14))}
        x, y = create_samples(data, list_length=3, num_target_steps=2, stride=1)
        self.assertEqual(len(x), 5)
        self.assertEqual([int(target[0][0, 0, 0]) for target in y], [3, 4, 5, 6, 12])
        self.assertEqual(len(create_samples(data, list_length=3, num_target_steps=2)[0]), 2)
        with self.assertRaises(ValueError):
            create_samples(data, stride=0)

    def test_whole_dates_disjoint(self):
        data = {f'202601{day:02d}{hour:02d}00': None for day in range(1, 21) for hour in (0, 23)}
        parts = chronological_frame_partitions(data)
        self.assertEqual([len(p) for p in parts.values()], [30, 6, 4])
        sets = [set(key[:8] for key in p) for p in parts.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_cleaning_does_not_mutate_shared_target(self):
        frame = torch.zeros(7, 4, 4)
        frame[0, :2, :2] = .05
        dataset = radar_dataset_multimodal.__new__(radar_dataset_multimodal)
        dataset.x = [[frame, frame, frame]]
        dataset.channel_map = {'radar': 0}
        dataset._persistent_rain_threshold = .01
        dataset._persistent_min_pixels = 4
        dataset._persistent_strong_threshold = .1
        dataset._remove_persistent_input_echoes()
        self.assertTrue(torch.all(frame[0, :2, :2] == .05))
        self.assertEqual(float(dataset.x[0][0][0].sum()), 0)


if __name__ == '__main__':
    unittest.main()
