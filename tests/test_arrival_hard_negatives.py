import unittest,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.data import ArrivalDataset
from data_processing.radar_codec import SOURCE

class HardNegativeTests(unittest.TestCase):
    def test_curriculum_schedule_and_warmup_matches_baseline(self):
        ds=self.dataset();ds.hard_negative_fraction=.1
        ds.hard_negative_warmup=5000;ds.hard_negative_ramp=5000
        for step,expected in [(0,0),(4999,0),(5000,0),(7500,.05),(10000,.1),(15000,.1)]:
            ds.set_step(step);self.assertAlmostEqual(ds.effective_hard_fraction(),expected)
        baseline=self.dataset();baseline.hard_negative_fraction=0
        ds.set_step(0)
        with patch('arrival.data.input_crop',side_effect=lambda store,key,y,x,*args:(key,y,x)):
            for i in range(200):self.assertEqual(ds[i],baseline[i])
        self.assertEqual(ds.sample_counts['hard_branch'],0)

    def dataset(self,rain=True):
        radar=np.zeros((16,16),np.float32)
        if rain:radar[5:8,5:8]=.5
        targets=np.full((16,16),-1)
        targets[0,:3]=[0,3,6];targets[9,9]=12;targets[15,15]=12
        config=SimpleNamespace(max_history=6,patch=5,coverage=.2,rain_threshold=.01,crop=64)
        labels=SimpleNamespace(config=config,get=lambda key:targets)
        store=SimpleNamespace(radar=lambda key:radar)
        return ArrivalDataset(store,labels,['a'],dict(decoder=SOURCE),hard_negative_fraction=.5)

    def test_hard_mask_and_sampling_counts(self):
        ds=self.dataset();mask=ds.hard_candidates('a')
        self.assertTrue(mask[9,9]);self.assertFalse(mask[15,15]);self.assertEqual(mask.sum(),1)
        with patch('arrival.data.input_crop',return_value=None):
            for i in range(4000):ds[i]
        counts=ds.sample_counts
        self.assertTrue(all(abs(n/4000-.25)<.04 for n in counts['groups']))
        self.assertAlmostEqual(counts['hard_branch']/counts['groups'][3],.5,delta=.06)

    def test_empty_hard_pool_fails(self):
        with self.assertRaisesRegex(ValueError,'No rain-adjacent'):self.dataset(False).build_group_index()

    def test_fixed_evaluation_unchanged(self):
        ds=self.dataset();ds.records=[('a',15,15)]
        with patch('arrival.data.input_crop',return_value=None):self.assertEqual(ds[0][1],12)
        self.assertEqual(sum(ds.sample_counts['groups']),0)
