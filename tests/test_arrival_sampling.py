import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.data import ArrivalDataset
from arrival.ranking import precision_recall


class SamplingTests(unittest.TestCase):
    def test_cumulative_projection_pools_violations(self):
        from arrival.calibration import project_cumulative
        x=np.array([[.4,.2,.9],[.1,.2,.3],[.9,.6,.3]])
        y=project_cumulative(x)
        np.testing.assert_allclose(y,[[.3,.3,.9],[.1,.2,.3],[.6,.6,.6]])
        np.testing.assert_allclose(project_cumulative(y),y)
        self.assertTrue(np.all(np.diff(y,axis=1)>=0))
        p=np.diff(np.column_stack([np.zeros(3),y,np.ones(3)]),axis=1)
        self.assertTrue(np.all(p>=0));np.testing.assert_allclose(p.sum(1),1)

    def test_ordered_calibrator_is_coherent_and_preserves_ranking(self):
        import torch
        from arrival.calibration import fit_ordered_calibrator
        from arrival.training import calibrated_probs
        torch.manual_seed(7)
        lp=torch.randn(80,13,dtype=torch.float64).log_softmax(1)
        y=torch.arange(80)%13
        p=calibrated_probs(lp,fit_ordered_calibrator(lp,y))
        self.assertTrue(np.all(p>=0));np.testing.assert_allclose(p.sum(1),1)
        for k in range(12):
            np.testing.assert_array_equal(np.argsort(p.cumsum(1)[:,k]),np.argsort(lp.exp().numpy().cumsum(1)[:,k]))

    def test_calibration_preserves_ranking_and_valid_distribution(self):
        import torch
        from arrival.training import fit_calibrator, calibrated_probs
        torch.manual_seed(67)
        lp=torch.randn(100,13,dtype=torch.float64).log_softmax(1)
        y=torch.cat([torch.full((80,),12),torch.arange(20)%12])
        calibrator=fit_calibrator(lp,y)
        p=calibrated_probs(lp,calibrator)
        self.assertTrue(np.all(p>=0))
        np.testing.assert_allclose(p.sum(1),1)
        raw=lp.exp().numpy().cumsum(1); cumulative=p.cumsum(1)
        for k in range(12):
            np.testing.assert_array_equal(np.argsort(raw[:,k]),np.argsort(cumulative[:,k]))
        self.assertTrue(np.all(np.diff(cumulative,axis=1)>=0))

    def test_global_balance_despite_many_dry_anchors(self):
        ds=ArrivalDataset.__new__(ArrivalDataset)
        ds.records=None;ds.anchors=list(range(100));ds.seed=67;ds.epoch=0
        ds.hard_negative_fraction=0.;ds.sample_counts={'groups':[0,0,0,0],'hard_branch':0}
        ds.store=None;ds.norm=None;ds.history=6;ds.channels=(0,)
        ds.labels=SimpleNamespace(config=SimpleNamespace(crop=64),
            get=lambda key: np.array([[0,3,6,12]]) if key==0 else np.array([[12]]))
        with patch('arrival.data.input_crop',return_value=None):
            targets=[ds[i][1] for i in range(4000)]
        counts=np.array([targets.count(k) for k in (0,3,6,12)])
        self.assertTrue(np.all(abs(counts/4000-.25)<.035),counts)

    def test_ties_and_perfect_ranking(self):
        self.assertAlmostEqual(precision_recall([.2]*4,[1,0,0,0])['average_precision'],.25)
        self.assertEqual(precision_recall([.9,.1],[1,0])['average_precision'],1.)
        self.assertIsNone(precision_recall([.9,.1],[0,0])['average_precision'])


if __name__=='__main__':unittest.main()
