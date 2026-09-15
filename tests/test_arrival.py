import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime,timedelta
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.data import DataConfig,arrival_map,make_manifests,shifted
from arrival.models import ArrivalNet,hazard_log_probs
from arrival.training import metrics,fit_calibrator,calibrated_probs
from data_processing.data_loading import build_env_data
from data_processing.radar_codec import SOURCE
from arrival.inference import predict_location


class ArrivalTests(unittest.TestCase):
    def test_inference_never_reads_future(self):
        anchor='202601011200';seen=[]
        def radar(key):
            self.assertLessEqual(key,anchor);seen.append(key)
            return np.zeros((16,16),dtype=np.float32)
        def frame(key):
            self.assertLessEqual(key,anchor);seen.append(key)
            return np.zeros((7,16,16),dtype=np.float32)
        store=SimpleNamespace(radar=radar,frame=frame)
        norm=dict(mean=[0]*4,std=[1]*4,clip=4,distance_scale=100,decoder=SOURCE)
        model=ArrivalNet(history=3,kind='cnn')
        result=predict_location(model,store,anchor,8,8,norm,DataConfig(crop=8,max_history=3))
        self.assertEqual(result['status'],'currently_dry')
        self.assertAlmostEqual(sum(result['probabilities']),1,places=5)
        self.assertIn('202601011150',seen)

    def test_first_arrival_and_coverage(self):
        config=DataConfig(crop=8,max_history=3)
        now=np.zeros((16,16));future=[now.copy() for _ in range(12)]
        future[0][7,6:10]=.2  # four pixels: below 20% of 25
        future[2][7,6:11]=.2  # five pixels: t+15 -> class 2
        labels=arrival_map(now,future,config)
        self.assertEqual(labels[8,8],2)
        self.assertEqual(labels[4,4],12)
        self.assertEqual(labels[0,0],-1)
        now[7,6:11]=.2
        self.assertEqual(arrival_map(now,future,config)[8,8],-1)
        with self.assertRaises(ValueError):arrival_map(now,future[:-1],config)

    def test_day_partition_has_no_shared_window_frames(self):
        begin=datetime(2026,1,1)
        keys=[(begin+timedelta(minutes=5*i)).strftime('%Y%m%d%H%M') for i in range(12*288)]
        store=SimpleNamespace(keys=keys,key_set=set(keys),environment_valid=lambda key:True)
        parts=make_manifests(store,DataConfig(crop=8,max_history=3,development_end='2026-01-13'))
        frames={p:{shifted(k,d) for k in anchors for d in range(-10,61,5)} for p,anchors in parts.items()}
        self.assertTrue(all(parts[s] for s in ['train','validation','calibration']))
        self.assertFalse(frames['train']&frames['validation'])
        self.assertFalse(frames['validation']&frames['calibration'])
        self.assertEqual(parts['test'],[])

    def test_hazard_distribution_extremes(self):
        logits=torch.tensor([[100.]*12,[-100.]*12,[0.]*12],requires_grad=True)
        lp=hazard_log_probs(logits)
        torch.testing.assert_close(lp.exp().sum(1),torch.ones(3))
        self.assertEqual(lp[0].argmax(),0);self.assertEqual(lp[1].argmax(),12)
        loss=-lp[:,5].mean();loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_architectures_backward(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for kind in ['cnn','gru','lstm']:
                for output in ['categorical','hazard']:
                    model=ArrivalNet(history=3,channels=2,kind=kind,output=output)
                    logp=model.log_probs(torch.randn(2,3,2,16,16))
                    self.assertEqual(logp.shape,(2,13))
                    torch.testing.assert_close(logp.exp().sum(1),torch.ones(2))
                    (-logp[:,3].mean()).backward()
        finally:torch.set_num_threads(previous)

    def test_arrival_metrics_do_not_hide_no_rain_misses(self):
        p=np.eye(13)[[12,5,2]];truth=np.array([2,3,12])
        m=metrics(p,truth)
        self.assertEqual(m['rainy_cases'],2);self.assertEqual(m['missed_arrivals'],1)
        self.assertEqual(m['false_arrivals'],1);self.assertEqual(m['arrival_mae_minutes_detected'],10)

    def test_calibration_accounts_for_prior_shift(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            scores=torch.full((30,13),-np.log(13));truth=torch.full((30,),12)
            calibrator=fit_calibrator(scores,truth);p=calibrated_probs(scores,calibrator)
            self.assertTrue(np.allclose(p.sum(1),1));self.assertGreater(p[:,12].mean(),1/13)
        finally:torch.set_num_threads(previous)

    def test_weather_future_rows_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weather.csv'
            path.write_text('timestamp,humidity,temperature,wind_dir,wind_speed,latitude,longitude\n'
                '2026-01-01 12:05:00+08:00,80,28,90,2,1.3,103.8\n')
            self.assertIsNone(build_env_data(path,verbose=False,as_of=datetime(2026,1,1,12,0)))
