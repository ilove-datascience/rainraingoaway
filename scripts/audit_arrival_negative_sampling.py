"""Estimate current no-arrival sampler exposure to rain-adjacent negatives."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve,distance_transform_edt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arrival.data import FrameStore,LabelStore,DataConfig

def main():
    prepared=json.loads((ROOT/'models/arrival/data_v1/prepared.json').read_text())
    store=FrameStore(ROOT);labels=LabelStore(store,DataConfig(**prepared['config']))
    anchors=prepared['manifests']['train'];rng=np.random.default_rng(67)
    keys=sorted(rng.choice(anchors,min(256,len(anchors)),replace=False))
    rows=[]
    for key in keys:
        target=labels.get(key);dry=target==12
        wet=convolve((store.radar(key)>.01).astype(np.int16),np.ones((5,5),np.int16),mode='constant')>=5
        distance=distance_transform_edt(~wet) if wet.any() else np.full(wet.shape,np.inf)
        if dry.any():rows.append(dict(key=key,no_arrival=int(dry.sum()),near5=int((dry&(distance<=5)).sum()),near10=int((dry&(distance<=10)).sum())))
    result=dict(sampled_training_anchors=len(keys),eligible_negative_anchors=len(rows),
        near5_fraction_under_current_negative_sampling=float(np.mean([r['near5']/r['no_arrival'] for r in rows])),
        near10_fraction_under_current_negative_sampling=float(np.mean([r['near10']/r['no_arrival'] for r in rows])),
        fraction_negative_anchors_with_near5=float(np.mean([r['near5']>0 for r in rows])),
        rows=rows,caveat='Seeded subset of training anchors. Fractions average per anchor, matching uniform eligible-anchor then uniform location negative sampling. Not a complete dataset census.')
    out=ROOT/'reports/arrival_spatial_diagnosis';out.mkdir(parents=True,exist_ok=True)
    (out/'training_negatives.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
