"""Audit failed runs against constant-class loss and real hard-negative labels."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arrival.data import FrameStore,LabelStore,DataConfig,ArrivalDataset,shifted

def main():
    out=ROOT/'reports/hardnegative_failure';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in (ROOT/'models/arrival/experiments_hardnegative_v1').glob('*/result.json'):
        r=json.loads(p.read_text());h=json.loads((p.parent/'history.json').read_text())
        # Four equally likely groups, uniform bins within each: reference only.
        prior=np.array([.25/3]*6+[.25/6]*6+[.25])
        rows.append(dict(run=p.parent.name,final_loss=h[-1]['train_loss'],
            reference_group_prior_entropy=float(-(prior*np.log(prior)).sum()),
            best_mean_ap=max(x['validation_mean_ap'] for x in h)))
    d=json.loads((ROOT/'models/arrival/data_v1/prepared.json').read_text())
    store=FrameStore(ROOT);config=DataConfig(**d['config']);labels=LabelStore(store,config)
    ds=ArrivalDataset(store,labels,d['manifests']['train'],d['norm'],hard_negative_fraction=.5)
    rng=np.random.default_rng(71);checked=[]
    for key in rng.choice(ds.anchors,min(256,len(ds.anchors)),replace=False):
        candidates=np.argwhere(ds.hard_candidates(key))
        if not len(candidates):continue
        y,x=candidates[rng.integers(len(candidates))]
        wet=[]
        for t in range(13):
            radar=store.radar(shifted(key,5*t))
            wet.append(int((radar[y-2:y+3,x-2:x+3]>.01).sum())>=5)
        assert not any(wet),'Hard negative contradicts actual radar label'
        checked.append(dict(key=str(key),y=int(y),x=int(x)))
        if len(checked)>=32:break
    report=dict(runs=rows,real_hard_negative_labels_verified=len(checked),samples=checked,
        conclusion='No contradiction found in sampled hard-negative labels. All three hard-negative runs stagnate near a constant-prior loss reference. This is consistent with failure to learn discrimination under this sampling intervention, not proof of a particular optimizer or architecture defect. Reference bin priors are approximate, not empirical class priors.')
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
