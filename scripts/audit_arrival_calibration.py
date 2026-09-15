"""Offline v5 calibration audit. No training, checkpoint edits or deployment."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.data import DataConfig,FrameStore,LabelStore,ArrivalDataset,fingerprint
from arrival.training import load_model,predict,fit_calibrator,calibrated_probs
from arrival.ranking import precision_recall
from arrival.calibration import fit_ordered_calibrator

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'models/arrival/runs/gru_t6_all_seed67_v5'
OUT=ROOT/'reports/arrival_v5_calibration';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)

def binary_fit(p,y):
    x=torch.logit(torch.tensor(p,dtype=torch.float64).clamp(1e-10,1-1e-10))
    y=torch.tensor(y,dtype=torch.float64)
    a=torch.zeros((),dtype=torch.float64,requires_grad=True)
    b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([a,b],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(x*a.exp().clamp(.05,20)+b,y)
        loss=loss+1e-4*(a*a+b*b);loss.backward();return loss
    opt.step(closure)
    return [float(a.exp().clamp(.05,20).detach()),float(b.detach())]

def apply(p,params):
    x=np.log(np.clip(p,1e-10,1-1e-10)/np.clip(1-p,1e-10,1))
    return 1/(1+np.exp(-np.clip(x*params[0]+params[1],-700,700)))

def summarize(p,y,days):
    result={}
    for i,h in enumerate((15,30,60)):
        t=y<h//5;q=p[:,i]
        bins=[]
        for indices in np.array_split(np.argsort(q),10):
            if len(indices):bins.append(dict(n=len(indices),predicted=float(q[indices].mean()),observed=float(t[indices].mean())))
        byday=[]
        for day in sorted(set(days)):
            m=days==day
            byday.append(dict(day=day,n=int(m.sum()),arrivals=int(t[m].sum()),brier=float(np.mean((q[m]-t[m])**2))))
        r=precision_recall(q,t)
        result[h]=dict(brier=float(np.mean((q-t)**2)),average_precision=r['average_precision'],
            prevalence=float(t.mean()),reliability=bins,by_day=byday)
    result['crossing_fraction']=float(np.mean(np.any(np.diff(p,axis=1)<-1e-9,axis=1)))
    return result

def main():
    prepared=json.loads((ROOT/'models/arrival/data_v1/prepared.json').read_text())
    records=json.loads((ROOT/'models/arrival/data_v1/evaluation_locations.json').read_text())
    model,state=load_model(RUN/'best.pt','cuda' if torch.cuda.is_available() else 'cpu')
    assert state['validation_records_hash']==fingerprint(records['validation'])
    assert state['norm']==prepared['norm']
    store=FrameStore(ROOT);config=DataConfig(**prepared['config']);labels=LabelStore(store,config)
    checkpoint_hash=hashlib.sha256((RUN/'best.pt').read_bytes()).hexdigest()
    sets={}
    for split in ('calibration','validation'):
        tag=fingerprint([checkpoint_hash,records[split]])[:16];cache=OUT/f'{split}_{tag}.npz'
        if cache.exists():
            d=np.load(cache);lp=torch.from_numpy(d['lp']);y=d['y']
        else:
            ds=ArrivalDataset(store,labels,prepared['manifests'][split],prepared['norm'],records=records[split])
            print('Predicting',split,len(ds),flush=True)
            lp,yt=predict(model,ds);y=yt.numpy();np.savez_compressed(cache,lp=lp.numpy(),y=y)
        sets[split]=(lp,y,np.array([r[0][:8] for r in records[split]]))
    clp,cy,cd=sets['calibration'];vlp,vy,vd=sets['validation']
    cp=clp.softmax(1).numpy().cumsum(1)[:,[2,5,11]]
    vp=vlp.softmax(1).numpy().cumsum(1)[:,[2,5,11]]
    params=[binary_fit(cp[:,i],cy<h//5) for i,h in enumerate((15,30,60))]
    independent=np.column_stack([apply(vp[:,i],params[i]) for i in range(3)])
    shared=json.loads((RUN/'calibrator.json').read_text())
    sharedp=calibrated_probs(vlp,shared).cumsum(1)[:,[2,5,11]]
    ordered=fit_ordered_calibrator(clp,torch.from_numpy(cy))
    orderedp=calibrated_probs(vlp,ordered).cumsum(1)[:,[2,5,11]]
    # Whole-day out-of-fold checks: fit only on other calibration dates.
    days=sorted(set(cd));folds=[days[i::3] for i in range(3)]
    oof=np.zeros_like(cp);oof_shared=np.zeros_like(cp);oof_ordered=np.zeros_like(cp)
    for fold in folds:
        mask=np.isin(cd,fold);train=~mask
        for i,h in enumerate((15,30,60)):
            oof[mask,i]=apply(cp[mask,i],binary_fit(cp[train,i],cy[train]<h//5))
        fitted=fit_calibrator(clp[train],torch.from_numpy(cy[train]))
        oof_shared[mask]=calibrated_probs(clp[mask],fitted).cumsum(1)[:,[2,5,11]]
        fitted_ordered=fit_ordered_calibrator(clp[train],torch.from_numpy(cy[train]))
        oof_ordered[mask]=calibrated_probs(clp[mask],fitted_ordered).cumsum(1)[:,[2,5,11]]
    result=dict(checkpoint_sha256=checkpoint_hash,checkpoint_step=state['step'],
        caveat='Validation selected the checkpoint. Calibration OOF is by date, not guaranteed independent storms. Independent horizon maps are diagnostic only; crossing probabilities cannot be exported as a coherent arrival distribution.',
        independent_parameters=params,
        validation_raw=summarize(vp,vy,vd),validation_shared=summarize(sharedp,vy,vd),
        validation_independent=summarize(independent,vy,vd),
        validation_ordered=summarize(orderedp,vy,vd),calibration_oof_ordered=summarize(oof_ordered,cy,cd),
        calibration_oof_shared=summarize(oof_shared,cy,cd),calibration_oof_independent=summarize(oof,cy,cd))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    (OUT/'ordered_calibrator_candidate.json').write_text(json.dumps(ordered,indent=2))
    lines=['# Arrival v5 calibration audit','',result['caveat'],'',
        '| Evaluation | Horizon | Brier (lower better) | AP |','|---|---:|---:|---:|']
    for name,r in result.items():
        if not isinstance(r,dict) or 'crossing_fraction' not in r:continue
        for h in (15,30,60):lines.append(f"| {name} | {h} | {r[h]['brier']:.5f} | {r[h]['average_precision']:.3f} |")
        print(name,{h:round(r[h]['brier'],5) for h in (15,30,60)},'crossings',r['crossing_fraction'],flush=True)
        lines.append('')
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
