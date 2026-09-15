"""Offline arrival warning replay at nine fixed crop-valid radar locations."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from arrival.data import DataConfig,FrameStore,LabelStore,input_crop,tick_time,shifted
from arrival.training import load_model
from arrival.calibration import project_cumulative
from experiment_arrival_projection import apply


def score_run(keys,wet,probabilities,horizon,threshold,confirmations=1):
    """At most one outstanding warning; expires at horizon or resolves on onset.

    Missing frames are handled by the caller splitting runs. Only onsets with
    a full forecast horizon in the run count in event recall.
    """
    if confirmations<1:raise ValueError('confirmations must be positive')
    h=horizon//5;alerts=[];onsets=[];pending=None;expiry=-1;matched=set();streak=0
    for i,key in enumerate(keys):
        streak=streak+1 if not wet[i] and probabilities[i]>=threshold else 0
        if i and wet[i] and not wet[i-1]:
            if i>=h:onsets.append(i)
            if pending is not None and i<=expiry:
                alerts[pending]['hit']=True;alerts[pending]['lead_minutes']=(i-alerts[pending]['index'])*5
                matched.add(i);pending=None
        if pending is not None and i>=expiry:pending=None
        if not wet[i] and pending is None and i+h<len(keys) and streak>=confirmations:
            pending=len(alerts);expiry=i+h
            alerts.append(dict(time=key,index=i,hit=False,lead_minutes=None))
    leads=[a['lead_minutes'] for a in alerts if a['hit']]
    return dict(alerts=len(alerts),true_alerts=len(leads),false_alerts=len(alerts)-len(leads),
        onsets=len(onsets),warned_onsets=sum(i in matched for i in onsets),lead_minutes=leads,
        location_hours=max(len(keys)-h,0)/12,alert_records=alerts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'models/arrival/runs/gru_t6_all_seed67_v5'))
    parser.add_argument('--calibration',default=str(ROOT/'reports/arrival_v5_calibration/projection_experiment.json'))
    parser.add_argument('--output',default=str(ROOT/'reports/arrival_notification_replay'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    run=Path(args.run)
    candidate=json.loads(Path(args.calibration).read_text())
    digest=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    assert digest==candidate['checkpoint_sha256']
    data=json.loads((ROOT/'models/arrival/data_v1/prepared.json').read_text())
    config=DataConfig(**data['config']);keys=data['manifests']['validation']
    store=FrameStore(ROOT);labels=LabelStore(store,config)
    model,state=load_model(run/'best.pt','cuda' if torch.cuda.is_available() else 'cpu')
    assert state['norm']==data['norm']
    H,W=store.radar(keys[0]).shape
    locations=[(int(y),int(x)) for y in np.linspace(32,H-32,5)[1:-1] for x in np.linspace(32,W-32,5)[1:-1]]
    cache=out/'forecasts.npz';meta=dict(checkpoint=digest,keys=keys,locations=locations,parameters=candidate['parameters'])
    tag=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    if cache.exists() and (out/'cache_tag.txt').read_text()==tag:
        d=np.load(cache);wet=d['wet'];probs=d['probs']
    else:
        wet=np.zeros((len(keys),len(locations)),dtype=bool)
        probs=np.zeros((len(keys),len(locations),3))
        with torch.inference_mode():
            for i,key in enumerate(keys):
                radar=store.radar(key)
                for j,(y,x) in enumerate(locations):wet[i,j]=int((radar[y-2:y+3,x-2:x+3]>.01).sum())>=5
                dry=np.flatnonzero(~wet[i])
                if len(dry):
                    batch=torch.stack([input_crop(store,key,*locations[j],data['norm']) for j in dry])
                    raw=model.log_probs(batch.to(next(model.parameters()).device)).double().softmax(1).cpu().numpy().cumsum(1)[:,:12]
                    calibrated=np.column_stack([apply(raw[:,k],candidate['parameters'][k]) for k in range(12)])
                    probs[i,dry]=project_cumulative(calibrated)[:,[2,5,11]]
                if i%200==0:print(f'Forecasted {i}/{len(keys)} timestamps',flush=True)
        np.savez_compressed(cache,wet=wet,probs=probs);(out/'cache_tag.txt').write_text(tag)
    # Restrict scoring to consecutive usable inputs. Exclude boundary-censored
    # alerts and onsets; missing forecasts do not imply dry conditions.
    starts=[0]+[i for i in range(1,len(keys)) if tick_time(keys[i])-tick_time(keys[i-1])!=timedelta(minutes=5)]+[len(keys)]
    rows=[]
    for hi,horizon in enumerate((15,30,60)):
        for threshold in (.01,.02,.05,.1,.2,.3,.5):
            total=dict(alerts=0,true_alerts=0,false_alerts=0,onsets=0,warned_onsets=0,location_hours=0.)
            leads=[]
            for j in range(len(locations)):
                for a,b in zip(starts[:-1],starts[1:]):
                    r=score_run(keys[a:b],wet[a:b,j],probs[a:b,j,hi],horizon,threshold)
                    for k in total:total[k]+=r[k]
                    leads+=r['lead_minutes']
            total.update(horizon=horizon,threshold=threshold,
                precision=total['true_alerts']/max(total['alerts'],1),
                onset_recall=total['warned_onsets']/max(total['onsets'],1),
                median_lead_minutes=float(np.median(leads)) if leads else None,
                false_alerts_per_location_day=total['false_alerts']/max(total['location_hours']/24,1e-9))
            rows.append(total)
    result=dict(settings=meta,runs=len(starts)-1,timestamps=len(keys),results=rows,
        caveat='Development replay on validation used to select checkpoint. Nine synthetic fixed radar-pixel locations, not actual users. Observed rain uses cleaned source 5/25 pixel criterion. No delivery latency. Gaps split episodes; boundary events excluded. Suppress repeat starts until observed rain or horizon expiry; no predicted-clear notifications. This is a proposed arrival warning policy, not production bot behavior.')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# Arrival notification replay','',result['caveat'],'',f'{len(keys)} timestamps; {len(starts)-1} uninterrupted runs; nine fixed locations.','',
        '| Horizon | Threshold | Alerts | Precision | Onsets warned | Median lead | False/location/day |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        line=f"| {r['horizon']} | {r['threshold']} | {r['alerts']} | {r['precision']:.1%} | {r['warned_onsets']}/{r['onsets']} | {r['median_lead_minutes']} | {r['false_alerts_per_location_day']:.2f} |"
        lines.append(line);print(line)
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
