"""Saved-forecast consistency replay and limited timing diagnosis. No deployment."""
import json
from pathlib import Path
import numpy as np
from replay_arrival_notifications import score_run
from arrival.data import tick_time
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1]

def main():
    folders={'v5':'arrival_notification_replay','spatial':'arrival_replay_gru_t6_depth2_spatial_seed67'}
    rows=[];diagnoses=[]
    for model,folder in folders.items():
        base=ROOT/'reports'/folder;meta=json.loads((base/'results.json').read_text())
        keys=meta['settings']['keys']
        with np.load(base/'forecasts.npz') as d:wet=d['wet'];p=d['probs']
        starts=[0]+[i for i in range(1,len(keys)) if tick_time(keys[i])-tick_time(keys[i-1])!=timedelta(minutes=5)]+[len(keys)]
        for hi,h in enumerate((15,30,60)):
            for threshold in (.1,.2,.3,.5):
                for confirmations in (1,2):
                    total={k:0 for k in ('alerts','true_alerts','false_alerts','onsets','warned_onsets','location_hours')};leads=[]
                    late=none=unknown=0
                    for j in range(wet.shape[1]):
                        for a,b in zip(starts[:-1],starts[1:]):
                            w=wet[a:b,j]
                            r=score_run(keys[a:b],w,p[a:b,j,hi],h,threshold,confirmations)
                            for k in total:total[k]+=r[k]
                            leads+=r['lead_minutes']
                            for alert in r['alert_records']:
                                if alert['hit']:continue
                                end=alert['index']+h//5
                                if end+3>=len(w):unknown+=1
                                elif any(w[end+1:end+4]):late+=1
                                else:none+=1
                    rows.append(dict(model=model,horizon=h,threshold=threshold,confirmations=confirmations,**total,
                        precision=total['true_alerts']/max(total['alerts'],1),recall=total['warned_onsets']/max(total['onsets'],1),
                        false_per_day=total['false_alerts']/max(total['location_hours']/24,1e-9),
                        median_lead=float(np.median(leads)) if leads else None))
                    diagnoses.append(dict(model=model,horizon=h,threshold=threshold,confirmations=confirmations,
                        rain_in_next_15_minutes_after_expiry=late,no_rain_through_expiry_plus_15=none,extension_unobserved=unknown))
    out=ROOT/'reports/arrival_warning_consistency';out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(dict(results=rows,timing_diagnosis=diagnoses,
        caveat='Development replay. Same numeric thresholds within each model. Timing diagnosis is at the exact target only; no-rain category does not distinguish nearby passing rain from failed development. Extension-unobserved gaps cannot be classified.'),indent=2))
    lines=['# Consecutive forecast warning experiment','',
        'Development replay; two confirmations mean two consecutive five-minute forecasts exceed the same threshold while currently dry. Gaps reset confirmation. No production changes.','',
        '| Model | Horizon | Threshold | Confirmations | Warned/onsets | False warnings | False/day | Median lead |',
        '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['horizon']} | {r['threshold']} | {r['confirmations']} | {r['warned_onsets']}/{r['onsets']} | {r['false_alerts']} | {r['false_per_day']:.3f} | {r['median_lead']} |")
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    for r in rows:
        if r['horizon']==60:print(r)
    print('Timing diagnoses at 60 minutes:',[d for d in diagnoses if d['horizon']==60 and d['threshold']==.3])

if __name__=='__main__':main()
