"""Descriptive matched-burden comparison; thresholds tuned on development only."""
import sys,json,argparse
from pathlib import Path
import numpy as np
from datetime import timedelta
from replay_arrival_notifications import score_run
from arrival.data import tick_time
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recovery',action='store_true');args=parser.parse_args()
    folders={'v5':ROOT/'reports/arrival_notification_replay',
             'depth2_spatial':ROOT/'reports/arrival_replay_gru_t6_depth2_spatial_seed67'}
    if args.recovery:
        recovery=ROOT/'reports/arrival_recovery_audit'
        folders={'v5':ROOT/'reports/arrival_notification_replay',
                 'fresh_baseline':recovery/'gru_recovery_baseline_seed67',
                 'hard10':recovery/'gru_recovery_hard10_seed67',
                 'curriculum':recovery/'gru_recovery_hard10_curriculum_seed67'}
    bundles={}
    for name,path in folders.items():
        meta=json.loads((path/'results.json').read_text())
        with np.load(path/'forecasts.npz') as saved:
            cache={key:saved[key] for key in ('wet','probs')}
        bundles[name]=(meta,cache)
    first,a=bundles['v5']
    for second,b in bundles.values():
        assert first['settings']['keys']==second['settings']['keys']
        assert first['settings']['locations']==second['settings']['locations']
        np.testing.assert_array_equal(a['wet'],b['wet'])
    keys=first['settings']['keys'];wet=a['wet']
    starts=[0]+[i for i in range(1,len(keys)) if tick_time(keys[i])-tick_time(keys[i-1])!=timedelta(minutes=5)]+[len(keys)]
    # Fixed common dense grid, including a no-warning option. Episode counts can
    # be nonmonotonic in threshold because earlier warnings suppress later ones.
    thresholds=np.unique(np.r_[np.linspace(.001,.999,500),np.geomspace(.00001,.1,100),1.000001])
    curves=[]
    for name,(meta,cache) in bundles.items():
        for hi,h in enumerate((15,30,60)):
            for threshold in thresholds:
                totals=dict(alerts=0,true_alerts=0,false_alerts=0,onsets=0,warned_onsets=0,location_hours=0.)
                leads=[]
                for j in range(wet.shape[1]):
                    for lo,up in zip(starts[:-1],starts[1:]):
                        r=score_run(keys[lo:up],wet[lo:up,j],cache['probs'][lo:up,j,hi],h,float(threshold))
                        for k in totals:totals[k]+=r[k]
                        leads.extend(r['lead_minutes'])
                curves.append(dict(model=name,horizon=h,threshold=float(threshold),**totals,
                    recall=totals['warned_onsets']/max(totals['onsets'],1),
                    precision=totals['true_alerts']/max(totals['alerts'],1),
                    false_per_day=totals['false_alerts']/max(totals['location_hours']/24,1e-9),
                    median_lead=float(np.median(leads)) if leads else None))
            print('Scored',name,h,flush=True)
    chosen=[]
    for h in (15,30,60):
        for cap in (.1,.25,.5,1.):
            for name in bundles:
                eligible=[r for r in curves if r['model']==name and r['horizon']==h and r['false_per_day']<=cap]
                winner=max(eligible,key=lambda r:(r['warned_onsets'],-r['false_alerts'],r['precision'],r['threshold']))
                chosen.append(dict(cap_false_per_day=cap,**winner))
    out=ROOT/'reports'/('arrival_recovery_matched_burden' if args.recovery else 'arrival_matched_burden');out.mkdir(parents=True,exist_ok=True)
    caveat='Thresholds optimized on this development replay: optimistic descriptive frontier, not independent testing or deployment settings. Equal false-warning caps, not necessarily identical achieved rates. Nine synthetic locations; gaps censored; no delivery latency.'
    (out/'comparison.json').write_text(json.dumps(dict(caveat=caveat,selected=chosen,curves=curves),indent=2))
    lines=['# Arrival models at matched false-warning caps','',caveat,'',
        '| Horizon | False/day cap | Model | Threshold | Warned/onsets | Precision | Actual false/day | Median lead |',
        '|---|---|---|---|---|---|---|---|']
    for r in chosen:
        line=f"| {r['horizon']} | {r['cap_false_per_day']} | {r['model']} | {r['threshold']:.4f} | {r['warned_onsets']}/{r['onsets']} | {r['precision']:.1%} | {r['false_per_day']:.3f} | {r['median_lead']} |"
        lines.append(line);print(line)
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
