"""Diagnose false start warnings using future radar, offline only."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from datetime import timedelta
from replay_arrival_notifications import score_run
from arrival.data import FrameStore,tick_time
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/arrival_spatial_diagnosis';out.mkdir(parents=True,exist_ok=True)
    store=FrameStore(ROOT)
    events=[];masks={}
    for model,folder in [('v5','arrival_notification_replay'),('spatial','arrival_replay_gru_t6_depth2_spatial_seed67')]:
        base=ROOT/'reports'/folder;meta=json.loads((base/'results.json').read_text())
        keys=meta['settings']['keys'];locations=meta['settings']['locations']
        with np.load(base/'forecasts.npz') as d:wet=d['wet'];p=d['probs']
        starts=[0]+[i for i in range(1,len(keys)) if tick_time(keys[i])-tick_time(keys[i-1])!=timedelta(minutes=5)]+[len(keys)]
        for j,(y,x) in enumerate(locations):
            for a,b in zip(starts[:-1],starts[1:]):
                r=score_run(keys[a:b],wet[a:b,j],p[a:b,j,2],60,.3)
                for alert in r['alert_records']:
                    if alert['hit']:continue
                    start=a+alert['index'];minimum=999.;initial=999.;closest_tick=None
                    for idx in range(start,start+13):
                        key=keys[idx]
                        if key not in masks:
                            radar=store.radar(key)
                            coverage=convolve((radar>.01).astype(np.int16),np.ones((5,5),dtype=np.int16),mode='constant')>=5
                            masks[key]=np.argwhere(coverage)
                        coords=masks[key]
                        distance=float(np.sqrt(((coords-[y,x])**2).sum(1)).min()) if len(coords) else 999.
                        if idx==start:initial=distance
                        elif distance<minimum:minimum=distance;closest_tick=key
                    events.append(dict(model=model,location=j,y=y,x=x,time=keys[start],
                        nearest_future_rain_pixels=None if minimum==999 else minimum,
                        nearest_initial_rain_pixels=None if initial==999 else initial,closest_tick=closest_tick,
                        category='within_2_pixels' if minimum<=2 else 'within_5_pixels' if minimum<=5 else 'within_10_pixels' if minimum<=10 else 'beyond_10_pixels'))
        print('Diagnosed',model,flush=True)
    summary={}
    for model in ('v5','spatial'):
        rows=[r for r in events if r['model']==model]
        summary[model]=dict(false_warnings=len(rows),categories={k:sum(r['category']==k for r in rows) for k in ('within_2_pixels','within_5_pixels','within_10_pixels','beyond_10_pixels')},
            location_counts={str(j):sum(r['location']==j for r in rows) for j in range(9)},
            day_counts={day:sum(r['time'][:8]==day for r in rows) for day in sorted({r['time'][:8] for r in rows})})
    (out/'results.json').write_text(json.dumps(dict(summary=summary,events=events,
        caveat='60-minute horizon, threshold .3, one-forecast policy. Distance is to a centre satisfying the same five-of-25-pixel rain criterion, not to individual radar speckles. Pixel distances are not physical distance estimates. Future data used for diagnosis only. This does not justify relabeling near misses as correct forecasts.'),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
