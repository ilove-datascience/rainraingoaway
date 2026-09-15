"""Continuous five-minute replay with production mask/location/state semantics.

Offline only: no Telegram calls, API calls, SQL, retraining or target cleaning.
"""
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_processing.data_loading import _read_cache, RADAR_CLEANING_VERSION
from data_processing.model_contract import validate_contract, file_hash
from data_processing.radar_codec import LEGACY
from models.multi_modal_convlstm import ConvLSTM_MM
from telegram_code.forecast_mask import clean_rain_mask, RAIN_PROBABILITY_THRESHOLD, MIN_COMPONENT_SIZE
from telegram_code.local_rain import local_rain
from telegram_code.rain_state import START, ENDED
from masking import lowerLat,upperLat,lowerLong,upperLong
from evaluation.notification_replay import replay_location
from evaluation.rain_diagnostics import region_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='reports/source_check/validation_groups.json')
    parser.add_argument('--checkpoint',default='models/model_best_long.pkl')
    parser.add_argument('--normalization',default='models/normalization_stats_long.json')
    parser.add_argument('--locations',help='JSON list of {name,latitude,longitude}; defaults to a 5x5 coverage grid')
    parser.add_argument('--output',default='reports/notification_replay')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    groups=json.loads(Path(args.manifest).read_text())
    assert len(groups)>0
    start=datetime.strptime(groups[0][2],'%Y%m%d%H%M')
    end=datetime.strptime(groups[-1][-1],'%Y%m%d%H%M')
    locations=json.loads(Path(args.locations).read_text()) if args.locations else [
        dict(name=f'grid_{i}_{j}',latitude=float(lat),longitude=float(lon))
        for i,lat in enumerate(np.linspace(lowerLat,upperLat,7)[1:-1])
        for j,lon in enumerate(np.linspace(lowerLong,upperLong,7)[1:-1])]
    validate_contract(args.checkpoint,LEGACY,args.normalization)
    norm=json.loads(Path(args.normalization).read_text())
    mean=torch.tensor([norm[n]['mean'] for n in ['temperature','humidity','wind_u','wind_v']]).view(1,1,4,1,1)
    std=torch.tensor([max(norm[n]['std'],1e-4) for n in ['temperature','humidity','wind_u','wind_v']]).view(1,1,4,1,1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ConvLSTM_MM(input_dim=7,hidden_dim=[32,64],kernel_size=[(3,3),(3,3)],num_layers=2,batch_first=True,bias=True,use_land_use=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint,map_location=device));model.eval()
    cache=Path('data/data/multimodal_cache')
    frames={};skipped=[]
    tick=start-timedelta(minutes=10)
    while tick<=end:
        key=tick.strftime('%Y%m%d%H%M')
        frame=_read_cache(cache/f'{key}.npy',cache/f'{key}.json',None,None,LEGACY)
        if frame is not None:frames[tick]=frame
        tick+=timedelta(minutes=5)
    ticks=[];tick=start
    while tick<=end:
        if all(tick-timedelta(minutes=d) in frames for d in (10,5,0)):
            ticks.append(tick)
        else:skipped.append(tick.isoformat())
        tick+=timedelta(minutes=5)
    if not ticks:raise RuntimeError('No usable continuous validation inputs')
    records=defaultdict(list);region_totals={lead:defaultdict(lambda:defaultdict(int)) for lead in [5,10]}
    with torch.inference_mode():
        for offset in range(0,len(ticks),4):
            batch=ticks[offset:offset+4]
            x=torch.from_numpy(np.stack([np.stack([frames[t-timedelta(minutes=d)] for d in (10,5,0)]) for t in batch])).float()
            x[:,:,1:5]=((x[:,:,1:5]-mean)/std).clamp(-norm.get('zscore_clip',4),norm.get('zscore_clip',4))
            x[:,:,6]=(x[:,:,6]/norm['distance']['scale']).clamp(0,1)
            x=x.to(device)
            raw5,_,_,log5=model(x,return_logits=True)
            nxt=torch.cat([x[:,1:],torch.cat([raw5,x[:,-1,1:]],1).unsqueeze(1)],1)
            raw10,_,_,log10=model(nxt,return_logits=True)
            for lead,raw,logits in [(5,raw5,log5),(10,raw10,log10)]:
                raws=raw[:,0].clamp_min(0).cpu().numpy();probs=logits[:,0].sigmoid().cpu().numpy()
                for t,r,p in zip(batch,raws,probs):
                    prediction=r*clean_rain_mask(p)
                    future=frames.get(t+timedelta(minutes=lead))
                    if future is not None:
                        for band,counts in region_diagnostics(future[0],prediction).items():
                            for name,value in counts.items():region_totals[lead][band][name]+=value
                    actual=np.flipud(frames[t][0]);forecast=np.flipud(prediction)
                    for loc in locations:
                        observed,_,_=local_rain(actual,loc['latitude'],loc['longitude'])
                        predicted,_,_=local_rain(forecast,loc['latitude'],loc['longitude'])
                        records[lead,loc['name']].append(dict(time=t,actual=observed,forecast=predicted))
            if offset%100==0:print(f'Replayed {min(offset+4,len(ticks))}/{len(ticks)} five-minute ticks',flush=True)
    alerts=[];onsets=[];local_rows=[]
    for (lead,name),rows in records.items():
        a,o=replay_location(rows,lead)
        alerts += [dict(lead=lead,location=name,**r) for r in a]
        onsets += [dict(lead=lead,location=name,**r) for r in o]
        local_rows += [dict(lead=lead,location=name,**r) for r in rows]
    pd.DataFrame(alerts).to_csv(out/'alerts.csv',index=False)
    pd.DataFrame(onsets).to_csv(out/'onsets.csv',index=False)
    pd.DataFrame(local_rows).to_csv(out/'local_forecasts.csv',index=False)
    result=dict(ticks=len(ticks),skipped_ticks=len(skipped),locations=locations,
                mask_threshold=RAIN_PROBABILITY_THRESHOLD,min_size=MIN_COMPONENT_SIZE,
                checkpoint_sha256=file_hash(args.checkpoint),normalization_sha256=file_hash(args.normalization),
                decoder_version=LEGACY,preprocessing_version=RADAR_CLEANING_VERSION,
                start=start.isoformat(),end=end.isoformat(),region_metrics=region_totals,
                limitations=['Synthetic coverage grid unless locations supplied; not actual user distribution.',
                'Assumes immediate successful delivery, no scraping/network latency.',
                'Gaps reset episodes; future radar unavailable means unscored alert.',
                '+10 replay is hypothetical; operational bot remains +5.',
                'Targets retain tiny echoes; region stratification is diagnostic, not cleaned truth.'],leads={})
    for lead in [5,10]:
        a=[r for r in alerts if r['lead']==lead];o=[r for r in onsets if r['lead']==lead]
        clears=[r for r in a if r['premature_clear'] is not None]
        starts=[r for r in a if r['reason']==START and r['future_rain'] is not None]
        result['leads'][lead]=dict(alert_count=len(a),onsets=len(o),
            warned_before_onset=sum(r['warned_in_horizon'] for r in o),
            alerted_at_onset=sum(r['alerted_at_onset'] for r in o),
            predicted_clears_scored=len(clears),premature_clears=sum(r['premature_clear'] for r in clears),
            start_alerts_scored=len(starts),false_start_alerts=sum(not r['future_rain'] for r in starts),
            observed_clear_alerts=sum(r['reason']==ENDED for r in a),
            alerts_unscored=sum(r['future_rain'] is None for r in a))
    (out/'summary.json').write_text(json.dumps(result,indent=2,default=str))
    (out/'manifest.json').write_text(json.dumps(dict(ticks=[t.isoformat() for t in ticks],gaps=skipped),indent=2))
    print(json.dumps(result['leads'],indent=2),flush=True)


if __name__=='__main__':main()
