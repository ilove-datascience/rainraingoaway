"""Calibrate three saved runs and replay them using one shared input pass."""
import sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arrival.data import FrameStore,LabelStore,DataConfig,ArrivalDataset,input_crop,fingerprint
from arrival.training import load_model,predict,calibrated_probs,metrics
from arrival.calibration import fit_projected_calibrator

def main():
    torch.set_num_threads(2)
    data=json.loads((ROOT/'models/arrival/data_v1/prepared.json').read_text())
    records=json.loads((ROOT/'models/arrival/data_v1/evaluation_locations.json').read_text())
    source=ROOT/'reports/arrival_notification_replay'
    ref=json.loads((source/'results.json').read_text());keys=ref['settings']['keys'];locations=ref['settings']['locations']
    with np.load(source/'forecasts.npz') as saved:wet=saved['wet']
    assert keys==data['manifests']['validation']
    store=FrameStore(ROOT);labels=LabelStore(store,DataConfig(**data['config']))
    dataset=ArrivalDataset(store,labels,data['manifests']['calibration'],data['norm'],records=records['calibration'])
    root=ROOT/'reports/arrival_recovery_audit';root.mkdir(parents=True,exist_ok=True)
    models=[]
    for name in ('gru_recovery_baseline_seed67','gru_recovery_hard10_seed67','gru_recovery_hard10_curriculum_seed67'):
        run=ROOT/'models/arrival/experiments_hardnegative_recovery_v1'/name
        model,state=load_model(run/'best.pt','cuda' if torch.cuda.is_available() else 'cpu')
        assert state['validation_records_hash']==fingerprint(records['validation']) and state['norm']==data['norm']
        out=root/name;out.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        print('Calibrating',name,flush=True)
        lp,y=predict(model,dataset)
        calibration=fit_projected_calibrator(lp,y);calibration.update(checkpoint_sha256=digest,calibration_records_hash=fingerprint(records['calibration']))
        (out/'calibrator.json').write_text(json.dumps(calibration,indent=2))
        with np.load(run/'validation_predictions.npz') as saved:
            vlp=torch.from_numpy(np.log(np.maximum(saved['probabilities'],1e-30)))
            result=metrics(calibrated_probs(vlp,calibration),saved['labels'])
        (out/'calibrated_validation.json').write_text(json.dumps(result,indent=2))
        models.append(dict(name=name,run=run,model=model,out=out,calibration=calibration,
            probs=np.zeros((len(keys),len(locations),3)),digest=digest))
    batch=[];positions=[]
    def flush():
        if not batch:return
        inputs=torch.stack(batch).to(next(models[0]['model'].parameters()).device)
        with torch.inference_mode():
            for entry in models:
                lp=entry['model'].log_probs(inputs).cpu()
                p=calibrated_probs(lp,entry['calibration']).cumsum(1)[:,[2,5,11]]
                for pos,values in zip(positions,p):entry['probs'][pos]=values
        batch.clear();positions.clear()
    for i,key in enumerate(keys):
        for j in np.flatnonzero(~wet[i]):
            batch.append(input_crop(store,key,*locations[j],data['norm']));positions.append((i,j))
            if len(batch)>=32:flush()
        if i%200==0:print('Replay input pass',i,'/',len(keys),flush=True)
    flush()
    for entry in models:
        out=entry['out'];np.savez_compressed(out/'forecasts.npz',wet=wet,probs=entry['probs'])
        meta=dict(checkpoint=entry['digest'],keys=keys,locations=locations,parameters=entry['calibration']['parameters'])
        (out/'cache_tag.txt').write_text(hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest())
        subprocess.run([sys.executable,str(ROOT/'scripts/replay_arrival_notifications.py'),
            '--run',str(entry['run']),'--calibration',str(out/'calibrator.json'),'--output',str(out)],check=True,cwd=ROOT)

if __name__=='__main__':main()
