"""Offline independent cumulative calibration followed by isotonic projection."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_arrival_calibration import binary_fit,apply,summarize
from arrival.calibration import project_cumulative
from arrival.data import fingerprint

def main():
    torch.set_num_threads(2)
    run=ROOT/'models/arrival/runs/gru_t6_all_seed67_v5'
    out=ROOT/'reports/arrival_v5_calibration'
    records=json.loads((ROOT/'models/arrival/data_v1/evaluation_locations.json').read_text())
    digest=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    sets={}
    for split in ('calibration','validation'):
        path=out/f'{split}_{fingerprint([digest,records[split]])[:16]}.npz'
        if not path.exists():raise RuntimeError('Run scripts/audit_arrival_calibration.py first to build matching caches')
        data=np.load(path)
        p=torch.from_numpy(data['lp']).double().softmax(1).numpy().cumsum(1)[:,:12]
        sets[split]=(p,data['y'],np.array([r[0][:8] for r in records[split]]))
    cp,cy,cd=sets['calibration'];vp,vy,vd=sets['validation']
    def fit(p,y):return [binary_fit(p[:,i],y<i+1) for i in range(12)]
    def transform(p,params):return np.column_stack([apply(p[:,i],params[i]) for i in range(12)])
    params=fit(cp,cy);unprojected=transform(vp,params);projected=project_cumulative(unprojected)
    oof=np.zeros_like(cp);oof_raw=np.zeros_like(cp);constant=np.zeros_like(cp)
    days=sorted(set(cd))
    for fold in range(3):
        test=np.isin(cd,days[fold::3]);train=~test
        oof_raw[test]=transform(cp[test],fit(cp[train],cy[train]))
        oof[test]=project_cumulative(oof_raw[test])
        constant[test]=np.array([(cy[train]<k).mean() for k in range(1,13)])
    audit=json.loads((out/'audit.json').read_text())
    result=dict(checkpoint_sha256=digest,method='independent_12_cumulative_logistic_then_isotonic_v1',
        caveat='Development experiment only. Projection guarantees cumulative ordering but can change ranking. Validation already selected checkpoint. Day folds do not guarantee independent storms.',parameters=params,
        validation_projected=summarize(projected[:,[2,5,11]],vy,vd),
        validation_unprojected=summarize(unprojected[:,[2,5,11]],vy,vd),
        calibration_oof_projected=summarize(oof[:,[2,5,11]],cy,cd),
        calibration_oof_unprojected=summarize(oof_raw[:,[2,5,11]],cy,cd),
        calibration_oof_constant=summarize(constant[:,[2,5,11]],cy,cd),
        calibration_oof_shared=audit['calibration_oof_shared'],validation_shared=audit['validation_shared'])
    assert np.all(np.diff(projected,axis=1)>=-1e-12)
    distribution=np.diff(np.column_stack([np.zeros(len(projected)),projected,np.ones(len(projected))]),axis=1)
    assert np.all(distribution>=-1e-12) and np.allclose(distribution.sum(1),1)
    (out/'projection_experiment.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'projection_validation_predictions.npz',probabilities=distribution,truth=vy)
    lines=['# Cumulative calibration projection experiment','',result['caveat'],'',
        '| Evaluation | Horizon | Brier | AP |','|---|---:|---:|---:|']
    for name,r in result.items():
        if not isinstance(r,dict) or 'crossing_fraction' not in r:continue
        for h in (15,30,60):
            m=r.get(h,r.get(str(h)))
            line=f"| {name} | {h} | {m['brier']:.5f} | {m['average_precision']:.4f} |"
            lines.append(line);print(line)
    print('Projected distributions valid; zero horizon crossings. Candidate not deployed.')
    (out/'projection_report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
