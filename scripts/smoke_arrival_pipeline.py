"""Small real-data integration check, not a forecasting-skill experiment."""
import sys,json,tempfile
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.data import DataConfig,FrameStore,LabelStore,prepare,ArrivalDataset,fixed_locations
from arrival.models import ArrivalNet
from arrival.training import fit,predict,load_model,fit_calibrator,calibrated_probs,metrics,motion_baseline,seed_everything
from arrival.plots import plot_history,plot_evaluation,plot_example

root=Path(__file__).resolve().parents[1]
out=root/'reports/arrival_smoke';out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);seed_everything(67)
store=FrameStore(root)
config=DataConfig(crop=64,max_history=12,stride=48,development_end='2026-06-05')
prepared=prepare(store,config,out/'data',normalization_frames=8)
labels=LabelStore(store,config);parts=prepared['manifests'];norm=prepared['norm']
train=ArrivalDataset(store,labels,parts['train'][:8],norm,per_anchor=2)
records=fixed_locations(labels,parts['validation'][:4],2)
validation=ArrivalDataset(store,labels,parts['validation'],norm,records=records)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ArrivalNet().to(device)
with tempfile.TemporaryDirectory() as directory:
    history=fit(model,train,validation,directory,budget=2,eval_every=1,patience=2,batch=2)
    model,state=load_model(Path(directory)/'best.pt',device)
    assert state['norm']==json.loads((out/'data/prepared.json').read_text())['norm']
    logp,truth=predict(model,validation,batch=2)
    calibrator=fit_calibrator(logp,truth)
    p=calibrated_probs(logp,calibrator)
    assert np.allclose(p.sum(1),1)
    result=metrics(p,truth)
    motion,shifts=motion_baseline(store,records[:2],config,max_shift=2)
    assert np.allclose(motion.sum(1),1)
    figure_names=iter(['training_smoke','evaluation_smoke','crop_smoke'])
    def save_figure():
        plt.gcf().savefig(out/(next(figure_names)+'.png'),dpi=100,bbox_inches='tight')
        plt.close('all')
    plt.show=save_figure
    plot_history(history);plot_evaluation(p,truth,result);plot_example(validation,0,p[0])
    summary=dict(purpose='Integration smoke only; no model quality claims',device=str(device),
        anchors={k:len(v) for k,v in parts.items()},validation_examples=len(records),
        train_updates=2,output_shape=list(p.shape),checkpoint_reload=True,calibration=True,
        motion_baseline=True,metrics_finite=bool(np.isfinite(result['nll'])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
