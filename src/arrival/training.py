"""Training, calibration, metrics, and baselines for local arrival experiments."""
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy import ndimage
from arrival.models import ArrivalNet
from arrival.data import shifted,fingerprint
from arrival.ranking import ranking_report


def seed_everything(seed,deterministic=False):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=deterministic
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def loader(dataset,batch=16,shuffle=False):
    # Dynamic sampling epoch is updated in the main process; no stale worker copy.
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=0)


def predict(model,dataset,batch=32,device=None):
    device=device or next(model.parameters()).device;model.eval();outputs=[];targets=[]
    with torch.inference_mode():
        for x,y in loader(dataset,batch):outputs.append(model.log_probs(x.to(device)).cpu());targets.append(y)
    if not outputs:raise ValueError('Empty evaluation dataset')
    return torch.cat(outputs),torch.cat(targets)


def fit(model,train,validation,out,budget=3000,eval_every=250,patience=8,batch=16,lr=1e-3,seed=67,selection='nll',deterministic=False):
    if selection not in ('nll', 'mean_ap'): raise ValueError('Unknown checkpoint selection')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError(f'{out}/best.pt exists; choose a new run directory')
    if len(train)==0 or len(validation)==0:raise ValueError('Empty dataset')
    device=next(model.parameters()).device
    seed_everything(seed,deterministic=deterministic)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    step=0;epoch=0;best=float('inf');best_nll=float('inf');stale=0;history=[];start=time.monotonic();loss_sum=0.;samples=0
    while step<budget and stale<patience:
        train.set_epoch(epoch)
        train.set_step(step)
        for x,y in loader(train,batch,True):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=F.nll_loss(model.log_probs(x.to(device)),y.to(device))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            step+=1;loss_sum+=loss.item()*len(y);samples+=len(y)
            train.set_step(step)
            if step%eval_every==0 or step==budget:
                lp,truth=predict(model,validation,batch,device)
                validation_loss=F.nll_loss(lp,truth).item()
                ranking=ranking_report(lp.exp().numpy(),truth.numpy())
                aps={h:r['average_precision'] for h,r in ranking.items()}
                valid_ap=[v for v in aps.values() if v is not None]
                mean_ap=float(np.mean(valid_ap)) if valid_ap else None
                if selection=='mean_ap' and len(valid_ap)!=3:
                    raise ValueError('AP selection requires arrivals at all three validation horizons')
                score=validation_loss if selection=='nll' else -mean_ap
                row=dict(step=step,epoch=epoch,train_loss=loss_sum/max(samples,1),validation_nll=validation_loss,
                         effective_hard_fraction=train.effective_hard_fraction(),
                         validation_ap=aps,validation_mean_ap=mean_ap,
                         elapsed_seconds=time.monotonic()-start)
                history.append(row);print(row,flush=True);loss_sum=0.;samples=0
                checkpoint=dict(state_dict=model.state_dict(),model_config=model.config,norm=train.norm,
                        reproducibility=dict(deterministic=deterministic,torch_version=str(torch.__version__),cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version()),
                        hard_negative_fraction=train.hard_negative_fraction,hard_negative_warmup=train.hard_negative_warmup,
                        hard_negative_ramp=train.hard_negative_ramp,sampler_revision='hard_branch_independent_rng_v2',
                        sample_counts=dict(groups=list(train.sample_counts['groups']),hard_branch=train.sample_counts['hard_branch']),
                        data_config=vars(train.labels.config),channels=train.channels,seed=seed,
                        train_anchor_hash=fingerprint(train.anchors),validation_records_hash=fingerprint(validation.records),
                        decoder=train.norm['decoder'],sampler='global_groups_v2',step=step,
                        selection=selection,validation_ap=aps,validation_nll=validation_loss)
                if validation_loss<best_nll:
                    best_nll=validation_loss
                    torch.save(checkpoint,out/'best_nll.pt')
                if score<best:
                    best=score;stale=0
                    torch.save(checkpoint,out/'best.pt')
                else:stale+=1
                (out/'history.json').write_text(json.dumps(history,indent=2))
            if step>=budget or stale>=patience:break
        epoch+=1
    state=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['state_dict'])
    return history


def load_model(path,device):
    state=torch.load(path,map_location=device,weights_only=False)
    model=ArrivalNet(**state['model_config']).to(device);model.load_state_dict(state['state_dict']);model.eval()
    return model,state


def fit_legacy_calibrator(log_probs,targets):
    """Natural-frequency calibration split; temperature plus regularized class bias.

    Bias addresses the training sampler's changed class priors. It does not make
    independent test evaluation optional. Only use trusted local tensors here.
    """
    scores=log_probs.detach().double();targets=targets.long()
    temperature=torch.zeros((),dtype=torch.float64,requires_grad=True)
    bias=torch.zeros(13,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([temperature,bias],max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        centered=bias-bias.mean()
        loss=F.cross_entropy(scores/temperature.exp().clamp(.05,20)+centered,targets)+.01*centered.square().mean()
        loss.backward();return loss
    optimizer.step(closure)
    return dict(temperature=float(temperature.exp().clamp(.05,20).detach()),bias=(bias-bias.mean()).detach().tolist())


def fit_calibrator(log_probs,targets):
    """Shared increasing logistic map of cumulative arrival probabilities.

    Shared positive slope/intercept preserve ranking at every horizon AND
    cumulative ordering across horizons. Fit only on the calibration split.
    """
    p=log_probs.detach().double().softmax(1)
    cumulative=p.cumsum(1)[:,:12].clamp(1e-10,1-1e-10)
    scores=torch.logit(cumulative)
    truth=(targets[:,None].to(scores.device)<torch.arange(1,13,device=scores.device)).double()
    log_slope=torch.zeros((),dtype=torch.float64,device=scores.device,requires_grad=True)
    intercept=torch.zeros((),dtype=torch.float64,device=scores.device,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_slope,intercept],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=F.binary_cross_entropy_with_logits(scores*log_slope.exp().clamp(.05,20)+intercept,truth)
        loss=loss+1e-4*(log_slope.square()+intercept.square())
        loss.backward();return loss
    optimizer.step(closure)
    return dict(method='shared_cumulative_logistic_v1',slope=float(log_slope.exp().clamp(.05,20).detach()),
                intercept=float(intercept.detach()))


def calibrated_probs(log_probs,calibrator=None):
    if calibrator is None:return log_probs.softmax(1).numpy()
    if calibrator.get('method')=='independent_12_cumulative_logistic_then_isotonic_v1':
        from arrival.calibration import apply_projected_calibrator
        return apply_projected_calibrator(log_probs,calibrator)
    if calibrator.get('method') in ('shared_cumulative_logistic_v1','ordered_cumulative_logistic_v1'):
        p=log_probs.double().softmax(1)
        c=p.cumsum(1)[:,:12].clamp(1e-10,1-1e-10)
        c=torch.sigmoid(torch.logit(c)*calibrator['slope']+torch.as_tensor(calibrator['intercept'],dtype=c.dtype,device=c.device))
        return torch.diff(torch.cat([torch.zeros_like(c[:,:1]),c,torch.ones_like(c[:,:1])],1),dim=1).numpy()
    adjusted=log_probs/calibrator['temperature']+torch.tensor(calibrator['bias'],dtype=log_probs.dtype)
    return adjusted.softmax(1).numpy()


def metrics(probabilities,labels):
    p=np.asarray(probabilities);y=np.asarray(labels);pred=p.argmax(1)
    if p.shape!=(len(y),13) or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
    confusion=np.zeros((13,13),dtype=int);np.add.at(confusion,(y,pred),1)
    tp=np.diag(confusion);support=confusion.sum(1);predicted=confusion.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(13,dtype=float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(13,dtype=float),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(13),where=precision+recall>0)
    rainy=y<12;both=rainy&(pred<12)
    horizon={}
    for minutes,k in [(15,3),(30,6),(60,12)]:
        probability=p[:,:k].sum(1);truth=y<k;decision=probability>=.5
        hit=int((decision&truth).sum());fp=int((decision&~truth).sum());fn=int((~decision&truth).sum())
        horizon[minutes]=dict(brier=float(np.mean((probability-truth)**2)),precision=hit/max(hit+fp,1),
                              recall=hit/max(hit+fn,1),f1=2*hit/max(2*hit+fp+fn,1))
    return dict(ranking=ranking_report(p,y),accuracy=float((pred==y).mean()),macro_f1_all_13=float(f1.mean()),
        nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
        per_class_precision=precision.tolist(),per_class_recall=recall.tolist(),support=support.tolist(),
        confusion=confusion.tolist(),horizons=horizon,
        arrival_mae_minutes_detected=float(np.mean(np.abs(pred[both]-y[both])*5)) if both.any() else None,
        rainy_cases=int(rainy.sum()),missed_arrivals=int((rainy&(pred==12)).sum()),
        false_arrivals=int((~rainy&(pred<12)).sum()),
        arrival_groups={name:dict(count=int(((y>=lo)&(y<=hi)).sum()),
            accuracy=float((pred[(y>=lo)&(y<=hi)]==y[(y>=lo)&(y<=hi)]).mean()) if ((y>=lo)&(y<=hi)).any() else None)
            for name,lo,hi in [('0–15',0,2),('15–30',3,5),('30–60',6,11),('none',12,12)]})


def persistence(records):
    p=np.zeros((len(records),13));p[:,12]=1
    return p


def motion_baseline(store,records,config,max_shift=8):
    """Global translation baseline on FULL radar domain, not optical flow.

    Match the previous frame to the current one over fixed interior pixels.
    No fitted future information; growth/decay and nonuniform flow are not modeled.
    Zero-filled edges are explicit and limit long-horizon interpretation.
    """
    shifts={};probabilities=np.zeros((len(records),13))
    for i,(key,y,x) in enumerate(records):
        if key not in shifts:
            previous=store.radar(shifted(key,-5));current=store.radar(key)
            best=(float('inf'),0,0)
            for dy in range(-max_shift,max_shift+1):
                for dx in range(-max_shift,max_shift+1):
                    moved=ndimage.shift(previous,(dy,dx),order=0,mode='constant',cval=0,prefilter=False)
                    error=np.mean((moved[max_shift:-max_shift,max_shift:-max_shift]-current[max_shift:-max_shift,max_shift:-max_shift])**2)
                    candidate=(float(error),abs(dy)+abs(dx),dy,dx)
                    if len(best)==3 or candidate<best:best=candidate
            shifts[key]=best[2:]
        dy,dx=shifts[key];current=store.radar(key);arrival=12;half=config.patch//2
        for k in range(1,13):
            # Pull back this target neighborhood instead of warping an entire frame.
            ys=np.arange(y-half,y+half+1)-dy*k;xs=np.arange(x-half,x+half+1)-dx*k
            patch=np.zeros((config.patch,config.patch))
            good_y=(ys>=0)&(ys<current.shape[0]);good_x=(xs>=0)&(xs<current.shape[1])
            patch[np.ix_(good_y,good_x)]=current[np.ix_(ys[good_y],xs[good_x])]
            if (patch>config.rain_threshold).sum()>=np.ceil(config.coverage*config.patch**2):arrival=k-1;break
        probabilities[i,arrival]=1
    return probabilities,shifts
