"""Candidate ordered cumulative calibration; fitted on calibration data only."""
import torch
import numpy as np
from torch.nn import functional as F


def fit_projected_calibrator(log_probs,targets):
    """Fit twelve positive-slope logistic maps; projection enforces ordering."""
    p=log_probs.detach().cpu().double().softmax(1).cumsum(1)[:,:12].clamp(1e-10,1-1e-10)
    x=torch.logit(p)
    y=(targets.detach().cpu()[:,None]<torch.arange(1,13)).double()
    parameters=[]
    for k in range(12):
        a=torch.zeros((),dtype=torch.float64,requires_grad=True)
        b=torch.zeros((),dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.LBFGS([a,b],max_iter=100,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss=F.binary_cross_entropy_with_logits(x[:,k]*a.exp().clamp(.05,20)+b,y[:,k])+1e-4*(a*a+b*b)
            loss.backward();return loss
        optimizer.step(closure)
        parameters.append([float(a.exp().clamp(.05,20).detach()),float(b.detach())])
    return dict(method='independent_12_cumulative_logistic_then_isotonic_v1',parameters=parameters)


def apply_projected_calibrator(log_probs,calibrator):
    p=log_probs.detach().cpu().double().softmax(1).cumsum(1)[:,:12].clamp(1e-10,1-1e-10)
    parameters=torch.tensor(calibrator['parameters'],dtype=torch.float64)
    c=torch.sigmoid(torch.logit(p)*parameters[:,0]+parameters[:,1]).numpy()
    c=project_cumulative(c)
    return np.diff(np.column_stack([np.zeros(len(c)),c,np.ones(len(c))]),axis=1)


def project_cumulative(values):
    """Least-squares isotonic projection across horizons (equal weights)."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Expected finite [samples,horizons]')
    result=np.empty_like(values)
    for i,row in enumerate(values):
        blocks=[]
        for value in row:
            blocks.append([float(value),1])
            while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
                b=blocks.pop();a=blocks.pop();n=a[1]+b[1]
                blocks.append([(a[0]*a[1]+b[0]*b[1])/n,n])
        result[i]=np.repeat([b[0] for b in blocks],[b[1] for b in blocks])
    return np.clip(result,0,1)


def fit_ordered_calibrator(log_probs,targets):
    scores=torch.logit(log_probs.detach().double().softmax(1).cumsum(1)[:,:12].clamp(1e-10,1-1e-10))
    truth=(targets[:,None].to(scores.device)<torch.arange(1,13,device=scores.device)).double()
    slope=torch.zeros((),dtype=torch.float64,requires_grad=True)
    base=torch.zeros((),dtype=torch.float64,requires_grad=True)
    increments=torch.full((11,),-4.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([slope,base,increments],max_iter=150,line_search_fn='strong_wolfe')
    def offsets():return torch.cat([base[None],base+F.softplus(increments).cumsum(0)])
    def closure():
        optimizer.zero_grad();bias=offsets()
        loss=F.binary_cross_entropy_with_logits(scores*slope.exp().clamp(.05,20)+bias,truth)
        loss+=1e-4*(slope.square()+bias.square().mean())
        loss.backward();return loss
    optimizer.step(closure)
    return dict(method='ordered_cumulative_logistic_v1',slope=float(slope.exp().clamp(.05,20).detach()),
                intercept=offsets().detach().tolist())
