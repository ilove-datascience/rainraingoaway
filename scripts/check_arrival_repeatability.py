"""Bounded repeatability diagnostic; not model-quality training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arrival.models import ArrivalNet
from arrival.training import seed_everything

def run(strict):
    seed_everything(67,deterministic=strict)
    model=ArrivalNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(123)
    batches=[(torch.randn(4,6,7,64,64,generator=generator).cuda(),torch.randint(13,(4,),generator=generator).cuda()) for _ in range(4)]
    losses=[]
    for i in range(40):
        x,y=batches[i%4];opt.zero_grad();loss=torch.nn.functional.nll_loss(model.log_probs(x),y)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()]),losses

def main():
    torch.set_num_threads(2);results={}
    if not torch.cuda.is_available():raise RuntimeError('CUDA required to audit GPU repeatability')
    for strict in (False,True):
        a,la=run(strict);b,lb=run(strict)
        results[str(strict)]=dict(identical_weights=bool(torch.equal(a,b)),max_weight_difference=float((a-b).abs().max()),max_loss_difference=max(abs(x-y) for x,y in zip(la,lb)))
    results['scope']='40 updates on fixed synthetic inputs, same process/device. Tests numerical repeatability only; cannot reconstruct the historical kernel state or prove the original divergence cause.'
    out=ROOT/'reports/arrival_recovery_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'repeatability.json').write_text(json.dumps(results,indent=2));print(results)

if __name__=='__main__':main()
