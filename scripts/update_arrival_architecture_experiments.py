"""Add controlled architecture comparisons without replacing saved notebook outputs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def update(s):
    if 'EXPERIMENTS = [' in s and 'gru_t6_wide96' not in s:
        s=s.replace('EXPERIMENTS = [','''EXPERIMENTS = [
    dict(name="gru_t6_wide96",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hidden_dim=96,pooling="global"),
    dict(name="gru_t6_spatial",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hidden_dim=64,pooling="spatial"),''')
    s=s.replace('ENABLED = {"cnn_t6_all","gru_t6_all"}','ENABLED = {"gru_t6_all","gru_t6_wide96","gru_t6_spatial"}')
    s=s.replace('UPDATE_BUDGET = 3000\n','UPDATE_BUDGET = 30000\n')
    s=s.replace('experiments_v2','experiments_architecture_v1')
    s=s.replace('kind=experiment["kind"],output=experiment["output"]).to(DEVICE)',
        'kind=experiment["kind"],output=experiment["output"],\n                hidden_dim=experiment.get("hidden_dim",64),pooling=experiment.get("pooling","global")).to(DEVICE)')
    s=s.replace('eval_every=250,patience=8,batch=16,seed=seed)',
        'eval_every=250,patience=40,batch=16,seed=seed,selection="mean_ap")')
    s=s.replace('parameter_count = sum(p.numel() for p in candidate.parameters())',
        'parameter_count = sum(p.numel() for p in candidate.parameters())\n            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()')
    s=s.replace('payload = dict(experiment=experiment,seed=seed,parameters=parameter_count,',
        'payload = dict(experiment=experiment,seed=seed,parameters=parameter_count,\n                peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,')
    s=s.replace('Use validation NLL, macro F1, missed/false arrivals and 15/30/60-minute Brier scores.',
        'Compare mean validation average precision and 15/30/60-minute AP, alongside NLL, Brier scores, parameter count, elapsed time and peak GPU memory. The enabled comparisons change one factor: baseline 64-channel GRU with global pooling, 96-channel GRU with global pooling, or 64-channel GRU with a spatial 4x4 pooled head. All use six frames, seven channels, seed 67 and up to 30,000 updates. Training runs sequentially. These are fresh runs, not resumptions of v5. Calibration and notification replay are separate follow-up evaluations after selecting a candidate.')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    source=''.join(c['source']);u=update(source)
    if source!=u:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';s=p.read_text(encoding='utf-8');p.write_text(update(s),encoding='utf-8')
