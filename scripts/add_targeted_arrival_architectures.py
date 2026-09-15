import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
names=['gru_t6_negative_baseline','gru_t6_hardnegative']
specs=[]
for architecture,pool in [('spatial','spatial'),('spatial_change','spatial_change')]:
    for fraction,suffix in [(0.,'baseline'),(.5,'hardnegative')]:
        name=f'gru_t6_depth2_{architecture}_{suffix}';names.append(name)
        specs.append(dict(name=name,kind='gru',history=6,channels=tuple(range(7)),output='categorical',
            hidden_dim=64,recurrent_layers=2,pooling=pool,hard_negative_fraction=fraction))
def update(s):
    if 'EXPERIMENTS = [' in s and 'gru_t6_depth2_spatial_change_baseline' not in s:
        s=s.replace('EXPERIMENTS = [','EXPERIMENTS = [\n'+'\n'.join('    '+repr(spec)+',' for spec in specs))
        s=s.replace('ENABLED = {"gru_t6_negative_baseline","gru_t6_hardnegative"}','ENABLED = '+repr(set(names)))
    s=s.replace('## Calibrate the winning two-layer spatial GRU','## Calibrate a selected sampling/architecture candidate')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=update(s)
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
n['cells'][0]['source']=['# Targeted arrival sampling and architecture comparisons\n\nSix runs are enabled: the original GRU, the two-layer spatial GRU, and a two-layer spatial head receiving both final features and final-minus-first recurrent features. Each has a matched run with and without hard-negative sampling. The feature difference is a learned temporal-change cue, not optical flow or a measured rain velocity. These are hypotheses to test, not assumed improvements.\n\nHard-negative sampling keeps the four arrival groups equally likely; within the no-arrival group, half the draws come from centres within five pixels of current qualifying rain that remain dry for the full future hour. General negatives remain. No labels are widened and validation sampling stays unchanged.\n\nRestart the kernel, set RUN_EXPERIMENTS=True and run in order. All runs share the configured update budget and seed. Existing completed matching runs are skipped; interrupted runs are preserved in retry folders. Select the calibration target after comparing results.\n']
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';s=update(p.read_text(encoding='utf-8'));p.write_text(s,encoding='utf-8')
