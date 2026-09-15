import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def update(s):
    if 'EXPERIMENTS = [' in s and 'gru_t6_hardnegative' not in s:
        s=s.replace('EXPERIMENTS = [','''EXPERIMENTS = [
    dict(name="gru_t6_negative_baseline",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.0),
    dict(name="gru_t6_hardnegative",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.5),''')
    if 'EXPERIMENTS = [' in s:
        s=re.sub(r'ENABLED = [^\n]+','ENABLED = {"gru_t6_negative_baseline","gru_t6_hardnegative"}',s)
        s=s.replace('experiments_architecture_v1','experiments_hardnegative_v1')
    s=s.replace('            training.seed = seed','            training.seed = seed\n            training.hard_negative_fraction = experiment.get("hard_negative_fraction",0.0)')
    s=s.replace('payload = dict(experiment=experiment,seed=seed,parameters=parameter_count,',
        'payload = dict(experiment=experiment,seed=seed,parameters=parameter_count,\n                sample_counts=training.sample_counts,')
    s=s.replace('PAIR = ("gru_t6_all_seed67", "gru_t6_wide96_seed67")',
        'PAIR = ("gru_t6_negative_baseline_seed67", "gru_t6_hardnegative_seed67")')
    s=s.replace('SELECTED_RUN = "gru_t6_depth2_spatial_seed67"','SELECTED_RUN = "gru_t6_hardnegative_seed67"  # Compare results before treating this as a winner.')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=update(s)
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
n['cells'][0]['source']=['# Rain-arrival experiments: hard-negative sampling\n\nOnly two experiments are currently enabled: the unchanged 64-channel GRU baseline and the same model with 50% of no-arrival draws selected within five pixels of current qualifying rain. Both keep the four arrival groups equally likely and retain the existing complete-future no-arrival labels. Validation remains naturally sampled. Existing architecture results are preserved in their previous folder.\n\nRestart the kernel, set RUN_EXPERIMENTS=True, and run in order. The first hard-negative index pass scans training radar and can take time. Compare notification burden after refitting calibration; do not assume the candidate is better. Counts of actual training draws are saved in result.json.\n']
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';s=p.read_text(encoding='utf-8');p.write_text(update(s),encoding='utf-8')
