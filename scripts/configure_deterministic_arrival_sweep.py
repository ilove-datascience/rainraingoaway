"""Configure only the paired multi-seed repeatability experiment."""
import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
config='''EXPERIMENTS = [
    dict(name="gru_deterministic_baseline",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.0),
    dict(name="gru_deterministic_hard10",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.1),
]
display(pd.DataFrame(EXPERIMENTS))
RUN_EXPERIMENTS = True
ENABLED = {e["name"] for e in EXPERIMENTS}
SEEDS = [67, 71, 79]
UPDATE_BUDGET = 40000
SWEEP_DIR = PROJECT_ROOT / "models/arrival/experiments_deterministic_v1"
'''
summary='''# Compare paired seeds; do not choose a method from its single best seed.
if len(comparison):
    score_columns = ["mean_AP","AP_15","AP_30","AP_60"]
    display(comparison.groupby("name")[score_columns].agg(["count","mean","std"]))
    paired = comparison.pivot(index="seed",columns="name",values="mean_AP")
    if {"gru_deterministic_baseline","gru_deterministic_hard10"}.issubset(paired.columns):
        paired["hard10_minus_baseline"] = paired["gru_deterministic_hard10"]-paired["gru_deterministic_baseline"]
        display(paired)
        paired.to_csv(SWEEP_DIR/"paired_seed_comparison.csv")
        print("Complete pairs:",paired.dropna().shape[0],"of",len(SEEDS))
    print("Calibration/replay still needed before notification conclusions. Three seeds are not a final independent test.")
'''
def transform(s):
    if s.startswith('import sys, json'):
        s='import os\nos.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n'+s
        s=s.replace('seed_everything(SEED)','seed_everything(SEED,deterministic=True)')
    s=s.replace('            seed_everything(seed)','            seed_everything(seed,deterministic=True)')
    s=s.replace('seed=seed,selection="mean_ap")','seed=seed,selection="mean_ap",deterministic=True)')
    s=s.replace('sample_counts=training.sample_counts,','sample_counts=training.sample_counts,deterministic=True,update_budget=UPDATE_BUDGET,')
    s=s.replace('completed["manifest_hash"] != prepared["manifest_hash"] or',
        'not completed.get("deterministic",False) or completed.get("update_budget") != UPDATE_BUDGET or\n                    completed["manifest_hash"] != prepared["manifest_hash"] or')
    s=s.replace('gru_recovery_baseline_seed67','gru_deterministic_baseline_seed67').replace('gru_recovery_hard10_curriculum_seed67','gru_deterministic_hard10_seed67')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=config if s.startswith('EXPERIMENTS = [') else transform(s)
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
n['cells'][0]['source']=['# Deterministic baseline versus 10% hard-negative sampling\n\nRestart the kernel, then Run All. Only two configurations are enabled, each with seeds 67, 71 and 79 (six fresh runs total). Training is enabled. Both use six frames, seven channels, the original 64-channel GRU, batch 16, and up to 40,000 updates with the same early stopping. Prior experiments remain in their old directories. Completed matching new runs are skipped; interrupted runs are preserved in retry folders.\n\nStrict deterministic execution is enabled before model initialization and inside training. Unsupported deterministic operations will raise rather than silently fall back. This improves same-device repeatability, not guarantees across hardware or framework versions. Calibration and replay remain disabled until the paired results have been reviewed.\n']
idx=next(i for i,c in enumerate(n['cells']) if 'comparison = pd.DataFrame(rows)' in ''.join(c['source']))
n['cells'].insert(idx+1,dict(cell_type='code',id='paired-seed-summary',metadata={},source=summary.splitlines(keepends=True),outputs=[],execution_count=None))
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
# Synchronize generator by replacing only experiment cells after construction.
p=root/'scripts/build_arrival_notebooks.py';s=p.read_text(encoding='utf-8')
block='\n# Deterministic paired sweep configuration.\n'
block+='for c in experiments:\n    source="".join(c["source"])\n'
block+='    if source.startswith("EXPERIMENTS = ["):\n        c["source"]='+repr(config.splitlines(keepends=True))+'\n'
for i,c in enumerate(n['cells']):
    if c['cell_type']=='code' and any(x in ''.join(c['source']) for x in ['CUBLAS_WORKSPACE_CONFIG','if RUN_EXPERIMENTS:','SELECTED_RUN =','PAIR =']):
        marker= 'import sys, json' if 'CUBLAS_WORKSPACE_CONFIG' in ''.join(c['source']) else ('if RUN_EXPERIMENTS:' if 'if RUN_EXPERIMENTS:' in ''.join(c['source']) else ('SELECTED_RUN =' if 'SELECTED_RUN =' in ''.join(c['source']) else 'PAIR ='))
        block+='    if '+repr(marker)+' in source:\n        c["source"]='+repr(c['source'])+'\n'
block+='experiments.insert(next(i for i,c in enumerate(experiments) if "comparison = pd.DataFrame(rows)" in "".join(c["source"]))+1,code('+repr(summary)+'))\n'
block+='experiments[0]["source"]='+repr(n['cells'][0]['source'])+'\n'
s=s.replace("if __name__=='__main__':",block+"\nif __name__=='__main__':")
p.write_text(s,encoding='utf-8')
