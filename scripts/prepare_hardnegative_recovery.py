import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def update(s):
    if 'EXPERIMENTS = [' in s and 'gru_recovery_baseline' not in s:
        s=s.replace('EXPERIMENTS = [','''EXPERIMENTS = [
    dict(name="gru_recovery_baseline",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.0),
    dict(name="gru_recovery_hard10",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.1),
    dict(name="gru_recovery_hard10_curriculum",kind="gru",history=6,channels=tuple(range(7)),output="categorical",hard_negative_fraction=0.1,hard_negative_warmup=5000,hard_negative_ramp=5000),''')
        s=re.sub(r'ENABLED = [^\n]+','ENABLED = {"gru_recovery_baseline","gru_recovery_hard10","gru_recovery_hard10_curriculum"}',s)
        s=s.replace('experiments_hardnegative_v1','experiments_hardnegative_recovery_v1')
    line='            training.hard_negative_fraction = experiment.get("hard_negative_fraction",0.0)'
    if line in s and 'training.hard_negative_warmup =' not in s:
        s=s.replace(line,line+'\n            training.hard_negative_warmup = experiment.get("hard_negative_warmup",0)\n            training.hard_negative_ramp = experiment.get("hard_negative_ramp",0)')
    s=s.replace('PAIR = ("gru_t6_negative_baseline_seed67", "gru_t6_hardnegative_seed67")',
        'PAIR = ("gru_recovery_baseline_seed67", "gru_recovery_hard10_curriculum_seed67")')
    s=s.replace('SELECTED_RUN = "gru_t6_hardnegative_seed67"', 'SELECTED_RUN = "gru_recovery_baseline_seed67"')
    if 'SELECTED_RUN =' in s:
        s=s.replace('selected = None','RUN_CALIBRATION = False  # Enable only after choosing a successful candidate.\nselected = None')
        s=s.replace('if selected_path is not None and (selected_path / "best.pt").exists():',
            'if RUN_CALIBRATION and selected_path is not None and (selected_path / "best.pt").exists():')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=update(s)
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
n['cells'][0]['source']=['# Hard-negative recovery experiments\n\nThe 50% intervention failed across architectures; it is disabled. Three controlled runs are enabled on the original GRU: unchanged sampling, 10% hard negatives within the no-arrival group from the start, and a 10% curriculum (zero for 5,000 updates, ramping to 10% by update 10,000). The four arrival groups remain balanced, so the final hard branch is about 2.5% of all draws. All use the same seed, architecture, optimizer and configured update budget. These are hypotheses, not validated improvements.\n\nBaseline sample selection is identical during curriculum warmup. Hard-negative branch randomness is isolated from general location sampling. Training history records the effective fraction, and checkpoints record schedule settings and sample counts. Validation is unchanged. Existing runs are preserved. Calibration is disabled until results are compared. Restart the kernel before running.\n']
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';p.write_text(update(p.read_text(encoding='utf-8')),encoding='utf-8')
