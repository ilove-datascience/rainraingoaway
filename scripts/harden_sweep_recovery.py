import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def update(s):
    s=s.replace('from arrival.run_paths import sweep_run_path','from arrival.run_paths import sweep_run_path, write_completed_result')
    s=s.replace('            (path / "result.json").write_text(json.dumps(payload,indent=2))\n','')
    marker='            np.savez_compressed(path / "validation_predictions.npz",probabilities=calibrated_probs(log_probs),labels=truth.numpy())'
    if marker in s and 'write_completed_result(path, payload)' not in s:
        s=s.replace(marker,marker+'\n            write_completed_result(path, payload)')
    s=s.replace('results = [json.loads(p.read_text()) for p in sorted(SWEEP_DIR.glob("*/result.json"))]',
        'from arrival.run_paths import completed_run\nresults = [json.loads(p.read_text()) for p in sorted(SWEEP_DIR.glob("*/result.json")) if completed_run(p.parent)]')
    s=s.replace('paths = [SWEEP_DIR / name / "validation_predictions.npz" for name in PAIR]\nif all(p.exists() for p in paths):',
        'from arrival.run_paths import find_completed_run\npaired_runs = [find_completed_run(SWEEP_DIR / name) for name in PAIR]\npaths = [p / "validation_predictions.npz" for p in paired_runs if p is not None]\nif len(paths)==2 and all(p.exists() for p in paths):')
    s=s.replace('selected_path = SWEEP_DIR / SELECTED_RUN',
        'from arrival.run_paths import find_completed_run\nselected_path = find_completed_run(SWEEP_DIR / SELECTED_RUN)')
    s=s.replace('if (selected_path / "best.pt").exists():','if selected_path is not None and (selected_path / "best.pt").exists():')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=update(s)
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';s=p.read_text(encoding='utf-8');p.write_text(update(s),encoding='utf-8')
