import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old='            path = SWEEP_DIR / f"{experiment[\'name\']}_seed{seed}"'
new='''            from arrival.run_paths import sweep_run_path
            path = sweep_run_path(SWEEP_DIR / f"{experiment['name']}_seed{seed}")'''
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for cell in n['cells']:
    s=''.join(cell['source']);updated=s.replace(old,new)
    updated=updated.replace('incomplete checkpoints are protected and require a new run directory or explicit recovery.',
        'interrupted runs are preserved and restarted from scratch in numbered attempt folders; this is not an optimizer-state resume.')
    if updated!=s:
        cell['source']=updated.splitlines(keepends=True)
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';s=p.read_text(encoding='utf-8').replace(old,new);p.write_text(s,encoding='utf-8')
