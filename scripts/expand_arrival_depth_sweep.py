"""Enable additional depth/width ablations in the existing sweep."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
variants=[
    dict(name='gru_t6_depth2',recurrent_layers=2),
    dict(name='gru_t6_depth3',recurrent_layers=3),
    dict(name='gru_t6_encoder_deep',encoder_extra_layers=2),
    dict(name='gru_t6_wide96_spatial',hidden_dim=96,pooling='spatial'),
    dict(name='gru_t6_depth2_spatial',recurrent_layers=2,pooling='spatial'),
    dict(name='gru_t6_wide96_depth2_spatial',hidden_dim=96,recurrent_layers=2,pooling='spatial'),
]
rows=[]
for v in variants:
    spec=dict(kind='gru',history=6,channels=tuple(range(7)),output='categorical',hidden_dim=64,pooling='global',recurrent_layers=1,encoder_extra_layers=0)
    spec.update(v);rows.append('    '+repr(spec)+',')
enabled=repr({'gru_t6_all','gru_t6_wide96','gru_t6_spatial'}|{v['name'] for v in variants})
def update(s):
    if 'EXPERIMENTS = [' in s and 'gru_t6_depth2\'' not in s:
        s=s.replace('EXPERIMENTS = [','EXPERIMENTS = [\n'+'\n'.join(rows))
    s=s.replace('ENABLED = {"gru_t6_all","gru_t6_wide96","gru_t6_spatial"}','ENABLED = '+enabled)
    s=s.replace('pooling=experiment.get("pooling","global")).to(DEVICE)',
        'pooling=experiment.get("pooling","global"),\n                recurrent_layers=experiment.get("recurrent_layers",1),\n                encoder_extra_layers=experiment.get("encoder_extra_layers",0)).to(DEVICE)')
    marker='            seed_everything(seed)\n            training ='
    if marker in s:
        s=s.replace(marker,'''            if (path / "result.json").exists():
                completed = json.loads((path / "result.json").read_text())
                if (completed["manifest_hash"] != prepared["manifest_hash"] or
                    completed["validation_records_hash"] != fingerprint(validation_records) or
                    json.dumps(completed["experiment"],sort_keys=True) != json.dumps(experiment,sort_keys=True)):
                    raise ValueError(f"Completed run configuration differs: {path}")
                print("Skipping completed run:", path.name)
                continue
            seed_everything(seed)
            training =''')
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=update(s)
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
note='''\n\nExpanded sweep: nine enabled models, including 2/3 recurrent layers, two extra stride-1 encoder convolutions, and combined depth/width/spatial heads. All retain six inputs, seven channels, batch 16 and a 30,000-update maximum. Completed matching runs are skipped; incomplete checkpoints are protected and require a new run directory or explicit recovery. Run models sequentially with other GPU notebook kernels shut down. Depth increases runtime and memory; parameter counts and peak allocated memory are recorded. No automatic batch-size changes are made, preserving comparisons.'''
n['cells'][0]['source']+=note.splitlines(keepends=True)
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
p=root/'scripts/build_arrival_notebooks.py';s=update(p.read_text(encoding='utf-8'))
s=s.replace("if __name__=='__main__':",'experiments[0]["source"] += '+repr(note.splitlines(keepends=True))+'\n\n'+"if __name__=='__main__':")
p.write_text(s,encoding='utf-8')
