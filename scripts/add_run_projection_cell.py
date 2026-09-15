"""Append and execute only the new offline experiment cell; preserve other cells."""
import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
path=root/'src/rain_arrival_model.ipynb'
n=json.loads(path.read_text(encoding='utf-8'))
source='''# Standalone v5 experiment: uses cached predictions, never retrains the model.
import sys, subprocess
from pathlib import Path
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/arrival").exists())
result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/experiment_arrival_projection.py")],
                        cwd=PROJECT_ROOT, capture_output=True, text=True)
print(result.stdout)
if result.returncode:
    raise RuntimeError(result.stderr)
'''
title='## Calibration experiment: independent curves with consistent horizons'
description=title+'''\n\nFits twelve cumulative logistic curves on calibration data, then applies least-squares isotonic projection across horizons. Tests probability error and ranking on held-out calibration dates and development validation. Projection prevents inconsistent horizons but may alter ranking. Uses the saved v5 checkpoint prediction caches; no training or deployment. Results are written to `reports/arrival_v5_calibration/projection_report.md`.'''
if not any(c.get('id')=='projection-experiment' for c in n['cells']):
    n['cells'].extend([dict(cell_type='markdown',id='projection-description',metadata={},source=description.splitlines(keepends=True)),
        dict(cell_type='code',id='projection-experiment',metadata={},source=source.splitlines(keepends=True),outputs=[],execution_count=None)])
cell=next(c for c in n['cells'] if c.get('id')=='projection-experiment')
r=subprocess.run([sys.executable,str(root/'scripts/experiment_arrival_projection.py')],cwd=root,capture_output=True,text=True)
cell['execution_count']=max([c.get('execution_count') or 0 for c in n['cells']])+1
cell['outputs']=[dict(output_type='stream',name='stdout',text=r.stdout.splitlines(keepends=True))]
if r.stderr:cell['outputs'].append(dict(output_type='stream',name='stderr',text=r.stderr.splitlines(keepends=True)))
path.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
# Preserve the new section if notebooks are regenerated later.
generator=root/'scripts/build_arrival_notebooks.py'
s=generator.read_text(encoding='utf-8')
if 'Calibration experiment: independent curves with consistent horizons' not in s:
    s=s.replace("if __name__=='__main__':",'main.extend([md('+repr(description)+'), code('+repr(source)+')])\n\n'+"if __name__=='__main__':")
    generator.write_text(s,encoding='utf-8')
print(r.stdout);print(r.stderr);sys.exit(r.returncode)
