"""Configure a longer ranking-selected run without overwriting earlier outputs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
extra='''
    # Full fixed validation sample, not just the 64-anchor baseline subset.
    # These are development scores: checkpoint selection already used validation.
    full_motion, _ = motion_baseline(store, validation_records, DATA)
    prior = np.asarray(prepared['class_counts']['train'], dtype=float)
    prior /= prior.sum()
    full_results = {
        'raw_gru': metrics(raw_probs, validation_truth),
        'calibrated_gru': metrics(calibrated_probs(validation_log_probs,calibrator), validation_truth),
        'motion': metrics(full_motion, validation_truth),
        'constant_training_prior': metrics(np.tile(prior,(len(validation_truth),1)),validation_truth),
    }
    rows=[]
    for name,result in full_results.items():
        for horizon,ranking in result['ranking'].items():
            scores=result['horizons'][int(horizon)]
            rows.append(dict(model=name,horizon=int(horizon),AP=ranking['average_precision'],
                             prevalence=ranking['prevalence'],**scores))
    display(pd.DataFrame(rows))
    (RUN_DIR/'full_validation_comparison.json').write_text(json.dumps(full_results,indent=2))
    plot_ranking(full_results['calibrated_gru']['ranking'])
'''
def update(source):
    if 'RUN_NAME =' in source:
        for version in ('v1','v2','v3'):
            source=source.replace('gru_t6_all_seed67_'+version,'gru_t6_all_seed67_v4')
        source=source.replace('UPDATE_BUDGET = 3000','UPDATE_BUDGET = 10000')
        source=source.replace('patience=8, batch=16, seed=SEED','patience=16, batch=16, seed=SEED, selection="mean_ap"')
    source=source.replace('Validation NLL selects the checkpoint.',
        'Mean validation average precision across 15/30/60 minutes selects the checkpoint. The lowest-NLL checkpoint is saved separately as best_nll.pt. The v4 run uses up to 10,000 updates, stopping after 16 checks without ranking improvement.')
    return source
path=root/'src/rain_arrival_model.ipynb'
n=json.loads(path.read_text(encoding='utf-8'))
for cell in n['cells']:
    original=''.join(cell['source']); source=update(original)
    if 'calibration_fit_metrics.json' in source and 'full_validation_comparison.json' not in source:
        source+='\n'+extra
    if source!=original:
        cell['source']=source.splitlines(keepends=True)
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
path.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
# Update the generator's main notebook cells, preserving experiment budgets.
path=root/'scripts/build_arrival_notebooks.py'
s=path.read_text(encoding='utf-8')
start=s.index('main=[');end=s.index('experiments=')
main=update(s[start:end])
needle="'''),md('''\n## Historical-only inference"
if 'full_validation_comparison.json' not in main:main=main.replace(needle,extra+needle)
path.write_text(s[:start]+main+s[end:],encoding='utf-8')
