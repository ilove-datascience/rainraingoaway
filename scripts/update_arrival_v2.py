"""Update notebook sources without replacing existing saved outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
extra = '''
    from arrival.ranking import plot_ranking, ranking_report
    # Compute explicitly: a running kernel may still hold the older metrics().
    raw_metrics['ranking'] = ranking_report(raw_probs, validation_truth)
    baseline_results['persistence']['ranking'] = ranking_report(persistence_probs, baseline_truth)
    baseline_results['global_translation']['ranking'] = ranking_report(motion_probs, baseline_truth)
    baseline_results['gru_same_subset']['ranking'] = ranking_report(calibrated_probs(lp), yt)
    plot_ranking(raw_metrics['ranking'])
    # Constant probabilities estimated exclusively from natural training labels.
    prior = np.asarray(prepared['class_counts']['train'], dtype=float)
    prior /= prior.sum()
    baseline_results['constant_training_prior'] = metrics(np.tile(prior, (len(yt),1)), yt)
    baseline_results['constant_training_prior']['ranking'] = ranking_report(np.tile(prior, (len(yt),1)), yt)
    comparison = []
    for name, result in baseline_results.items():
        for horizon, ranking in result['ranking'].items():
            comparison.append(dict(model=name, horizon=int(horizon),
                average_precision=ranking['average_precision'], prevalence=ranking['prevalence'],
                brier=result['horizons'][int(horizon)]['brier']))
    display(pd.DataFrame(comparison))
    (RUN_DIR / 'validation_metrics.json').write_text(json.dumps(raw_metrics,indent=2))
    (RUN_DIR / 'matched_baselines.json').write_text(json.dumps(baseline_results,indent=2))
'''
old = 'Training chooses equally among the nonempty arrival groups at each anchor. Empty groups are skipped; the overall mix is not guaranteed to be exactly 25% each.'
new = 'Training selects one of four arrival groups with equal probability globally, then an eligible training anchor and location. Expected group frequencies are 25% each; individual batches vary.'
for path in (ROOT/'src').glob('rain_arrival_*.ipynb'):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        updated = source.replace(old,new).replace('gru_t6_all_seed67_v1','gru_t6_all_seed67_v2').replace('experiments_v1','experiments_v2')
        if 'raw_metrics = metrics(' in updated and 'matched_baselines.json' not in updated:
            updated += '\n' + extra
        if updated != source:
            cell['source'] = updated.splitlines(keepends=True)
            if cell['cell_type']=='code': cell['outputs']=[];cell['execution_count']=None
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
# Keep the generator consistent with the updated notebooks.
generator = ROOT/'scripts/build_arrival_notebooks.py'
source = generator.read_text(encoding='utf-8').replace(old,new).replace('gru_t6_all_seed67_v1','gru_t6_all_seed67_v2').replace('experiments_v1','experiments_v2')
needle = "'''),md('''\n## Calibrate probabilities"
if 'matched_baselines.json' not in source:
    source = source.replace(needle, extra + needle)
generator.write_text(source,encoding='utf-8')
