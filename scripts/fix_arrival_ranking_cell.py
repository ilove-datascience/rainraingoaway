"""Make ranking evaluation independent of a stale imported metrics function."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
old = "from arrival.ranking import plot_ranking\n    plot_ranking(raw_metrics['ranking'])"
new = """from arrival.ranking import plot_ranking, ranking_report
    # Compute explicitly: a running kernel may still hold the older metrics().
    raw_metrics['ranking'] = ranking_report(raw_probs, validation_truth)
    baseline_results['persistence']['ranking'] = ranking_report(persistence_probs, baseline_truth)
    baseline_results['global_translation']['ranking'] = ranking_report(motion_probs, baseline_truth)
    baseline_results['gru_same_subset']['ranking'] = ranking_report(calibrated_probs(lp), yt)
    plot_ranking(raw_metrics['ranking'])"""
old_prior = "baseline_results['constant_training_prior'] = metrics(np.tile(prior, (len(yt),1)), yt)"
new_prior = old_prior + "\n    baseline_results['constant_training_prior']['ranking'] = ranking_report(np.tile(prior, (len(yt),1)), yt)"
for path in [root/'src/rain_arrival_model.ipynb', root/'scripts/build_arrival_notebooks.py', root/'scripts/update_arrival_v2.py']:
    if path.suffix == '.ipynb':
        doc = json.loads(path.read_text(encoding='utf-8'))
        for cell in doc['cells']:
            source = ''.join(cell['source'])
            updated = source.replace(old,new).replace(old_prior,new_prior)
            if updated != source:
                cell['source'] = updated.splitlines(keepends=True)
                cell['outputs'] = []; cell['execution_count'] = None
        path.write_text(json.dumps(doc,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    else:
        source = path.read_text(encoding='utf-8')
        path.write_text(source.replace(old,new).replace(old_prior,new_prior),encoding='utf-8')
