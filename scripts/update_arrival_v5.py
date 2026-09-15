"""Configure longer training and coherent rank-preserving calibration."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
extra='''
    # Compare the previous calibration on the same development examples.
    from arrival.training import fit_legacy_calibrator
    legacy = fit_legacy_calibrator(calibration_log_probs, calibration_truth)
    legacy_result = metrics(calibrated_probs(validation_log_probs,legacy),validation_truth)
    (RUN_DIR/'legacy_calibration_comparison.json').write_text(json.dumps(legacy_result,indent=2))
    print('Legacy calibration AP:', {h:r['average_precision'] for h,r in legacy_result['ranking'].items()})
    # Threshold tradeoffs are development diagnostics, not chosen deployment settings.
    threshold_rows=[]
    calibrated_validation=calibrated_probs(validation_log_probs,calibrator)
    truth_array=np.asarray(validation_truth)
    for minutes in (15,30,60):
        probability=calibrated_validation[:,:minutes//5].sum(1)
        truth=truth_array<minutes//5
        for threshold in (.01,.02,.05,.10,.20,.30,.50):
            decision=probability>=threshold
            tp=int((decision & truth).sum()); fp=int((decision & ~truth).sum())
            threshold_rows.append(dict(horizon=minutes,threshold=threshold,
                precision=tp/max(tp+fp,1),recall=tp/max(int(truth.sum()),1),
                true_alerts=tp,false_alerts=fp,missed=int(truth.sum())-tp))
    display(pd.DataFrame(threshold_rows))
    (RUN_DIR/'validation_threshold_tradeoffs.json').write_text(json.dumps(threshold_rows,indent=2))
'''
def replace(s):
    s=s.replace('gru_t6_all_seed67_v4','gru_t6_all_seed67_v5').replace('UPDATE_BUDGET = 10000','UPDATE_BUDGET = 30000')
    s=s.replace('patience=16, batch=16','patience=40, batch=16')
    s=s.replace('The v4 run uses up to 10,000 updates, stopping after 16 checks','The v5 run uses up to 30,000 updates, stopping after 40 checks')
    s=s.replace('Temperature plus a regularized class bias is fitted on calibration data to account for balanced training sampling.',
        'A shared increasing logistic map calibrates cumulative arrival probabilities using calibration data only. It preserves ranking at every horizon and ensures longer-horizon arrival probability never decreases. The previous class-bias calibration is retained as a comparison. This shared map is less flexible, so compare probability error as well as ranking.')
    return s
for path in (root/'src').glob('rain_arrival_*.ipynb'):
    n=json.loads(path.read_text(encoding='utf-8'))
    for c in n['cells']:
        s=''.join(c['source']);u=replace(s)
        if 'full_validation_comparison.json' in u and 'validation_threshold_tradeoffs.json' not in u:u+='\n'+extra
        if u!=s:
            c['source']=u.splitlines(keepends=True)
            if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
    path.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
path=root/'scripts/build_arrival_notebooks.py'
s=replace(path.read_text(encoding='utf-8'))
needle="'''),md('''\n## Historical-only inference"
if 'validation_threshold_tradeoffs.json' not in s:s=s.replace(needle,extra+needle)
path.write_text(s,encoding='utf-8')
