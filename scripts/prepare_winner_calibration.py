import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
extra='''
    import hashlib
    calibrator['checkpoint_sha256'] = hashlib.sha256((selected_path/'best.pt').read_bytes()).hexdigest()
    calibrator['calibration_records_hash'] = fingerprint(calibration_records)
    (selected_path/'calibrator.json').write_text(json.dumps(calibrator,indent=2))
    validation = make_dataset(state['model_config']['history'],tuple(state['channels']),'validation')
    validation_lp, validation_y = predict(selected,validation)
    comparison = {'raw':metrics(calibrated_probs(validation_lp),validation_y),
                  'projected':metrics(calibrated_probs(validation_lp,calibrator),validation_y)}
    (selected_path/'calibration_validation_metrics.json').write_text(json.dumps(comparison,indent=2))
    np.savez_compressed(selected_path/'calibration_predictions.npz',log_probs=log_probs.numpy(),labels=truth.numpy())
    display(pd.DataFrame([dict(method=name,horizon=h,AP=r['average_precision'],
        brier=m['horizons'][int(h)]['brier']) for name,m in comparison.items() for h,r in m['ranking'].items()]))
    plot_evaluation(calibrated_probs(validation_lp,calibrator),validation_y,comparison['projected'])
    print('Winner calibration ready for offline notification replay:',selected_path)
'''
replay='''# Run after the winner calibration cell; offline only, no notifications sent.
RUN_NOTIFICATION_REPLAY = False  # Set True to measure the new winner's alert performance.
if RUN_NOTIFICATION_REPLAY:
    import subprocess, sys
    if globals().get('selected_path') is None:
        raise RuntimeError('Run the selection/calibration cell first and select a completed run.')
    if not (selected_path / 'calibrator.json').is_file():
        raise RuntimeError(f'No saved calibration for {selected_path.name}. Set RUN_CALIBRATION = True in the preceding cell, run it to completion, then retry replay. No retraining is needed.')
    subprocess.run([sys.executable,str(PROJECT_ROOT/'scripts/replay_arrival_notifications.py'),
                    '--run',str(selected_path),'--calibration',str(selected_path/'calibrator.json'),
                    '--output',str(PROJECT_ROOT/'reports'/f'arrival_replay_{selected_path.name}')],
                   cwd=PROJECT_ROOT,check=True)
'''
def update(s):
    s=s.replace('RUN_EXPERIMENTS = True','RUN_EXPERIMENTS = False')
    s=s.replace('SELECTED_RUN = "gru_t6_all_seed67"','SELECTED_RUN = "gru_t6_depth2_spatial_seed67"')
    if 'SELECTED_RUN =' in s:
        s=s.replace('    calibrator = fit_calibrator(log_probs,truth)',
            '    from arrival.calibration import fit_projected_calibrator\n    calibrator = fit_projected_calibrator(log_probs,truth)')
        s=s.replace('    print("Calibrator saved. Freeze this candidate before evaluating any newly reserved test period.")',extra)
    return s
p=root/'src/rain_arrival_experiments.ipynb';n=json.loads(p.read_text(encoding='utf-8'))
for c in n['cells']:
    s=''.join(c['source']);u=update(s)
    if '## Calibrate the selected candidate' in s:
        u='''## Calibrate the winning two-layer spatial GRU

Training is disabled by default. Run setup/data cells and this calibration cell to load the saved winner; no retraining is needed. Fit twelve cumulative logistic curves on the separate calibration partition, then project across horizons to prevent crossings. This is the improved v5 candidate method, refitted to the winner. Calibration-fit scores are in-sample; validation scores are development results because validation selected this model. The optional replay cell below measures arrival notifications offline. Leave final testing disabled.'''
    if u!=s:
        c['source']=u.splitlines(keepends=True)
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
if not any(c.get('id')=='winner-replay' for c in n['cells']):
    i=next(i for i,c in enumerate(n['cells']) if 'SELECTED_RUN =' in ''.join(c['source']))
    n['cells'].insert(i+1,dict(cell_type='code',id='winner-replay',metadata={},source=replay.splitlines(keepends=True),outputs=[],execution_count=None))
p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
# Keep the generator's experiment section synchronized without touching main notebook.
p=root/'scripts/build_arrival_notebooks.py';s=p.read_text(encoding='utf-8')
a=s.index('experiments=');b=s.index("if __name__=='__main__':")
section=update(s[a:b]);s=s[:a]+section+s[b:]
insertion='experiments.insert(next(i for i,c in enumerate(experiments) if "SELECTED_RUN =" in "".join(c["source"]))+1,code('+repr(replay)+'))\n\n'
s=s.replace("if __name__=='__main__':",insertion+"if __name__=='__main__':")
p.write_text(s,encoding='utf-8')
