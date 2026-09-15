"""Summarize a completed offline replay without rerunning model inference."""
import argparse
import json
from pathlib import Path
import pandas as pd


def ratio(a,b):
    return f'{100*a/b:.1f}%' if b else 'n/a'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/notification_replay')
    args=parser.parse_args();out=Path(args.output)
    summary=json.loads((out/'summary.json').read_text())
    alerts=pd.read_csv(out/'alerts.csv')
    lines=['# Continuous validation notification replay','',
        f"Evaluated {summary['ticks']} usable five-minute timestamps at {len(summary['locations'])} locations. "
        f"Missing-input timestamps: {summary['skipped_ticks']}. "
        f"Period: {summary['start']} to {summary['end']}.",'',
        f"Uses production mask p >= {summary['mask_threshold']}, component size >= {summary['min_size']}, "
        'the real one-pixel cross buffer, forecast threshold 0.003, observed threshold 0.01, and episode transitions.','',
        '## Notification outcomes','',
        '| Lead | Radar onsets | Warned before onset | Alerted at onset | Premature predicted clears | False start forecasts |',
        '|---|---:|---:|---:|---:|---:|']
    for lead,s in summary['leads'].items():
        a=alerts[alerts.lead==int(lead)]
        starts=a[(a.reason=='Rain is predicted at your location') & a.future_rain.notna()]
        false_starts=int((starts.future_rain==False).sum())
        s['start_alerts_scored']=len(starts);s['false_start_alerts']=false_starts
        s['observed_clear_alerts']=int((a.reason=='Radar no longer shows rain at your location').sum())
        lines.append(f"| +{lead} | {s['onsets']} | {s['warned_before_onset']} ({ratio(s['warned_before_onset'],s['onsets'])}) "
            f"| {s['alerted_at_onset']} | {s['premature_clears']}/{s['predicted_clears_scored']} "
            f"| {false_starts}/{len(starts)} |")
    lines += ['', 'Onsets are dry-to-wet changes at sampled locations. Advance warning means a start alert in the preceding forecast horizon; same-tick alerts are listed separately. '
        'A premature predicted clear means later radar is still wet at the forecast time. False start forecasts are dry at the forecast time, '
        'not necessarily an absence of rain throughout the following interval. Observed-clear alerts follow the observed dry transition by construction; '
        'this is not independent gauge validation.','', '## Original targets and region-size diagnostics','',
        'Tiny echoes are retained. Region detection means any overlap, not complete coverage. '
        'Full-radar counts retain all false positives; size-specific recall does not replace them.','',
        '| Lead | Target region size | Rain pixels | Pixel recall | Region overlap recall |',
        '|---|---|---:|---:|---:|']
    for lead,groups in summary['region_metrics'].items():
        for name,s in groups.items():
            if name=='all_pixels':continue
            lines.append(f"| +{lead} | {name} | {s['target_pixels']} | {ratio(s['hits'],s['target_pixels'])} "
                         f"| {ratio(s['detected_regions'],s['regions'])} |")
    lines += ['', '| Lead | Full-radar precision | Full-radar recall | Full-radar CSI |', '|---|---:|---:|---:|']
    for lead,groups in summary['region_metrics'].items():
        s=groups['all_pixels'];tp,fp,fn=s['tp'],s['fp'],s['fn']
        lines.append(f'| +{lead} | {ratio(tp,tp+fp)} | {ratio(tp,tp+fn)} | {ratio(tp,tp+fp+fn)} |')
    lines += ['', '## Limits',''] + ['- '+s for s in summary['limitations']]
    lines += ['', 'No sends, SQL writes, model updates or label cleanup were performed. '
        'These are validation diagnostics, not a new held-out result. Onset and clearing percentages are sensitive to tiny source echoes; '
        'review the spatially stratified cases before treating them as real user incident rates.','',
        'Machine-readable outputs: `alerts.csv`, `onsets.csv`, `local_forecasts.csv`, `summary.json`, `manifest.json`.']
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:16]))


if __name__=='__main__':main()
