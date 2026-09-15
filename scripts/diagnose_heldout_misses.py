"""Explain fixed-setting test errors; does not select new settings or train."""
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the exact checked dataset, normalization and checkpoint setup.
setup = Path(__file__).with_name('run_heldout_test.py').read_text(encoding='utf-8')
exec(compile(setup.split('# HELD-OUT TEST:')[0], str(__file__), 'exec'))
out = Path('reports/heldout_misses')
out.mkdir(parents=True, exist_ok=True)
keys = sorted(dataset.data)
runs, run, previous = [], [], None
for key in keys:
    tick = datetime.strptime(key.removesuffix('.png'), '%Y%m%d%H%M')
    if previous is not None and tick - previous != timedelta(minutes=5):
        runs.append(run)
        run = []
    run.append(key)
    previous = tick
runs.append(run)
groups = [run[i:i+5] for run in runs for i in range(0, len(run)-4, 5)]
assert len(groups) == len(dataset)
rows, regions, arrays = [], [], []
structure = np.ones((3, 3), dtype=np.uint8)
offset = 0
model2.eval()
with torch.no_grad():
    for batch, (inputs, targets) in enumerate(test_loader, 1):
        inputs = inputs.to(device)
        raw5, _, _, log5 = model2(inputs, return_logits=True)
        raw10, _, _, log10 = model2(build_rollout_inputs(inputs, raw5), return_logits=True)
        last = inputs[:, -1, 0].cpu().numpy()
        raw = torch.cat([raw5, raw10], 1).cpu().numpy()
        probability = torch.cat([log5.sigmoid(), log10.sigmoid()], 1).cpu().numpy()
        targets = targets.numpy()
        for b in range(len(targets)):
            sample = offset + b
            arrays.append((last[b], targets[b], raw[b], probability[b]))
            for h, lead in enumerate((5, 10)):
                target, pred, prob = targets[b,h], np.maximum(raw[b,h], 0), probability[b,h]
                rain = target > .01
                base = prob >= .4
                labels, _ = ndimage.label(base, structure)
                keep = np.bincount(labels.ravel()) >= 40
                keep[0] = False
                mask = keep[labels]
                final = mask & (pred > .01)
                missed = rain & ~final
                prior = last[b] > .01
                nearby = ndimage.binary_dilation(final, structure, iterations=2)
                row = dict(sample=sample, lead=lead, observed=groups[test_indices[sample]][2],
                           target_time=groups[test_indices[sample]][3+h], rain_pixels=int(rain.sum()),
                           missed=int(missed.sum()), predicted=int(final.sum()),
                           false_positive=int((final & ~rain).sum()),
                           low_probability=int((missed & ~base).sum()),
                           removed_by_filter=int((missed & base & ~mask).sum()),
                           low_intensity=int((missed & mask).sum()),
                           persistence_catches=int((missed & prior).sum()),
                           ongoing_pixels=int((rain & prior).sum()),
                           ongoing_missed=int((missed & prior).sum()),
                           new_pixels=int((rain & ~prior).sum()),
                           new_missed=int((missed & ~prior).sum()),
                           within_2px=int((missed & nearby).sum()))
                for name, lo, hi in [('weak', .01, .1), ('medium', .1, 1/3), ('strong', 1/3, np.inf)]:
                    band = (target > lo) & (target <= hi)
                    row[name+'_pixels'] = int(band.sum())
                    row[name+'_missed'] = int((band & missed).sum())
                rows.append(row)
                truth_labels, count = ndimage.label(rain, structure)
                for component in range(1, count+1):
                    region = truth_labels == component
                    size = int(region.sum())
                    regions.append(dict(sample=sample, lead=lead, size=size,
                        size_bin='1-5' if size<=5 else '6-20' if size<=20 else '21-100' if size<=100 else '>100',
                        detected=bool((region & final).any()), missed=int((region & missed).sum())))
        offset += len(targets)
        if batch % 10 == 0:
            print(f'Diagnostic progress: {batch}/{len(test_loader)}', flush=True)
events = pd.DataFrame(rows)
region_table = pd.DataFrame(regions)
events.to_csv(out/'events.csv', index=False)
region_table.to_csv(out/'regions.csv', index=False)
# Cache fixed predictions so later inspection does not require another GPU pass.
np.savez_compressed(out/'fixed_predictions.npz',
    last=np.stack([a[0] for a in arrays]), target=np.stack([a[1] for a in arrays]),
    raw=np.stack([a[2] for a in arrays]), probability=np.stack([a[3] for a in arrays]))
summary = {}
for lead in (5, 10):
    table = events[events.lead == lead]
    sums = table.drop(columns=['sample', 'lead']).select_dtypes(include='number').sum().to_dict()
    sums['rainy_frames'] = int((table.rain_pixels > 0).sum())
    sums['completely_missed_rainy_frames'] = int(((table.rain_pixels>0) & (table.missed==table.rain_pixels)).sum())
    sums['regions'] = region_table[region_table.lead == lead].groupby('size_bin').agg(
        count=('size','size'), detected=('detected','sum'), pixels=('size','sum'), missed=('missed','sum')).to_dict('index')
    summary[lead] = sums
    print('LEAD', lead, json.dumps(sums), flush=True)
    worst = table.nlargest(3, 'missed')
    print(worst.to_string(index=False), flush=True)
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), constrained_layout=True)
    for row_index, (_, event) in enumerate(worst.iterrows()):
        last, target, raw, probability = arrays[int(event['sample'])]
        h = 0 if lead == 5 else 1
        labels, _ = ndimage.label(probability[h] >= .4, structure)
        keep = np.bincount(labels.ravel()) >= 40
        keep[0] = False
        gated = np.maximum(raw[h], 0) * keep[labels]
        misses = (target[h] > .01) & ~(gated > .01)
        panels = [last, target[h], probability[h], gated, misses]
        for col, (panel, title) in enumerate(zip(panels, ['Last input', 'Actual target', 'Probability', 'Fixed forecast', 'Missed rain'])):
            axes[row_index,col].imshow(panel, vmin=0, vmax=1, cmap='magma' if col==2 else 'viridis')
            axes[row_index,col].set_title(title)
            axes[row_index,col].set_xticks([])
            axes[row_index,col].set_yticks([])
        axes[row_index,0].set_ylabel(str(event['target_time'])+'\n'+str(int(event['missed']))+' missed pixels')
    fig.suptitle(f'+{lead} minutes: three frames with most missed rainy pixels (not independent storms)')
    fig.savefig(out/f'worst_plus{lead}.png', dpi=130)
    plt.close(fig)
(out/'summary.json').write_text(json.dumps(summary, indent=2))
