"""Read-only temporal and PNG-decoding audit of pre-test radar echoes."""
from pathlib import Path
from datetime import datetime, timedelta
import sys, json
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_processing.pngtojson import intensityColors
from data_processing.data_loading import remove_small_echoes

root = Path(__file__).resolve().parents[1]
folder = root / 'data/70km/png'
out = root / 'reports/small_echo_audit'
out.mkdir(parents=True, exist_ok=True)
palette = np.array([[int(c[i:i+2],16) for i in (1,3,5)] for c in intensityColors], dtype=np.int32)
files = {p.stem:p for p in folder.glob('*.png') if p.stem < '202608261945'}
eligible = []
for name in sorted(files):
    t = datetime.strptime(name,'%Y%m%d%H%M')
    neighbors = [(t+timedelta(minutes=d)).strftime('%Y%m%d%H%M') for d in (-5,0,5)]
    if all(n in files for n in neighbors): eligible.append(neighbors)
selected = [eligible[i] for i in np.linspace(0,len(eligible)-1,min(240,len(eligible)),dtype=int)]
structure = np.ones((3,3),dtype=np.uint8)
rows, image_rows, examples = [], [], {}
occurrence = np.zeros((120,217),dtype=np.int64)
def decode(name):
    rgba=np.array(Image.open(files[name]).convert('RGBA'))
    colors,inverse=np.unique(rgba[:,:,:3].reshape(-1,3),axis=0,return_inverse=True)
    distances=((colors.astype(np.int32)[:,None,:]-palette[None,:,:])**2).sum(2)
    nearest=distances.argmin(1)
    values=np.ceil((nearest+1)/len(palette)*100)/100
    raw=values[inverse].reshape(rgba.shape[:2]).astype(np.float32)
    raw[rgba[:,:,3]==0]=0
    exact=(distances.min(1)==0)[inverse].reshape(raw.shape)
    return raw,remove_small_echoes(raw),rgba,exact
for sample,names in enumerate(selected):
    decoded=[decode(n) for n in names]
    grids=[a[1] for a in decoded]
    masks=[g>.01 for g in grids]
    labeled=[ndimage.label(m,structure)[0] for m in masks]
    sizes=[np.bincount(l.ravel()) for l in labeled]
    adjacent=[ndimage.binary_dilation(m,structure,iterations=2) for m in (masks[0],masks[2])]
    raw,clean,rgba,exact=decoded[1]
    small_ids=np.flatnonzero((sizes[1]>=1)&(sizes[1]<=5));small_ids=small_ids[small_ids!=0]
    small=np.isin(labeled[1],small_ids)
    occurrence+=small
    image_rows.append(dict(time=names[1],rain_pixels=int(masks[1].sum()),small_pixels=int(small.sum()),
        raw_rain_pixels=int((raw>.01).sum()), nonpalette_small=int((small&~exact).sum()),
        partial_alpha_small=int((small&(rgba[:,:,3]<255)).sum())))
    for region_id in small_ids:
        region=labeled[1]==region_id
        exact_neighbors=[]
        large_neighbors=[]
        for h in [0,2]:
            ids=np.unique(labeled[h][region]);ids=ids[ids!=0]
            exact_neighbors.append(any(np.array_equal(labeled[h]==i,region) for i in ids))
            nearby_ids=np.unique(labeled[h][ndimage.binary_dilation(region,structure,iterations=2)])
            nearby_ids=nearby_ids[nearby_ids!=0]
            large_neighbors.append(any(sizes[h][i]>5 for i in nearby_ids))
        before=bool((region&adjacent[0]).any());after=bool((region&adjacent[1]).any())
        row=dict(time=names[1],size=int(region.sum()),peak=float(clean[region].max()),
            exact_both=all(exact_neighbors),exact_either=any(exact_neighbors),
            nearby_before=before,nearby_after=after,isolated=not before and not after,
            near_larger=any(large_neighbors),x=float(np.where(region)[1].mean()),y=float(np.where(region)[0].mean()))
        rows.append(row)
        category='fixed' if row['exact_both'] else 'isolated' if row['isolated'] else 'near_larger' if row['near_larger'] else 'other'
        if category not in examples: examples[category]=(names,grids,region)
    if (sample+1)%60==0: print('Audited',sample+1,'triplets',flush=True)
events=pd.DataFrame(rows);frames=pd.DataFrame(image_rows)
events.to_csv(out/'components.csv',index=False);frames.to_csv(out/'frames.csv',index=False)
summary={'triplets':len(selected),'eligible_triplets':len(eligible),'components':len(events),
    'component_rates':{k:float(events[k].mean()) for k in ['exact_both','exact_either','nearby_before','nearby_after','isolated','near_larger']},
    'sizes':events.groupby('size').agg(count=('size','size'),peak_median=('peak','median'),fixed_rate=('exact_both','mean'),isolated_rate=('isolated','mean')).to_dict('index'),
    'small_pixels':int(frames.small_pixels.sum()),'all_rain_pixels':int(frames.rain_pixels.sum()),
    'nonpalette_small':int(frames.nonpalette_small.sum()),'partial_alpha_small':int(frames.partial_alpha_small.sum()),
    'peak_quantiles':events.peak.quantile([0,.25,.5,.75,1]).to_dict(),
    'range':[selected[0][1],selected[-1][1]]}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
fig,axes=plt.subplots(len(examples),3,figsize=(12,3*len(examples)),constrained_layout=True)
for row,(category,(names,grids,region)) in enumerate(examples.items()):
    ys,xs=np.where(region);x0=max(0,int(xs.min())-12);x1=min(217,int(xs.max())+13);y0=max(0,int(ys.min())-12);y1=min(120,int(ys.max())+13)
    for col in range(3):
        axes[row,col].imshow(grids[col][y0:y1,x0:x1],vmin=0,vmax=1,cmap='viridis')
        axes[row,col].scatter(xs-x0,ys-y0,facecolors='none',edgecolors='red',s=75)
        axes[row,col].set_title(category+' | '+names[col]);axes[row,col].axis('off')
fig.suptitle('Small echo audit: same center-frame coordinates marked in all three frames')
fig.savefig(out/'temporal_examples.png',dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,6));im=ax.imshow(occurrence,cmap='magma');fig.colorbar(im,ax=ax,label='Sampled center frames containing a 1–5 pixel echo')
ax.set_title('Small-echo spatial recurrence across 240 pre-test center frames');fig.savefig(out/'spatial_recurrence.png',dpi=130);plt.close(fig)
