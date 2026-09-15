"""Compare source decoding and causal cleanup candidates on validation only."""
from pathlib import Path
from datetime import datetime,timedelta
import sys,json,re
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_processing.pngtojson import intensityColors

root=Path(__file__).resolve().parents[1]
out=root/'reports/source_check'
html=(out/'data_gov.html').read_text(encoding='utf-8')
legend=re.findall(r'\| (\d+), (\d+), (\d+) \| `(#\w{6})` \| ([^|]+) \|',html)
assert len(legend)==33, len(legend)
palette=np.array([[int(x) for x in row[:3]] for row in legend],dtype=np.int32)
old=np.array([[int(c[i:i+2],16) for i in (1,3,5)] for c in intensityColors],dtype=np.int32)
nearest=((palette[:,None,:]-old[None,:,:])**2).sum(2).argmin(1)
legacy_values=np.ceil((nearest+1)/30*100)/100
table=pd.DataFrame(dict(source_index=np.arange(33),hex=[r[3] for r in legend],
    official_category=[r[4].strip() for r in legend],legacy_index=nearest,
    legacy_normalized=legacy_values,source_rank_normalized=np.ceil((np.arange(33)+1)/33*100)/100))
table['bot_category']=np.where(legacy_values<1/3,'Light',np.where(legacy_values<2/3,'Moderate','Heavy'))
table.to_csv(out/'palette_mapping.csv',index=False)
packed=palette[:,0]*65536+palette[:,1]*256+palette[:,2]
order=np.argsort(packed);packed_sorted=packed[order]
groups=json.loads((out/'validation_groups.json').read_text())
centers=sorted(set(g[h] for g in groups for h in (3,4)))
needed=set()
for name in centers:
    t=datetime.strptime(name,'%Y%m%d%H%M')
    needed.update((t+timedelta(minutes=d)).strftime('%Y%m%d%H%M') for d in (-10,-5,0,5))
decoded={};colors=np.zeros(33,dtype=np.int64);unknown=0
structure=np.ones((3,3),dtype=np.uint8)
for i,name in enumerate(sorted(needed)):
    p=root/'data/70km/png'/f'{name}.png'
    if not p.exists():continue
    rgba=np.array(Image.open(p).convert('RGBA'));rgb=rgba[:,:,:3].astype(np.int32)
    codes=rgb[:,:,0]*65536+rgb[:,:,1]*256+rgb[:,:,2]
    pos=np.searchsorted(packed_sorted,codes);pos=np.minimum(pos,32)
    recognized=packed_sorted[pos]==codes;opaque=rgba[:,:,3]>0
    unknown+=int((opaque&~recognized).sum())
    bins=order[pos]
    colors+=np.bincount(bins[opaque&recognized],minlength=33)
    # Do not silently invent a mapping for an unexpected source color.
    if (opaque&~recognized).any():raise ValueError('Unrecognized opaque source RGB in '+name)
    raw=legacy_values[bins].astype(np.float32);raw[~opaque]=0
    labels,_=ndimage.label(raw>.01,structure)
    sizes=np.bincount(labels.ravel());peaks=ndimage.maximum(raw,labels,index=np.arange(len(sizes)))
    keep=(sizes>=4)|(peaks>=.10);keep[0]=False
    clean=raw*keep[labels]
    decoded[name]=clean
    if (i+1)%500==0:print('Decoded',i+1,'validation/context frames',flush=True)
table['opaque_pixels']=colors
table.to_csv(out/'palette_mapping.csv',index=False)
rows=[];hotspots=[]
for center in centers:
    t=datetime.strptime(center,'%Y%m%d%H%M')
    names=[(t+timedelta(minutes=d)).strftime('%Y%m%d%H%M') for d in (-10,-5,0,5)]
    if not all(n in decoded for n in names):continue
    grids=[decoded[n] for n in names];masks=[g>.01 for g in grids]
    labels=[ndimage.label(m,structure)[0] for m in masks]
    sizes=[np.bincount(l.ravel()) for l in labels]
    previous_support=ndimage.binary_dilation(masks[0]|masks[1],structure,iterations=2)
    next_support=ndimage.binary_dilation(masks[3],structure,iterations=2)
    for x,y in [(107,70),(144,40),(140,43)]:
        label=labels[2][y,x]
        hotspots.append(dict(time=center,x=x,y=y,wet=bool(masks[2][y,x]),
            small=bool(label and sizes[2][label]<=5),intensity=float(grids[2][y,x])))
    for component in np.flatnonzero((sizes[2]>=1)&(sizes[2]<=5)):
        if component==0:continue
        region=labels[2]==component;peak=float(grids[2][region].max())
        past=bool((region&previous_support).any());future=bool((region&next_support).any())
        exact=[]
        for h in (0,1):
            ids=np.unique(labels[h][region]);ids=ids[ids!=0]
            exact.append(any(np.array_equal(labels[h]==i,region) for i in ids))
        nearby_ids=np.unique(labels[3][ndimage.binary_dilation(region,structure,iterations=2)])
        nearby_ids=nearby_ids[nearby_ids!=0]
        larger_next=any(sizes[3][j]>5 for j in nearby_ids)
        rows.append(dict(time=center,size=int(region.sum()),peak=peak,past_support=past,
            future_support=future,larger_next=larger_next,
            remove_all=True,
            remove_unsupported_weak=peak<=.30 and not past,
            remove_fixed_weak=peak<=.30 and all(exact)))
df=pd.DataFrame(rows);df.to_csv(out/'validation_small_components.csv',index=False)
pd.DataFrame(hotspots).to_csv(out/'validation_hotspots.csv',index=False)
summary={'source_url':'https://data.gov.sg/datasets/d_418e9ac3414fd927b7405631e0a7bc82/view',
    'decoded_frames':len(decoded),'unknown_opaque_pixels':unknown,'source_colors_seen':int((colors>0).sum()),
    'legacy_unique_bins':len(set(nearest.tolist())),
    'category_disagreement_pure_categories_pixels':int(colors[(table.official_category.isin(['Light','Moderate','Heavy']) & (table.official_category!=table.bot_category)).values].sum()),
    'opaque_pixels':int(colors.sum()),'eligible_centers':int(df.time.nunique()),'small_components':len(df),
    'small_pixels':int(df['size'].sum()),'rules':{}}
for rule in ['remove_all','remove_unsupported_weak','remove_fixed_weak']:
    part=df[df[rule]]
    summary['rules'][rule]=dict(components=len(part),pixels=int(part['size'].sum()),
        with_future_support=int(part.future_support.sum()),with_larger_next=int(part.larger_next.sum()),
        strong_components=int((part.peak>1/3).sum()))
(out/'solution_checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
# Longer continuous pre-test sequence around a training-derived recurring hotspot.
start=datetime(2026,6,2,0,0);longrows=[]
for i in range(288):
    name=(start+timedelta(minutes=5*i)).strftime('%Y%m%d%H%M')
    p=root/'data/70km/png'/f'{name}.png'
    if not p.exists():continue
    rgba=np.array(Image.open(p).convert('RGBA'))
    for x,y in [(107,70),(144,40),(140,43)]:
        color=rgba[y,x]
        longrows.append(dict(time=name,x=x,y=y,opaque=bool(color[3]),rgb=','.join(map(str,color[:3]))))
pd.DataFrame(longrows).to_csv(out/'continuous_hotspots_june2.csv',index=False)
print('Completed causal candidate and continuous hotspot checks.',flush=True)
