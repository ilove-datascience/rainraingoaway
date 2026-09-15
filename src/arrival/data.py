"""Causal frame loading, complete-window manifests and dynamic target sampling."""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from data_processing.radar_codec import SOURCE, decode_png
from data_processing.data_loading import build_env_data, remove_small_echoes, RADAR_CLEANING_VERSION

CHANNELS = ('radar','temperature','humidity','wind_u','wind_v','station_mask','distance')
CLASSES = tuple(f'{i*5}–{(i+1)*5} min' for i in range(12)) + ('No rain within 60 min',)


@dataclass(frozen=True)
class DataConfig:
    crop: int = 64
    max_history: int = 12  # identical eligible anchors for T=3/6/9/12
    future: int = 12
    patch: int = 5
    rain_threshold: float = .01
    coverage: float = .20
    stride: int = 1
    development_end: str = '2026-08-26'  # exclusive; old held-out period excluded
    test_start: str | None = None  # configure ONLY a genuinely new untouched period
    seed: int = 67

    def __post_init__(self):
        if self.future!=12 or self.max_history not in (3,6,9,12):raise ValueError('Require 12 future steps and supported history')
        if self.crop<self.patch or self.patch%2!=1 or not 0<self.coverage<=1 or self.stride<1:
            raise ValueError('Invalid crop, neighborhood, coverage or stride')


def tick_time(key):
    return datetime.strptime(key, '%Y%m%d%H%M')


def shifted(key, minutes):
    return (tick_time(key)+timedelta(minutes=minutes)).strftime('%Y%m%d%H%M')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save_array(path,array):
    temporary=path.with_name(path.stem+'.'+uuid.uuid4().hex+'.npy')
    try:
        np.save(temporary,array)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FrameStore:
    """Never consult a weather CSV later than the frame, or future observations.

    Archived observation times are available; historical API publication times are
    not. This enforces observation-time causality, not a measured delivery replay.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.radar_dir = self.root/'data/70km/png'
        self.env_dir = self.root/'data/environment'
        self.cache = self.root/'data/arrival_cache'/SOURCE/RADAR_CLEANING_VERSION/'causal_env_v1'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.keys = sorted(p.stem for p in self.radar_dir.glob('*.png') if len(p.stem)==12 and p.stem.isdigit())
        self.key_set = set(self.keys)

    @lru_cache(maxsize=256)
    def radar(self,key):
        return remove_small_echoes(decode_png(self.radar_dir/f'{key}.png',SOURCE))

    def signature(self,key):
        paths=[self.radar_dir/f'{key}.png',self.env_dir/f'weather_{key}.csv']
        return [(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]

    @lru_cache(maxsize=40000)
    def environment_valid(self,key):
        p=self.env_dir/f'weather_{key}.csv'
        if not p.exists():return False
        try:
            df=pd.read_csv(p)
            required=['humidity','temperature','wind_dir','wind_speed','latitude','longitude','timestamp']
            if not set(required)<=set(df.columns):return False
            df=df.dropna(subset=required)
            when=pd.to_datetime(df.timestamp,utc=True,errors='coerce')
            cutoff=pd.Timestamp(tick_time(key),tz='Asia/Singapore').tz_convert('UTC')
            return bool((when<=cutoff).any())
        except (ValueError,OSError,pd.errors.ParserError):return False

    @lru_cache(maxsize=48)
    def frame(self,key):
        signature=fingerprint(self.signature(key))
        path=self.cache/f'{key}_{signature[:16]}.npy'
        if path.exists():return np.load(path)
        radar=self.radar(key)
        env=build_env_data(self.env_dir/f'weather_{key}.csv',verbose=False,
                           height=radar.shape[0],width=radar.shape[1],as_of=tick_time(key))
        if env is None:raise ValueError(f'No causal complete weather observations at {key}')
        frame=np.concatenate([radar[None],env]).astype(np.float32)
        if not np.isfinite(frame).all():raise ValueError(f'Non-finite input at {key}')
        save_array(path,frame)
        return frame


def make_manifests(store,config):
    """Split entire days first; every history/future frame stays in its split."""
    end=datetime.fromisoformat(config.development_end).strftime('%Y%m%d')
    days=sorted({k[:8] for k in store.keys if k[:8]<end})
    if len(days)<10:raise ValueError('At least ten development days are required')
    a,b=int(.75*len(days)),int(.90*len(days))
    assignment={day:name for name,part in [('train',days[:a]),('validation',days[a:b]),('calibration',days[b:])] for day in part}
    if config.test_start:
        test_day=datetime.fromisoformat(config.test_start).strftime('%Y%m%d')
        if test_day<end:raise ValueError('Test start overlaps the development period')
        assignment.update({k[:8]:'test' for k in store.keys if k[:8]>=test_day})
    manifests={name:[] for name in ['train','validation','calibration','test']}
    for i,key in enumerate(store.keys):
        if i%config.stride:continue
        split=assignment.get(key[:8])
        if split is None:continue
        history=[shifted(key,-5*j) for j in range(config.max_history-1,-1,-1)]
        future=[shifted(key,5*j) for j in range(1,config.future+1)]
        window=history+future
        if not all(k in store.key_set and assignment.get(k[:8])==split for k in window):continue
        if not all(store.environment_valid(k) for k in history):continue
        manifests[split].append(key)
    if any(not manifests[s] for s in ['train','validation','calibration']):
        raise ValueError('One development partition has no complete eligible anchors')
    return manifests


def arrival_map(history_radar,future_radar,config):
    """-1 = currently wet/outside crop support, 0..11 = first future hit, 12 = dry."""
    if len(future_radar)!=12:raise ValueError('Require all twelve future observations; missing is not no-rain')
    if config.patch%2!=1:raise ValueError('Neighborhood width must be odd')
    required=int(np.ceil(config.coverage*config.patch**2))
    kernel=np.ones((config.patch,config.patch),dtype=np.int16)
    def wet(grid):
        return ndimage.convolve((grid>config.rain_threshold).astype(np.int16),kernel,mode='constant')>=required
    now=wet(history_radar)
    labels=np.full(now.shape,12,dtype=np.int8)
    for i,grid in enumerate(future_radar):
        labels[(labels==12)&wet(grid)]=i
    labels[now]=-1
    # Even crop: the target is at [crop//2,crop//2], e.g. [32,32].
    left=config.crop//2;right=config.crop-left
    eligible=np.zeros(now.shape,dtype=bool)
    eligible[left:now.shape[0]-right+1,left:now.shape[1]-right+1]=True
    labels[~eligible]=-1
    return labels


class LabelStore:
    def __init__(self,store,config):
        self.store,self.config=store,config
        tag=fingerprint(dict(config=asdict(config),decoder=SOURCE,cleaning=RADAR_CLEANING_VERSION))[:16]
        self.cache=store.root/'data/arrival_cache'/'labels'/tag
        self.cache.mkdir(parents=True,exist_ok=True)

    @lru_cache(maxsize=64)
    def get(self,key):
        keys=[key]+[shifted(key,5*i) for i in range(1,13)]
        signatures=[(self.store.radar_dir/f'{k}.png').stat().st_mtime_ns for k in keys]
        path=self.cache/f'{key}_{fingerprint(signatures)[:12]}.npy'
        if path.exists():return np.load(path)
        labels=arrival_map(self.store.radar(key),[self.store.radar(k) for k in keys[1:]],self.config)
        save_array(path,labels)
        return labels

    def inspect(self,anchors):
        counts=np.zeros(13,dtype=np.int64);usable=[]
        for i,key in enumerate(anchors):
            labels=self.get(key);valid=labels[labels>=0]
            if len(valid):usable.append(key);counts+=np.bincount(valid,minlength=13)
            if (i+1)%1000==0:print(f'Labelled {i+1}/{len(anchors)} anchors',flush=True)
        return usable,counts


def fit_normalization(store,anchors,max_frames=None,history=12):
    # Uniformly chosen unique TRAINING frame times; full resolution statistics.
    keys=sorted({shifted(k,-5*i) for k in anchors for i in range(history)})
    if max_frames is not None and len(keys)>max_frames:
        keys=[keys[i] for i in np.linspace(0,len(keys)-1,max_frames,dtype=int)]
    total=np.zeros(4);squared=np.zeros(4);n=0
    for key in keys:
        a=store.frame(key)[1:5].astype(np.float64)
        total+=a.sum((1,2));squared+=(a*a).sum((1,2));n+=a.shape[1]*a.shape[2]
    if not n:raise ValueError('No training normalization frames')
    mean=total/n;std=np.sqrt(np.maximum(squared/n-mean**2,1e-8))
    h,w=store.radar(keys[0]).shape
    return dict(mean=mean.tolist(),std=std.tolist(),distance_scale=float(np.hypot(h-1,w-1)),
                decoder=SOURCE,channels=list(CHANNELS),fit_frame_keys=keys,clip=4.)


def prepare(store,config,artifact_dir,normalization_frames=512):
    """Prepare once, reuse identical manifests for both notebooks."""
    artifact_dir=Path(artifact_dir);artifact_dir.mkdir(parents=True,exist_ok=True)
    metadata=artifact_dir/'prepared.json'
    if metadata.exists():
        saved=json.loads(metadata.read_text())
        if saved['config']!=asdict(config):raise ValueError('Prepared data config differs; choose another artifact directory')
        return saved
    labels=LabelStore(store,config);manifests=make_manifests(store,config);counts={}
    for split,anchors in manifests.items():
        if split=='test':
            counts[split]=None  # Do not inspect final-test labels during preparation.
            continue
        print(f'Preparing {split}: {len(anchors)} complete anchors',flush=True)
        manifests[split],histogram=labels.inspect(anchors)
        counts[split]=histogram.tolist()
    norm=fit_normalization(store,manifests['train'],normalization_frames,config.max_history)
    saved=dict(config=asdict(config),manifests=manifests,class_counts=counts,norm=norm,
               manifest_hash=fingerprint(manifests),
               caveat='Observation timestamps checked; historical publication/ingestion times unavailable.')
    metadata.write_text(json.dumps(saved,indent=2))
    return saved


def fixed_locations(labels,anchors,per_anchor=32,seed=67):
    rng=np.random.default_rng(seed);records=[]
    for key in anchors:
        candidates=np.argwhere(labels.get(key)>=0)
        if not len(candidates):continue
        selected=rng.choice(len(candidates),min(per_anchor,len(candidates)),replace=False)
        records.extend((key,int(y),int(x)) for y,x in candidates[selected])
    return records


def input_crop(store,key,y,x,norm,history=6,channels=tuple(range(7)),crop=64):
    """Historical-only inference input. Does not read labels or future radar."""
    if norm.get('decoder')!=SOURCE:raise ValueError('Input normalization uses a different decoder')
    left=crop//2;right=crop-left
    h,w=store.radar(key).shape
    if not (left<=y<=h-right and left<=x<=w-right):
        raise ValueError('Location lacks a complete centered crop')
    a=np.stack([store.frame(shifted(key,-5*i))[:,y-left:y-left+crop,x-left:x-left+crop]
                for i in range(history-1,-1,-1)]).copy()
    mean=np.asarray(norm['mean'],dtype=np.float32)[None,:,None,None]
    std=np.asarray(norm['std'],dtype=np.float32)[None,:,None,None]
    a[:,1:5]=np.clip((a[:,1:5]-mean)/std,-norm['clip'],norm['clip'])
    a[:,6]=np.clip(a[:,6]/norm['distance_scale'],0,1)
    return torch.from_numpy(a[:,tuple(channels)])


class ArrivalDataset(Dataset):
    def __init__(self,store,labels,anchors,norm,history=6,channels=tuple(range(7)),
                 per_anchor=32,seed=67,records=None,hard_negative_fraction=0.):
        if history not in (3,6,9,12) or history>labels.config.max_history:raise ValueError('Invalid history')
        self.store,self.labels,self.anchors,self.norm=store,labels,anchors,norm
        self.history,self.channels,self.per_anchor,self.seed=history,tuple(channels),per_anchor,seed
        self.records=records;self.epoch=0
        if not 0<=hard_negative_fraction<=1:raise ValueError('Invalid hard-negative fraction')
        self.hard_negative_fraction=hard_negative_fraction
        self.hard_negative_warmup=0
        self.hard_negative_ramp=0
        self.training_step=0
        self.sample_counts={'groups':[0,0,0,0],'hard_branch':0}
        if norm['decoder']!=SOURCE:raise ValueError('Arrival inputs require matching source normalization')

    def set_epoch(self,epoch):self.epoch=epoch
    def set_step(self,step):self.training_step=step

    def effective_hard_fraction(self):
        warmup=getattr(self,'hard_negative_warmup',0)
        ramp=getattr(self,'hard_negative_ramp',0)
        step=getattr(self,'training_step',0)
        if step<warmup:return 0.
        factor=min(1.,(step-warmup)/ramp) if ramp else 1.
        return self.hard_negative_fraction*factor
    def build_group_index(self):
        """Index eligible anchors per group, using training labels only."""
        groups = [[] for _ in range(4)]
        self.hard_anchors=[]
        for key in self.anchors:
            label = self.labels.get(key)
            if self.hard_negative_fraction and self.hard_candidates(key).any():self.hard_anchors.append(key)
            for i, (lo, hi) in enumerate(((0,2),(3,5),(6,11),(12,12))):
                if np.any((label >= lo) & (label <= hi)): groups[i].append(key)
        if any(not group for group in groups):
            raise ValueError('All four training arrival groups need eligible anchors')
        self.group_anchors = groups
        if self.hard_negative_fraction and not self.hard_anchors:raise ValueError('No rain-adjacent no-arrival training locations')
        print('Eligible training anchors by arrival group:', [len(g) for g in groups], flush=True)

    @lru_cache(maxsize=64)
    def hard_candidates(self,key):
        config=self.labels.config
        wet=ndimage.convolve((self.store.radar(key)>config.rain_threshold).astype(np.int16),
            np.ones((config.patch,config.patch),dtype=np.int16),mode='constant')>=int(np.ceil(config.coverage*config.patch**2))
        near=ndimage.distance_transform_edt(~wet)<=5 if wet.any() else np.zeros_like(wet)
        return near & (self.labels.get(key)==12)

    def __len__(self):return len(self.records) if self.records is not None else len(self.anchors)*self.per_anchor
    def __getitem__(self,index):
        if self.records is not None:key,y,x=self.records[index]
        else:
            if not hasattr(self, 'group_anchors'): self.build_group_index()
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            group = int(rng.integers(4))
            # Independent branch RNG preserves baseline general-negative draws
            # whenever the hard branch is not selected.
            branch_rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,9187]))
            hard=group==3 and branch_rng.random()<self.effective_hard_fraction()
            keys = self.hard_anchors if hard else self.group_anchors[group]
            key = keys[rng.integers(len(keys))]; label = self.labels.get(key)
            lo, hi = ((0,2),(3,5),(6,11),(12,12))[group]
            pool = np.argwhere(self.hard_candidates(key) if hard else ((label >= lo) & (label <= hi)))
            y,x=pool[rng.integers(len(pool))]
            self.sample_counts['groups'][group]+=1
            self.sample_counts['hard_branch']+=int(hard)
        return input_crop(self.store,key,y,x,self.norm,self.history,self.channels,self.labels.config.crop),int(self.labels.get(key)[y,x])
