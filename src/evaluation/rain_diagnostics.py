"""Keep original targets intact and stratify errors by target-region size."""
import numpy as np
from scipy import ndimage


def region_diagnostics(target, prediction, threshold=.01):
    rain = np.asarray(target) > threshold
    predicted = np.asarray(prediction) > threshold
    labels, _ = ndimage.label(rain, np.ones((3,3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    result = {}
    for name, lo, hi in [('tiny_1_5',1,5), ('small_6_20',6,20),
                         ('medium_21_100',21,100), ('large_gt100',101,np.inf)]:
        ids = np.flatnonzero((sizes>=lo)&(sizes<=hi))
        ids = ids[ids!=0]
        region = np.isin(labels, ids)
        result[name] = dict(regions=len(ids), detected_regions=len(np.unique(labels[region&predicted])),
                            target_pixels=int(region.sum()), hits=int((region&predicted).sum()))
    result['all_pixels'] = dict(tp=int((rain&predicted).sum()), fp=int((~rain&predicted).sum()),
                                fn=int((rain&~predicted).sum()))
    return result
