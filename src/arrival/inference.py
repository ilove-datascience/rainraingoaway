"""Arrival inference from historical frames only; no future-label dependency."""
import numpy as np
import torch
from arrival.data import input_crop,CLASSES
from arrival.training import calibrated_probs


def predict_location(model,store,key,y,x,norm,config,calibrator=None,channels=tuple(range(7))):
    tensor=input_crop(store,key,y,x,norm,model.history,channels,config.crop)
    half=config.patch//2
    patch=store.radar(key)[y-half:y+half+1,x-half:x+half+1]
    if (patch>config.rain_threshold).sum()>=np.ceil(config.coverage*config.patch**2):
        return dict(status='already_raining',probabilities=None)
    model.eval()
    with torch.inference_mode():
        lp=model.log_probs(tensor.unsqueeze(0).to(next(model.parameters()).device)).cpu()
    p=calibrated_probs(lp,calibrator)[0]
    return dict(status='currently_dry',most_likely=CLASSES[int(p.argmax())],probabilities=p.tolist(),
                within_minutes={minutes:float(p[:minutes//5].sum()) for minutes in (15,30,60)},
                calibrated=calibrator is not None)
