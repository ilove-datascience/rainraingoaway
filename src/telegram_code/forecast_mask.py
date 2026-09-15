"""Shared production mask settings used by the bot and offline replay."""
import numpy as np
from scipy import ndimage

RAIN_PROBABILITY_THRESHOLD = 0.55
MIN_COMPONENT_SIZE = 25


def clean_rain_mask(probability):
    labels, _ = ndimage.label(probability >= RAIN_PROBABILITY_THRESHOLD,
                             structure=np.ones((3, 3), dtype=np.uint8))
    keep = np.bincount(labels.ravel()) >= MIN_COMPONENT_SIZE
    keep[0] = False
    return keep[labels]
