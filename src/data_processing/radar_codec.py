"""Versioned radar color decoding. Existing checkpoints default to legacy input."""
import math
import numpy as np
from PIL import Image

LEGACY = 'legacy_nearest30_v1'
SOURCE = 'nea_exact33_v1'
SOURCE_URL = 'https://data.gov.sg/datasets/d_418e9ac3414fd927b7405631e0a7bc82/view'
SOURCE_HEX = (
    '00FFFF','00EFEF','00D1D5','00BABF','00979A','00837D','008045','008938',
    '00A235','00B729','00CA11','00DA0D','00F507','00FF00','43FF41','48FF46',
    'FFFF3B','FFFF00','FFF000','FFDC00','FFC600','FFB200','FFA500','FF8A00',
    'FF7200','FF4900','FF1F00','E50000','C10000','B6006A','D200A5','D400AA','FF00FF',
)
SOURCE_CATEGORIES = (('Light',)*11 + ('Light to Moderate',)*5 + ('Moderate',)*4
                     + ('Moderate to Heavy',)*5 + ('Heavy',)*8)


def validate_version(version):
    if version not in (LEGACY, SOURCE):
        raise ValueError(f'Unknown radar decoder version: {version}')
    return version


def decode_png(path, version=LEGACY):
    """Return normalized ordinal intensity. Unknown opaque source colors fail closed.

    Legacy is deliberately unchanged, including its nearest-color approximation.
    Source values are ceil((index+1)/33*100)/100; they are NOT mm/hour.
    """
    validate_version(version)
    if version == LEGACY:
        from .pngtojson import png_to_intensity_grid
        return np.asarray(png_to_intensity_grid(path), dtype=np.float32) / 100.0
    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA'))
    rgb = rgba[:, :, :3].astype(np.int32)
    packed = rgb[:, :, 0]*65536 + rgb[:, :, 1]*256 + rgb[:, :, 2]
    source = np.array([int(c,16) for c in SOURCE_HEX])
    order = np.argsort(source)
    positions = np.minimum(np.searchsorted(source[order], packed), len(source)-1)
    opaque = rgba[:, :, 3] > 0
    unknown = opaque & (source[order][positions] != packed)
    if unknown.any():
        colors = [f'#{c:06X}' for c in np.unique(packed[unknown])[:5]]
        raise ValueError(f'Unknown opaque radar colors ({unknown.sum()} pixels): {colors}')
    values = np.ceil((order[positions]+1)/33*100).astype(np.float32)/100
    return np.where(opaque, values, 0).astype(np.float32)


def source_category(value):
    """Label an observed source rank; never apply this to legacy model outputs."""
    if not math.isfinite(value):
        raise ValueError('Invalid radar intensity')
    if value <= 0:
        return 'No rain'
    levels = np.ceil(np.arange(1,34)/33*100)/100
    index = int(np.argmin(np.abs(levels-value)))
    if abs(float(levels[index])-value) > 1e-5:
        raise ValueError('Observed intensity is not an exact source level')
    return SOURCE_CATEGORIES[index]
