"""Offline synthetic render/encoding benchmark; no Telegram or database traffic."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

from PIL import Image
from matplotlib.figure import Figure
import numpy as np
from preview_weather_ui import METHODS, ROOT, renderer, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radar', type=Path, help='Also benchmark a saved source radar PNG')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.check_output(['git', 'show', '85a9159:src/telegram_code/methods.py'],
                                       cwd=ROOT, encoding='utf-8')
    rows, sparse, observed = sample()
    yy, xx = np.indices(sparse.shape)
    dense = np.clip(.45 + .3*np.sin(xx/19)*np.cos(yy/13), 0, 1)
    levels = np.r_[0., np.ceil(np.arange(1, 34) / 33 * 100) / 100]
    dense = levels[np.abs(dense[..., None] - levels).argmin(axis=-1)]
    results = []
    cases = [('sparse', sparse), ('widespread', dense)]
    if args.radar:
        from data_processing.radar_codec import decode_png, SOURCE
        from data_processing.data_loading import remove_small_echoes
        cases.append(('saved-radar', np.flipud(remove_small_echoes(decode_png(args.radar, SOURCE)))))
    for weather, grid in cases:
        for actual in (True, False):
            for version, source in [('before', baseline), ('after', METHODS.read_text(encoding='utf-8'))]:
                draw = renderer(source)['build_group_map']
                durations = []
                for repeat in range(args.repeats + 1):
                    start = time.perf_counter()
                    image, _ = draw(rows, grid, observed, actual=actual)
                    payload = image.getvalue()
                    image.close()
                    elapsed = time.perf_counter()-start
                    if repeat:
                        durations.append(elapsed)
                stem = f'{weather}-{"actual" if actual else "forecast"}-{version}'
                extension = 'jpg' if payload.startswith(b'\xff\xd8') else 'png'
                (args.output / f'{stem}.{extension}').write_bytes(payload)
                result = dict(case=stem, bytes=len(payload), render_median_s=statistics.median(durations),
                              render_max_s=max(durations))
                # Explicitly an upload budget estimate, excluding DB, queue and server processing.
                result['estimated_render_upload_1Mbps_plus_500ms_s'] = result['render_max_s'] + len(payload)*8/1_000_000 + .5
                if version == 'after':
                    save = Figure.savefig
                    def lossless(fig, *pos, **kwargs):
                        kwargs.pop('pil_kwargs', None)
                        kwargs['format'] = 'png'
                        return save(fig, *pos, **kwargs)
                    with patch.object(Figure, 'savefig', lossless):
                        reference, _ = draw(rows, grid, observed, actual=actual)
                    a = np.asarray(Image.open(reference).convert('RGB'), dtype=float)
                    b = np.asarray(Image.open(BytesIO(payload)).convert('RGB'), dtype=float)
                    result['jpeg_psnr_db_same_resolution'] = float(10*np.log10(255**2/np.mean((a-b)**2)))
                    reference.close()
                results.append(result)
    (args.output / 'benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
