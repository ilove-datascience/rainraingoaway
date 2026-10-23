"""Render the UI draft using synthetic weather, without contacting Telegram or a database."""
import argparse
import ast
import asyncio
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data_processing.radar_codec import source_category
from telegram_code.local_rain import in_coverage, local_rain
from telegram_code import feedback
from telegram_code.feedback_context import _grid_snapshots
from telegram_code.menus import main_menu

METHODS = ROOT / 'src/telegram_code/methods.py'


def renderer(source):
    """Load only plotting functions; no bot/model startup or weather reads."""
    names = {'render_heatmap', '_render_heatmap', 'build_group_map'}
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    env = dict(__file__=str(METHODS), plt=plt, np=np, Path=Path, BytesIO=BytesIO,
               PowerNorm=PowerNorm, _plot_lock=threading.Lock(), in_coverage=in_coverage,
               local_rain=local_rain, source_category=source_category,
               intensity_label=lambda value: 'Light' if value < 1/3 else 'Moderate' if value < 2/3 else 'Heavy')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(METHODS), 'exec'), env)
    return env


def sample():
    rows = [dict(label=name, latitude=lat, longitude=lon, location_id=index)
            for index, (name, lat, lon) in enumerate([
                ('Main', 1.30, 103.80), ('Office', 1.35, 103.90),
                ('School', 1.40, 103.82), ('Gym', 1.34, 103.70),
                ('Parents', 1.30, 103.94), ('Station', 1.42, 103.75)])]
    grid = np.zeros((240, 428), dtype=np.float32)
    yy, xx = np.indices(grid.shape)
    for index, amplitude in ((1, .53), (2, .21), (4, .38)):
        _, (x, y), _ = local_rain(grid, rows[index]['latitude'], rows[index]['longitude'])
        grid += amplitude * np.exp(-((xx-x)/13)**2 - ((yy-y)/7)**2)
    grid[grid < .025] = 0
    levels = np.r_[0., np.ceil(np.arange(1, 34) / 33 * 100) / 100]
    grid = levels[np.abs(grid[..., None] - levels).argmin(axis=-1)].astype(np.float32)
    return rows, grid, datetime(2026, 10, 20, 12)


def panel(ax, title, subtitle):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(0, .99, title, va='top', weight='bold', fontsize=15, color='#17304c')
    ax.text(0, .95, subtitle, va='top', fontsize=10, color='#60748b')


def button(ax, x, y, width, text, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), width, .048,
        boxstyle='round,pad=0.006,rounding_size=0.008',
        facecolor='#e0eff9' if accent else '#edf2f7', edgecolor='#d6e1eb', linewidth=.7))
    ax.text(x + width/2, y + .024, text, ha='center', va='center',
            fontsize=9, color='#214967')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/ui-preview-20261020')
    parser.add_argument('--baseline-ref', default='86bb9a1', help='Git revision used for the before comparison')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, grid, observed = sample()
    new = renderer(METHODS.read_text(encoding='utf-8'))
    original = subprocess.check_output(['git', 'show', f'{args.baseline_ref}:src/telegram_code/methods.py'],
                                       cwd=ROOT, text=True, encoding='utf-8')
    old = renderer(original)
    for name, functions in [('before', old), ('draft', new)]:
        original_savefig = Figure.savefig
        def save_preview(figure, *positional, **kwargs):
            if name == 'draft':
                # Keep marker outlines and numbers as vector paths when zooming.
                with matplotlib.rc_context({'svg.fonttype': 'path'}):
                    original_savefig(figure, args.output / 'draft-radar.svg', format='svg',
                                     facecolor=figure.get_facecolor())
            return original_savefig(figure, *positional, **kwargs)
        with patch.object(Figure, 'savefig', save_preview):
            image, caption = functions['build_group_map'](rows, grid, observed, actual=True)
        try:
            (args.output / f'{name}-radar.png').write_bytes(image.getvalue())
        finally:
            image.close()
        (args.output / f'{name}-caption.txt').write_text(caption, encoding='utf-8')

    # Create the real keyboard in memory, replacing only its database write and clock.
    snapshots = _grid_snapshots(rows, 'radar', observed, observed, radar=grid)
    store = lambda chat, facts, now: [(f'{index:016d}', fact) for index, fact in enumerate(facts)]
    with patch.object(feedback, '_store_snapshots', side_effect=store), \
         patch.object(feedback, '_utc_now', return_value=datetime(2026, 10, 20, 4, tzinfo=timezone.utc)):
        keyboard = asyncio.run(feedback.prepare_feedback(123, snapshots))
    caption = (args.output / 'draft-caption.txt').read_text(encoding='utf-8')

    fig = plt.figure(figsize=(18, 10.7), dpi=150, facecolor='#f4f7fb')
    fig.text(.035, .965, 'UI PREVIEW', fontsize=13, color='#11675e', weight='bold')
    fig.text(.035, .917, 'Clearer maps. Easier feedback.', fontsize=26, color='#17304c', weight='bold')
    fig.text(.965, .953, 'LOCAL DRAFT  /  NOT PUSHED', ha='right', fontsize=11, color='#60748b')
    fig.text(.035, .884, 'Illustrative sample weather and locations. Telegram controls are shown schematically.',
             fontsize=11, color='#60748b')

    left = fig.add_axes([.025, .035, .435, .82])
    left.imshow(plt.imread(args.output / 'draft-radar.png'))
    left.axis('off')
    middle = fig.add_axes([.485, .04, .29, .80])
    panel(middle, 'Message & feedback', 'Numbers match the map and saved places.')
    y = .875
    for line in caption.splitlines():
        if line:
            # Preserve all caption content, wrapping only the radar-scale detail.
            parts = line.split(' — ')
            middle.text(0, y, parts[0], fontsize=10, color='#17304c', va='top')
            y -= .037
            if len(parts) > 1:
                middle.text(.035, y, '— ' + parts[1], fontsize=9, color='#60748b', va='top')
                y -= .033
        else:
            y -= .013
    middle.text(0, y-.005, 'Were you there? Report at the shown SGT time.\nReports close 30 minutes after that time.',
                fontsize=9, color='#60748b', va='top', linespacing=1.5)
    y -= .115
    for row in keyboard.inline_keyboard:
        for column, item in enumerate(row):
            button(middle, column * .505, y, .475, item.text, accent=True)
        y -= .061

    right = fig.add_axes([.80, .04, .18, .80])
    panel(right, 'Menu', 'Weather first. Place editing below.')
    y = .82
    for row in main_menu().keyboard:
        for column, item in enumerate(row):
            button(right, column * .51, y, .46, item.text, accent=y > .80)
        y -= .065
    right.text(0, .47, 'Same actions and commands.\nSame report time windows.\nNo forecast-model changes.',
               fontsize=10, color='#60748b', linespacing=1.6, va='top')
    fig.savefig(args.output / 'ui-preview.png', facecolor=fig.get_facecolor())
    plt.close(fig)

    comparison = plt.figure(figsize=(16, 10), dpi=120, facecolor='#f4f7fb')
    for index, (name, label) in enumerate([('before', 'Current'), ('draft', 'Proposed')]):
        ax = comparison.add_axes([.02 + index*.5, .04, .46, .86])
        ax.imshow(plt.imread(args.output / f'{name}-radar.png'))
        ax.axis('off')
        ax.set_title(label, fontsize=20, color='#17304c', pad=16)
    comparison.text(.5, .025, 'Illustrative sample weather and locations — local preview only',
                    ha='center', fontsize=11, color='#60748b')
    comparison.savefig(args.output / 'before-after.png', facecolor=comparison.get_facecolor())
    plt.close(comparison)
    print(args.output / 'ui-preview.png')


if __name__ == '__main__':
    main()
