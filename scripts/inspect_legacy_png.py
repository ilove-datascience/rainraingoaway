"""Inspect the legacy 0–100 colour decoder; not the source-aware training codec.

Run from the repository root:
    python scripts/inspect_legacy_png.py path/to/image.png
Outputs go to ignored reports/png-inspection/ unless --output-dir is supplied.
"""

import argparse
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data_processing.pngtojson import (
    png_to_intensity_grid,
    png_to_xy_intensity,
    points_to_intensity_grid,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='PNG to inspect with the legacy decoder')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/png-inspection')
    args = parser.parse_args()
    path = args.image
    if not path.is_file():
        parser.error(f'PNG not found: {path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use the simpler grid-based intensity extraction
    grid = png_to_intensity_grid(path)

    # Print a couple of sample points
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    print(f"Image: {w}x{h}")
    print("intensity at (0,0):", grid[0][0])
    if h > 100 and w > 200:
        print("intensity at (200,100):", grid[100][200])

    # Also write a JSON of xy-intensity points (non-zero only)
    points = png_to_xy_intensity(path, include_zero=False)
    out_path = args.output_dir / "points.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(points, f)

    print(f"Wrote {out_path} (points: {len(points)})")

    # Reconstruct the simple grid representation from the points file.
    recon_grid = points_to_intensity_grid(points, width=w, height=h)
    grid_out_path = args.output_dir / "grid.json"
    with grid_out_path.open("w", encoding="utf-8") as f:
        json.dump(recon_grid, f)

    print(f"Wrote {grid_out_path} (size: {w}x{h})")


if __name__ == "__main__":
    main()
