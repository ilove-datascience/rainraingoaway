"""Test runner for PNG intensity extraction and grid reconstruction.

Place the PNG you attached at the repository root as `test_input.png`, or pass a path as the first argument.
Install runtime dependencies if needed:

pip install pillow

Run:

python test_pngtojson.py [path/to/image.png]
"""

from pathlib import Path
import sys
import json

from PIL import Image

from data.pngtojson import (
    png_to_intensity_grid,
    png_to_xy_intensity,
    points_to_intensity_grid,
)


def load_png_as_dict(path: Path):
    img = Image.open(path).convert("RGBA")
    width, height = img.size
    pixels = list(img.getdata())
    data = []
    for r, g, b, a in pixels:
        data.extend([r, g, b, a])

    return {"width": width, "height": height, "data": data}


def main():
    path = Path("test_input.png")
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"PNG not found: {path}. Save the attached PNG as {path} or pass its path as an argument.")
        sys.exit(2)

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
    out_path = Path("test_output_points.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(points, f)

    print(f"Wrote {out_path} (points: {len(points)})")

    # Reconstruct the simple grid representation from the points file.
    recon_grid = points_to_intensity_grid(points, width=w, height=h)
    grid_out_path = Path("test_output_grid.json")
    with grid_out_path.open("w", encoding="utf-8") as f:
        json.dump(recon_grid, f)

    print(f"Wrote {grid_out_path} (size: {w}x{h})")


if __name__ == "__main__":
    main()
