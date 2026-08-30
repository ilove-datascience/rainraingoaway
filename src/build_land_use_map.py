"""Rasterize the Master Plan land-use layer onto the radar grid."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from masking import lowerLat, lowerLong, upperLat, upperLong


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "data" / "MasterPlan2025LandUseLayer.geojson"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "land_use_masks.npy"
DEFAULT_METADATA = Path(__file__).resolve().parents[1] / "data" / "land_use_masks.json"
DEFAULT_PREVIEW = Path(__file__).resolve().parents[1] / "data" / "land_use_map_preview.png"
HEIGHT, WIDTH = 120, 217
SUPERSAMPLE = 4


def coordinate_to_pixel(longitude, latitude, width, height):
    x = (longitude - lowerLong) / (upperLong - lowerLong) * (width - 1)
    y = (upperLat - latitude) / (upperLat - lowerLat) * (height - 1)
    return x, y


def polygon_rings(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    return []


def draw_ring(draw, ring, scale, width, height):
    points = []
    for longitude, latitude in ring:
        x, y = coordinate_to_pixel(longitude, latitude, width, height)
        points.append((x * scale, y * scale))
    if len(points) >= 3:
        draw.polygon(points, fill=255)


def rasterize(source_path, output_path, metadata_path, preview_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    features = source.get("features", [])
    classes = sorted(
        {
            feature.get("properties", {}).get("LU_DESC")
            for feature in features
            if feature.get("properties", {}).get("LU_DESC")
        }
    )
    class_index = {name: index for index, name in enumerate(classes)}
    scale = SUPERSAMPLE
    large_size = (WIDTH * scale, HEIGHT * scale)
    masks = np.zeros((len(classes), HEIGHT, WIDTH), dtype=np.float32)

    class_images = [Image.new("L", large_size, 0) for _ in classes]
    class_drawers = [ImageDraw.Draw(image) for image in class_images]

    for feature in features:
        land_use = feature.get("properties", {}).get("LU_DESC")
        geometry = feature.get("geometry")
        if land_use not in class_index or not geometry:
            continue
        drawer = class_drawers[class_index[land_use]]
        for polygon in polygon_rings(geometry):
            draw_ring(drawer, polygon[0], scale, WIDTH, HEIGHT)
            for hole in polygon[1:]:
                draw_ring(drawer, hole, scale, WIDTH, HEIGHT)
                points = []
                for longitude, latitude in hole:
                    x, y = coordinate_to_pixel(longitude, latitude, WIDTH, HEIGHT)
                    points.append((x * scale, y * scale))
                if len(points) >= 3:
                    drawer.polygon(points, fill=0)

    for index, image in enumerate(class_images):
        resized = image.resize((WIDTH, HEIGHT), Image.Resampling.BOX)
        masks[index] = np.asarray(resized, dtype=np.float32) / 255.0

    # Resolve any overlapping source polygons without changing uncovered pixels.
    total = masks.sum(axis=0)
    overlap = total > 1.0
    masks[:, overlap] /= total[overlap]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, masks)
    metadata = {
        "source": str(source_path),
        "shape": list(masks.shape),
        "classes": classes,
        "supersample": SUPERSAMPLE,
        "bounds": {
            "lower_latitude": lowerLat,
            "upper_latitude": upperLat,
            "lower_longitude": lowerLong,
            "upper_longitude": upperLong,
        },
        "coverage_min": float(masks.sum(axis=0).min()),
        "coverage_max": float(masks.sum(axis=0).max()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    dominant = masks.argmax(axis=0)
    uncovered = masks.sum(axis=0) == 0
    dominant = dominant.astype(np.float32)
    dominant[uncovered] = np.nan
    plt.imsave(preview_path, dominant, cmap="tab20", vmin=0, vmax=max(len(classes) - 1, 1))
    plt.close()

    print(f"Saved masks: {output_path} shape={masks.shape}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Classes: {len(classes)}")
    print(f"Covered pixels: {np.count_nonzero(masks.sum(axis=0))}/{HEIGHT * WIDTH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    args = parser.parse_args()
    rasterize(args.source, args.output, args.metadata, args.preview)


if __name__ == "__main__":
    main()
