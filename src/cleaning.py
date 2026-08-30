"""Standalone connected-component experiment for radar echo cleaning.

This module deliberately does not modify the training data loader or cache. Run it
against real radar PNGs to inspect what the conservative filter would remove before
integrating the behavior into data_processing.data_loading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from data_processing.pngtojson import png_to_intensity_grid


def remove_small_echoes(
    radar: np.ndarray,
    rain_threshold: float = 0.01,
    min_pixels: int = 4,
    strong_threshold: float = 0.10,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove tiny weak 8-connected radar components without changing intensities.

    A component is retained when it has at least ``min_pixels`` pixels or contains
    a return at least as strong as ``strong_threshold``. The returned statistics
    describe the proposed change and are useful for threshold tuning.
    """
    radar = np.asarray(radar, dtype=np.float32)
    if radar.ndim != 2:
        raise ValueError(f"radar must be 2D, got shape {radar.shape}")
    if min_pixels < 1:
        raise ValueError("min_pixels must be at least 1")

    rain_mask = radar > rain_threshold
    structure = np.ones((3, 3), dtype=np.uint8)
    component_labels, component_count = label(rain_mask, structure=structure)
    keep_mask = np.zeros_like(rain_mask, dtype=bool)
    removed_components = 0
    removed_pixels = 0

    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        size = int(component.sum())
        max_intensity = float(radar[component].max())
        keep_component = size >= min_pixels or max_intensity >= strong_threshold

        if keep_component:
            keep_mask |= component
        else:
            removed_components += 1
            removed_pixels += size

    cleaned = radar.copy()
    cleaned[~keep_mask] = 0.0
    stats = {
        "components": int(component_count),
        "removed_components": removed_components,
        "rain_pixels_before": int(rain_mask.sum()),
        "removed_pixels": removed_pixels,
        "rain_pixels_after": int((cleaned > rain_threshold).sum()),
    }
    return cleaned, stats


def load_radar_png(path: Path) -> np.ndarray:
    """Load one radar PNG as the normalized 0..1 intensity grid."""
    grid = np.asarray(png_to_intensity_grid(path), dtype=np.float32)
    return grid / 100.0


def find_pngs(folder: Path, sample_count: int, seed: int) -> list[Path]:
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {folder}")

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        len(pngs), size=min(sample_count, len(pngs)), replace=False
    )
    return [pngs[index] for index in selected_indices]


def plot_comparison(
    radar: np.ndarray,
    cleaned: np.ndarray,
    path: Path,
    stats: dict[str, int],
) -> None:
    """Plot one original/cleaned pair with a shared intensity scale."""
    vmax = max(float(radar.max()), float(cleaned.max()), 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (radar, cleaned),
        ("Original radar", "After small-echo filtering"),
    ):
        image_plot = axis.imshow(image, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(image_plot, ax=axes, label="Normalized radar intensity")
    fig.suptitle(
        f"{path.name} | removed {stats['removed_pixels']} pixels "
        f"in {stats['removed_components']} components"
    )
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path("data/70km/png"),
        help="Folder containing radar PNGs",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rain-threshold", type=float, default=0.01)
    parser.add_argument("--min-pixels", type=int, default=4)
    parser.add_argument("--strong-threshold", type=float, default=0.10)
    args = parser.parse_args()

    selected_paths = find_pngs(args.folder, args.samples, args.seed)
    total_stats = {
        "components": 0,
        "removed_components": 0,
        "rain_pixels_before": 0,
        "removed_pixels": 0,
        "rain_pixels_after": 0,
    }

    for path in selected_paths:
        radar = load_radar_png(path)
        cleaned, stats = remove_small_echoes(
            radar,
            rain_threshold=args.rain_threshold,
            min_pixels=args.min_pixels,
            strong_threshold=args.strong_threshold,
        )
        for key in total_stats:
            total_stats[key] += stats[key]

        print(
            f"{path.name}: "
            f"components={stats['components']}, "
            f"removed={stats['removed_components']} components / "
            f"{stats['removed_pixels']} pixels, "
            f"rain pixels {stats['rain_pixels_before']} -> {stats['rain_pixels_after']}"
        )
        plot_comparison(radar, cleaned, path, stats)

    print("\nAggregate result")
    print(f"Sampled frames: {len(selected_paths)}")
    print(f"Components: {total_stats['components']}")
    print(f"Removed components: {total_stats['removed_components']}")
    print(f"Removed pixels: {total_stats['removed_pixels']}")
    print(
        f"Removed rain pixels: "
        f"{total_stats['removed_pixels'] / max(total_stats['rain_pixels_before'], 1):.2%}"
    )


if __name__ == "__main__":
    main()
