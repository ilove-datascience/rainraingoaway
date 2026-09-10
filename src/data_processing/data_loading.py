SEED = 67
import json
import os
import threading
import torch 
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np 
from pathlib import Path
import sys
from scipy.ndimage import label

RADAR_CLEANING_VERSION = "connected_components_v1_min4_strong010"
RADAR_RAIN_THRESHOLD = 0.01
RADAR_MIN_PIXELS = 4
RADAR_STRONG_THRESHOLD = 0.10
ENV_TICK_TOLERANCE_MINUTES = 15

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data_processing.pngtojson import points_to_intensity_grid, png_to_xy_intensity
    from masking import lat_long_to_pixel
else:
    from .pngtojson import points_to_intensity_grid, png_to_xy_intensity
    from masking import lat_long_to_pixel


def remove_small_echoes(
    radar,
    rain_threshold=RADAR_RAIN_THRESHOLD,
    min_pixels=RADAR_MIN_PIXELS,
    strong_threshold=RADAR_STRONG_THRESHOLD,
):
    """Remove tiny weak 8-connected radar components while preserving intensities."""
    radar = np.asarray(radar, dtype=np.float32)
    if radar.ndim != 2:
        raise ValueError(f"radar must be 2D, got shape {radar.shape}")
    if min_pixels < 1:
        raise ValueError("min_pixels must be at least 1")

    rain_mask = radar > rain_threshold
    structure = np.ones((3, 3), dtype=np.uint8)
    component_labels, component_count = label(rain_mask, structure=structure)
    keep_mask = np.zeros_like(rain_mask, dtype=bool)

    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        if component.sum() >= min_pixels or radar[component].max() >= strong_threshold:
            keep_mask |= component

    cleaned = radar.copy()
    cleaned[~keep_mask] = 0.0
    return cleaned

def load_data(folder_path, total):
    # Load the data from the specified folder
    data = dict()
    count = 0 
    for file_name in os.listdir(folder_path):
        
        if file_name.endswith('.png'):
            
            if total is not None and count >= total:
                break
            
            intensity_points = png_to_xy_intensity(os.path.join(folder_path, file_name),include_zero=True)
            #intensity_points = png_to_xy_binary(os.path.join(folder_path, file_name),include_zero=True)
            intensity_grid = points_to_intensity_grid(intensity_points )
            intensity_df = pd.DataFrame(intensity_grid)
            intensity_df = pd.DataFrame(remove_small_echoes(intensity_df.values / 100.0))
            data[file_name] = intensity_df
            count += 1
    print(f"Loaded {count} images.")
    
    return data

def load_specific_data(file_names:list, folder_path):
    data = list()
    count = 0 
    existing_files = set(os.listdir(folder_path))
    for file_name in sorted(file_names):

        png_name = f"{file_name}.png"
        if png_name in existing_files:

            intensity_points = png_to_xy_intensity(os.path.join(folder_path, png_name),include_zero=True)
            #intensity_points = png_to_xy_binary(os.path.join(folder_path, file_name),include_zero=True)
            intensity_grid = points_to_intensity_grid(intensity_points )
            intensity_df = pd.DataFrame(intensity_grid)
            intensity_df = pd.DataFrame(remove_small_echoes(intensity_df.values / 100.0))
            data.append(intensity_df)
            count += 1
    print(f"Loaded {count} images.")
    
    return data

#prev_key = first_key
def create_samples(data, list_length=10, min_list_length = 7, num_target_steps=1):
    """Group consecutive 5-minute frames into (inputs, targets) samples.

    `num_target_steps` controls how many trailing frames are split off as the
    target instead of just one (multi-step/direct forecasting). Input length
    is unaffected by this — only the target grows. With num_target_steps=1
    this reproduces the original single-step behaviour exactly.
    """
    if list_length < 1 or num_target_steps < 1:
        raise ValueError("list_length and num_target_steps must be positive")

    group_size = list_length + num_target_steps
    keys = sorted(data.keys())
    if not keys:
        return [], []

    runs: list[list[torch.Tensor]] = []
    current_run: list[torch.Tensor] = []
    previous_time = None

    for key in keys:
        key_time = datetime.strptime(str(key).removesuffix(".png"), "%Y%m%d%H%M")
        if previous_time is not None and key_time - previous_time != timedelta(minutes=5):
            if current_run:
                runs.append(current_run)
            current_run = []

        current_run.append(torch.from_numpy(np.asarray(data[key])).float())
        previous_time = key_time

    if current_run:
        runs.append(current_run)

    inputs = []
    targets = []
    for run in runs:
        for start in range(0, len(run) - group_size + 1, group_size):
            group = run[start:start + group_size]
            inputs.append(group[:list_length])
            if num_target_steps == 1:
                targets.append(group[-1])
            else:
                targets.append(group[-num_target_steps:])

    return inputs, targets
            
  
def _get_cache_paths(cache_dir, radar_name):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{radar_name}.npy"
    metadata_file = cache_dir / f"{radar_name}.json"
    return cache_dir, cache_file, metadata_file


def _read_cache(cache_file, metadata_file, radar_path, env_path):
    if not cache_file.exists() or not metadata_file.exists():
        return None

    try:
        metadata = json.loads(metadata_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if metadata.get("radar_cleaning_version") != RADAR_CLEANING_VERSION:
        return None

    try:
        return np.load(cache_file)
    except Exception:
        return None


def _write_cache(cache_file, metadata_file, radar_path, env_path, array):
    cache_file = Path(cache_file)
    metadata_file = Path(metadata_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, array)

    metadata = {
        "radar_mtime": os.path.getmtime(radar_path),
        "env_mtime": os.path.getmtime(env_path),
        "radar_cleaning_version": RADAR_CLEANING_VERSION,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))


def _load_multimodal_frame(task):
    file_name, radar_dir, env_dir, verbose, log_lock, use_cache, cache_dir = task
    radar_name = file_name.replace('.png', '')
    env_file_name = f"weather_{radar_name}.csv"
    env_path = os.path.join(env_dir, env_file_name)

    if not os.path.exists(env_path):
        return None, 'missing_env', radar_name, env_file_name

    radar_path = os.path.join(radar_dir, file_name)

    try:
        if use_cache:
            cache_dir_path, cache_file, metadata_file = _get_cache_paths(cache_dir, radar_name)
            cached_array = _read_cache(cache_file, metadata_file, radar_path, env_path)
            if cached_array is not None:
                if verbose:
                    with log_lock:
                        print(f"loaded {file_name} from cache with shape {cached_array.shape}")
                return cached_array.astype(np.float32), 'ok', radar_name, env_file_name

        intensity_points = png_to_xy_intensity(radar_path, include_zero=True)
        intensity_grid = points_to_intensity_grid(intensity_points)
        radar_grid = remove_small_echoes(np.asarray(intensity_grid, dtype=np.float32) / 100.0)

        env_stack = build_env_data(
            env_path,
            verbose=False,
            height=radar_grid.shape[0],
            width=radar_grid.shape[1],
        )
        if env_stack is None:
            return None, 'incomplete_env', radar_name, env_file_name

        radar_stack = radar_grid[np.newaxis, :, :]  # (1, H, W)
        combined_stack = np.concatenate([radar_stack, env_stack], axis=0).astype(np.float32)

        if use_cache:
            _write_cache(
                cache_dir_path / f"{radar_name}.npy",
                cache_dir_path / f"{radar_name}.json",
                radar_path,
                env_path,
                combined_stack,
            )

        if verbose:
            with log_lock:
                print(f"loaded {file_name} with shape {combined_stack.shape}")
        return combined_stack, 'ok', radar_name, env_file_name
    except Exception as exc:
        if verbose:
            with log_lock:
                print(f"Skipping {radar_name}: failed to load frame ({exc})")
        return None, 'error', radar_name, env_file_name


def _find_env_csv(tick, folder_path_env, tolerance_minutes=ENV_TICK_TOLERANCE_MINUTES):
    """Return a weather CSV path usable for `tick`, trying nearby ticks if needed.

    Radar and weather are scraped on independent cadences, so their timestamps
    can drift a few minutes apart. Search outward in 5-minute steps before
    giving up.
    """
    exact_path = os.path.join(folder_path_env, f"weather_{tick}.csv")
    if os.path.exists(exact_path):
        return exact_path

    tick_dt = datetime.strptime(str(tick), "%Y%m%d%H%M")
    for offset in range(5, tolerance_minutes + 1, 5):
        for candidate_dt in (tick_dt - timedelta(minutes=offset), tick_dt + timedelta(minutes=offset)):
            candidate_path = os.path.join(
                folder_path_env, f"weather_{candidate_dt.strftime('%Y%m%d%H%M')}.csv"
            )
            if os.path.exists(candidate_path):
                return candidate_path

    return None


def build_and_cache_frame(tick, img_name="70km", folder_path_radar=None,
                          folder_path_env=None, cache_dir=None, verbose=True):
    """Build the 7-channel multimodal frame for one tick and write it to cache.

    Returns the [7, H, W] float32 array on success, or None if the radar PNG or
    the matching weather CSV is missing/incomplete (caller can retry later).
    Never raises for missing env — designed to be safe to call from the scraper.
    """
    base = Path(__file__).resolve().parents[2]
    if folder_path_radar is None:
        folder_path_radar = str(base / "data" / img_name / "png")
    if folder_path_env is None:
        folder_path_env = str(base / "data" / "environment")
    if cache_dir is None:
        cache_dir = os.path.join(
            Path(folder_path_radar).resolve().parents[1], "data", "multimodal_cache"
        )

    radar_name = str(tick)
    radar_path = os.path.join(folder_path_radar, f"{radar_name}.png")

    if not os.path.exists(radar_path):
        if verbose:
            print(f"cache skip {radar_name}: radar png missing")
        return None

    env_path = _find_env_csv(tick, folder_path_env)
    if env_path is None:
        if verbose:
            print(
                f"cache skip {radar_name}: no weather CSV within "
                f"{ENV_TICK_TOLERANCE_MINUTES} min of weather_{radar_name}.csv"
            )
        return None

    # Already cached? Return the cached array.
    cache_dir_path, cache_file, metadata_file = _get_cache_paths(cache_dir, radar_name)
    cached = _read_cache(cache_file, metadata_file, radar_path, env_path)
    if cached is not None:
        return cached.astype(np.float32)

    try:
        intensity_points = png_to_xy_intensity(radar_path, include_zero=True)
        intensity_grid = points_to_intensity_grid(intensity_points)
        radar_grid = remove_small_echoes(np.asarray(intensity_grid, dtype=np.float32) / 100.0)

        env_stack = build_env_data(
            env_path, verbose=False,
            height=radar_grid.shape[0], width=radar_grid.shape[1],
        )
        if env_stack is None:
            if verbose:
                print(f"cache skip {radar_name}: env data incomplete")
            return None

        combined = np.concatenate(
            [radar_grid[np.newaxis, :, :], env_stack], axis=0
        ).astype(np.float32)

        _write_cache(
            cache_dir_path / f"{radar_name}.npy",
            cache_dir_path / f"{radar_name}.json",
            radar_path, env_path, combined,
        )
        if verbose:
            print(f"cached multimodal frame {radar_name} shape {combined.shape}")
        return combined
    except Exception as exc:
        if verbose:
            print(f"cache failed for {radar_name}: {exc}")
        return None


def load_data_multimodal(folder_path_radar, folder_path_env, total=None, verbose=True, num_workers=None, use_cache=True, cache_dir=None):
    """Load multimodal radar frames and environmental channels in parallel when possible, with optional disk caching."""
    data = dict()
    skipped_missing_env = 0
    skipped_incomplete_env = 0
    skipped_error = 0

    radar_files = sorted(
        file_name for file_name in os.listdir(folder_path_radar) if file_name.endswith('.png')
    )
    if total is not None:
        radar_files = radar_files[:total]

    if num_workers is None:
        num_workers = min(8, max(1, os.cpu_count() or 1))
    num_workers = max(1, int(num_workers))

    if cache_dir is None:
        cache_dir = os.path.join(Path(folder_path_radar).resolve().parents[1], "data", "multimodal_cache")

    log_lock = threading.Lock()
    tasks = [
        (file_name, folder_path_radar, folder_path_env, verbose, log_lock, use_cache, cache_dir)
        for file_name in radar_files
    ]

    if num_workers == 1:
        results = [_load_multimodal_frame(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_load_multimodal_frame, tasks))

    for combined_stack, status, radar_name, env_file_name in results:
        if status == 'ok':
            data[f"{radar_name}.png"] = combined_stack
        elif status == 'missing_env':
            skipped_missing_env += 1
            if verbose:
                print(f"Skipping {radar_name}: missing env file {env_file_name}")
        elif status == 'incomplete_env':
            skipped_incomplete_env += 1
            if verbose:
                print(f"Skipping {radar_name}: env data incomplete")
        else:
            skipped_error += 1

    skipped_total = skipped_missing_env + skipped_incomplete_env + skipped_error
    print(
        f"Loaded {len(data)} images. Skipped {skipped_total} frames "
        f"({skipped_missing_env} missing env files, {skipped_incomplete_env} incomplete env files, {skipped_error} load errors)."
    )

    return data

def build_env_data(env_path, verbose=True, height=120, width=217):
    data_cols = ["humidity", "temperature", "wind_dir", "wind_speed"]
    loc_cols = ["longitude", "latitude"]
    env_df = pd.read_csv(env_path)
    required_cols = loc_cols + data_cols

    missing_cols = [column for column in required_cols if column not in env_df.columns]
    if missing_cols:
        if verbose:
            print(f"Skipping {env_path}: missing columns {missing_cols}")
        return None

    env_df = env_df.drop(columns=[column for column in ["timestamp", "station_name"] if column in env_df.columns])
    env_df = env_df.dropna(subset=required_cols)

    if env_df.empty:
        if verbose:
            print(f"Sk  ipping {env_path}: no complete env rows after dropping missing values")
        return None

    station_coords = []
    station_values = {
        "temperature": [],
        "humidity": [],
        "wind_u": [],
        "wind_v": [],
    }

    for _, row in env_df.iterrows():
        pixelx, pixely = lat_long_to_pixel(
            long=row[loc_cols[0]],
            lat=row[loc_cols[1]],
            width=width,
            height=height,
        )
        station_coords.append((pixelx, pixely))

        temperature = float(row["temperature"])
        humidity = float(row["humidity"])
        wind_speed = float(row["wind_speed"])
        wind_dir = float(row["wind_dir"])
        wind_dir_rad = np.deg2rad(wind_dir)

        # Convert wind direction to U/V components using the meteorological convention.
        wind_u = -wind_speed * np.sin(wind_dir_rad)
        wind_v = -wind_speed * np.cos(wind_dir_rad)

        station_values["temperature"].append(temperature)
        station_values["humidity"].append(humidity)
        station_values["wind_u"].append(wind_u)
        station_values["wind_v"].append(wind_v)

    if not station_coords:
        return None

    station_coords = np.asarray(station_coords, dtype=np.float32)
    station_x = station_coords[:, 0]
    station_y = station_coords[:, 1]

    temperature_grid = np.zeros((height, width), dtype=np.float32)
    humidity_grid = np.zeros((height, width), dtype=np.float32)
    wind_u_grid = np.zeros((height, width), dtype=np.float32)
    wind_v_grid = np.zeros((height, width), dtype=np.float32)
    station_mask = np.zeros((height, width), dtype=np.float32)
    distance_grid = np.full((height, width), np.inf, dtype=np.float32)

    for idx, (x, y) in enumerate(station_coords):
        temperature_grid[int(y), int(x)] = station_values["temperature"][idx]
        humidity_grid[int(y), int(x)] = station_values["humidity"][idx]
        wind_u_grid[int(y), int(x)] = station_values["wind_u"][idx]
        wind_v_grid[int(y), int(x)] = station_values["wind_v"][idx]
        station_mask[int(y), int(x)] = 1.0
        distance_grid[int(y), int(x)] = 0.0

    for y in range(height):
        for x in range(width):
            if station_mask[y, x] > 0.0:
                continue

            distances = np.sqrt((station_x - x) ** 2 + (station_y - y) ** 2)
            if np.all(np.isinf(distances)):
                continue

            nearest_distance = float(np.min(distances))
            distance_grid[y, x] = nearest_distance

            weights = 1.0 / (distances**2 + 1e-8)
            weights = weights / weights.sum()

            temperature_grid[y, x] = float(np.sum(np.asarray(station_values["temperature"], dtype=np.float32) * weights))
            humidity_grid[y, x] = float(np.sum(np.asarray(station_values["humidity"], dtype=np.float32) * weights))
            wind_u_grid[y, x] = float(np.sum(np.asarray(station_values["wind_u"], dtype=np.float32) * weights))
            wind_v_grid[y, x] = float(np.sum(np.asarray(station_values["wind_v"], dtype=np.float32) * weights))

    env_stack = np.stack(
        [
            temperature_grid,
            humidity_grid,
            wind_u_grid,
            wind_v_grid,
            station_mask,
            distance_grid,
        ],
        axis=0,
    )
    return env_stack.astype(np.float32)

# testing multimodal loading 
if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parents[2]
    radar_folder = workspace_root / "data" / "70km" / "png"
    env_folder = workspace_root / "data" / "environment"

    holdout_data = load_data_multimodal(str(radar_folder), str(env_folder), verbose=False)
    print(f"Main loaded {len(holdout_data)} multimodal holdout frames")
    if holdout_data:
        first_key = next(iter(holdout_data))
        expected_shape = holdout_data[first_key].shape
        print(f"Expected shape: {expected_shape}")

        mismatched_keys = []
        for key, value in holdout_data.items():
            if value.shape != expected_shape:
                mismatched_keys.append((key, value.shape))

        if mismatched_keys:
            print("Shape mismatches found:")
            for key, shape in mismatched_keys:
                print(f"  {key}: {shape}")
        else:
            print("All holdout frames have the same shape.")
    else:
        print("No holdout frames were loaded.")