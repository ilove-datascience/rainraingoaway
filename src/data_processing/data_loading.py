SEED = 67
import json
import os
import threading
import torch 
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np 
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data_processing.pngtojson import points_to_intensity_grid, png_to_xy_intensity
    from masking import lat_long_to_pixel
else:
    from .pngtojson import points_to_intensity_grid, png_to_xy_intensity
    from masking import lat_long_to_pixel

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
            intensity_df = intensity_df / 100.0
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
            intensity_df = intensity_df / 100.0
            data.append(intensity_df)
            count += 1
    print(f"Loaded {count} images.")
    
    return data

#prev_key = first_key
def create_samples(data, list_length=10, min_list_length = 7):
    count = 1 
    data_grouped = list()
    sample_list= list()
    
    
    ttl_cnt = 0 
    keys = sorted(data.keys())
    prev_key = keys[0]

    for key in keys[1:]:
        value = data[key]
        value = torch.from_numpy(np.asarray(value)).float()
        ttl_cnt+=1
        key_time = datetime.strptime(str(key).strip(".png"),"%Y%m%d%H%M")
        prev_key_time = datetime.strptime(str(prev_key).strip(".png"),"%Y%m%d%H%M")
        change = key_time - prev_key_time
        change = int(change.seconds/60)
        #print(f"doing key {key}, with change {change}, cureent length is {len(sample_list)}, with count {count}")
        
    
        if change > 5: # if non continous jump, end sample
            #data_grouped.append(sample_list)
            print(f"sample with lenth {len(sample_list)} created due to discontinuation in series, with change {change}")
            sample_list = list()
            sample_list.append(value)
            
            count=1
            
        elif change == 5 and count <list_length: # add value to sample if we arent at the limit
            sample_list.append(value)
            
            #print("appended")
            count += 1 
            
        elif change == 5 and count >= list_length: # shelve sample and start a new sample
            data_grouped.append(sample_list)
           # print(f"sample with lenth {len(sample_list)} created")
            sample_list = list()
            sample_list.append(value)
            
            count=1
        else:
            print("FUCCCKKK YOU FOING HERE" ) # should never reach this
        
        prev_key = key # pass on key information 
        
    #removing samples too short
    filtered=list()
    for idx, item in enumerate(data_grouped):
        length = len(item)

        if length < min_list_length:
           pass
        else:
            filtered.append(item)
    #print(f"removed {len(data_grouped)- len(filtered)} items for length")
    data_grouped = filtered
    x=list()
    y=list()

    for i in data_grouped:
        length= len(i)
        
        #print(length)
        last = i.pop(length-1)
        length= len(i)
        
        x.append(i)
        y.append(last)
        
    return x,y
            
  
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

    radar_mtime = os.path.getmtime(radar_path)
    env_mtime = os.path.getmtime(env_path)

    if metadata.get("radar_mtime") != radar_mtime or metadata.get("env_mtime") != env_mtime:
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
        intensity_df = pd.DataFrame(intensity_grid)
        intensity_df = intensity_df / 100.0

        env_stack = build_env_data(
            env_path,
            verbose=False,
            height=intensity_df.shape[0],
            width=intensity_df.shape[1],
        )
        if env_stack is None:
            return None, 'incomplete_env', radar_name, env_file_name

        radar_stack = intensity_df.values[np.newaxis, :, :]  # (1, H, W)
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
            print(f"Skipping {env_path}: no complete env rows after dropping missing values")
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