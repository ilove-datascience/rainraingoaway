SEED = 67
import os
import torch 
from datetime import datetime
import pandas as pd
import numpy as np 
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data_processing.pngtojson import points_to_intensity_grid, png_to_xy_intensity, png_to_xy_binary
    from masking import lat_long_to_pixel
else:
    from .pngtojson import points_to_intensity_grid, png_to_xy_intensity, png_to_xy_binary
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
            
  
def load_data_multimodal(folder_path_radar, folder_path_env, total=None, verbose=True):
    # Load the data from the specified folder
    data = dict()
    count = 0 
    skipped_missing_env = 0
    skipped_incomplete_env = 0
    radar_files = sorted(os.listdir(folder_path_radar))

    for file_name in radar_files:
        
        if file_name.endswith('.png'):
            
            if total is not None and count >= total:
                break
            
            # remove .png from radar file name
            radar_name = file_name.replace('.png', '')

            env_file_name = f"weather_{radar_name}.csv"
            env_path = os.path.join(folder_path_env, env_file_name)

            if not os.path.exists(env_path):
                if verbose:
                    print(f"Skipping {radar_name}: missing env file {env_file_name}")
                skipped_missing_env += 1
                continue

            intensity_points = png_to_xy_intensity(
                os.path.join(folder_path_radar, file_name),
                include_zero=True
            )

            intensity_grid = points_to_intensity_grid(intensity_points)
            intensity_df = pd.DataFrame(intensity_grid)
            intensity_df = intensity_df / 100.0

            env_stack = build_env_data(env_path, verbose=verbose)
            if env_stack is None:
                if verbose:
                    print(f"Skipping {radar_name}: env data incomplete")
                skipped_incomplete_env += 1
                continue

            radar_stack = intensity_df.values[np.newaxis, :, :]  # (1, 120, 217)
            combined_stack = np.concatenate([radar_stack, env_stack], axis=0).astype(np.float32)

            data[file_name] = combined_stack
            if verbose:
                print(f"loaded {file_name} with shape {combined_stack.shape}")
            count += 1
            
    skipped_total = skipped_missing_env + skipped_incomplete_env
    print(
        f"Loaded {count} images. Skipped {skipped_total} frames "
        f"({skipped_missing_env} missing env files, {skipped_incomplete_env} incomplete env files)."
    )
    
    return data

def build_env_data(env_path, verbose=True):
    data_cols = ["humidity", "temperature", "wind_dir", "wind_speed"]
    loc_cols=["longitude","latitude"]
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

    env_data = {}
    for i in data_cols:
        i_df = pd.DataFrame(index=range(120), columns=range(217), dtype=float)
        for index,row in env_df.iterrows():
            pixelx,pixely= lat_long_to_pixel(long=row[loc_cols[0]], lat=row[loc_cols[1]],width=217, height=120)
            value = row[i]
            i_df.iloc[pixely, pixelx] = value

        # Fill unmapped pixels so the multimodal tensor stays finite.
        i_df = i_df.fillna(0.0)

        env_data[i]= i_df
            

   
    #print(env_df.head(100))
    env_stack = np.stack([
    env_data["humidity"].values,
    env_data["temperature"].values,
    env_data["wind_dir"].values,
    env_data["wind_speed"].values
    ])
    #print(f"built env stack for {os.path.basename(env_path)} with shape {env_stack.shape}")
    return env_stack

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