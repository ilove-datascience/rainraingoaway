SEED = 67
import os
import torch 
from datetime import datetime
import pandas as pd




from .pngtojson import points_to_intensity_grid, png_to_xy_intensity, png_to_xy_binary

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
        value= torch.from_numpy(value.to_numpy()).float()
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
            
            
