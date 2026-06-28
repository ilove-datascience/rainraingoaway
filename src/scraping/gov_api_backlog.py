import os
import datetime
import random
import time
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from gov_api import fetch_once, save_to_csv
else:
    from .gov_api import fetch_once, save_to_csv

def load_data_names(folder_path, file_type = ".png"):
    # Load the data from the specified folder
    data = []
    count = 0 
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_type):
            file_name = file_name.strip(file_type)
            data.append(file_name)
            count += 1
    print(f"Loaded {count}")
    
    return data


def main():
    data_env = load_data_names("C:\\Users\\Jacobs laptop\\rainraingoaway\\data\\environment", ".csv")
    data_radar = load_data_names("C:\\Users\\Jacobs laptop\\rainraingoaway\\data\\70km\\png", ".png")
    data_env_clean= []
    for i, sample in enumerate(data_env):
        
        data_env_clean.append(sample.strip("weather_"))

    radar_set= set(data_radar)
    env_set= set(data_env_clean)
    missing = [] 

    for item in radar_set:
        if item not in env_set:
            missing.append(item)
            print(f"Missing {item}")
            
            
    print(f"Missing {len(missing)} items")
    for i in missing:
        query_date = datetime.datetime.strptime(str(i), "%Y%m%d%H%M")
        query_date = query_date.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"fetching for {query_date}")
        try:
            data_df = fetch_once(query_date)
        except Exception as exc:
            print(f"skipping {query_date} after fetch failure: {exc}")
            continue

        output_path = save_to_csv(data_df, timestamp=i)
        sleep_time = 5 + random.randint(0, 5)
        print(f"sleep for {sleep_time}")
        time.sleep(sleep_time)

    print("DONE WAHOO YIPPIE WAHO")

if __name__ =="__main__":
    main()