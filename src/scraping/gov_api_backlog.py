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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_data_names(folder_path, file_type = ".png"):
    # A fresh checkout may not have an environment directory yet.
    data = sorted(path.stem for path in Path(folder_path).glob(f"*{file_type}") if path.is_file())
    print(f"Loaded {len(data)}")
    return data


def main():
    data_env = load_data_names(PROJECT_ROOT / "data" / "environment", ".csv")
    data_radar = load_data_names(PROJECT_ROOT / "data" / "70km" / "png", ".png")
    data_env_clean= []
    for i, sample in enumerate(data_env):
        
        data_env_clean.append(sample.removeprefix("weather_"))

    radar_set= set(data_radar)
    env_set= set(data_env_clean)
    missing = [] 

    for item in sorted(radar_set):
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
