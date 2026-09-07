import threading
from pathlib import Path
from scraping.rain_areas import check_history, datetime_now_str, SG_OFFSET_HOURS, run_scraper_forever, get_previous_ticks, fetch_radar_snapshot
from scraping.gov_api import main as run_weather_scraper_forever
from telegram_code.telegram_bot import run_bot
from scraping.gov_api_backlog import load_data_names
SEED = 67
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "70km" / "png")
import torch
from datetime import datetime
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device name:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

torch.manual_seed(SEED)
from data_processing.radar_dataset import radar_dataset
from models.multi_modal_convlstm import ConvLSTM_MM

def load_model():
    model = ConvLSTM_MM(
        input_dim=7,
        hidden_dim=[32, 64],
        kernel_size=[(3, 3), (3, 3)],
        num_layers=2,
        batch_first=True,
        bias=True,
        use_land_use=False,
    )


    model = model.to(device)
    production_model_path = BASE_DIR / "models" / "model_best_latest.pkl"
    if production_model_path.exists():
        model_path = production_model_path
    else:
        checkpoints = sorted(
            (BASE_DIR / "models").glob("model_best_*.pkl"),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No model_best_*.pkl checkpoint found in {BASE_DIR / 'models'}")
        model_path = checkpoints[-1]
    print(f"Loading model checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model 


def load_missing_png():
    dt_now= datetime_now_str(offset_hours=SG_OFFSET_HOURS)	
    prev_ticks= get_previous_ticks(dt_now)
    for prev in prev_ticks:
        _,_,_=fetch_radar_snapshot(img_name="70km", dt = prev)


def ensure_normalization_stats():
    """Load saved normalization stats, computing and saving them once if missing."""
    import json
    stats_path = BASE_DIR / "models" / "normalization_stats.json"
    if stats_path.exists():
        print(f"Loaded normalization stats: {stats_path}")
        return json.loads(stats_path.read_text(encoding="utf-8"))

    print("No saved normalization stats found; computing from training split (one-off)...")
    from data_processing.multimodal_radar_dataset import radar_dataset_multimodal
    dataset = radar_dataset_multimodal(
        str(BASE_DIR / "data" / "70km" / "png"),
        str(BASE_DIR / "data" / "environment"),
        list_length=3,
        total=None,
        num_workers=8,
        use_land_use=False,
    )
    split_idx = int(0.8 * len(dataset))
    dataset.fit_normalization(list(range(0, split_idx)))
    stats = dataset.get_normalization_stats()
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved normalization stats -> {stats_path}")
    return stats
      
def main() -> None:
    dt_start= datetime_now_str(offset_hours=SG_OFFSET_HOURS)
    scraper_thread = threading.Thread(target=run_scraper_forever, daemon=True, name="radar-scraper")
    scraper_thread.start()
    print(f"Started radar scraper thread, time: {dt_start}")
    weather_thread = threading.Thread(target=run_weather_scraper_forever, daemon=True, name="weather-scraper")
    weather_thread.start()
    print("Started weather scraper thread")
    norm_stats = ensure_normalization_stats()
    model = load_model()
    previous_loaded = check_history("70km", dt=dt_start)
    print(f"Past 15 mins data available: {previous_loaded}")
    if not previous_loaded:
        load_missing_png()
    # Keep telegram polling in the main thread. 
    run_bot(model=model, folder_path=DATA_PATH, norm_stats=norm_stats)


if __name__ == "__main__":
	main()
