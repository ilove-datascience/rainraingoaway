import json
import queue
import threading
from pathlib import Path

import torch

from data_processing.multimodal_radar_dataset import radar_dataset_multimodal
from models.multi_modal_convlstm import ConvLSTM_MM
from scraping.gov_api import main as run_weather_scraper_forever
from scraping.frame_cache import run_frame_cache_worker
from scraping.rain_areas import (
    SG_OFFSET_HOURS,
    check_history,
    datetime_now_str,
    fetch_radar_snapshot,
    get_previous_ticks,
    run_scraper_forever,
)
from telegram_code.telegram_bot import run_bot

SEED = 67
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "70km" / "png")

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
    dt_now = datetime_now_str(offset_hours=SG_OFFSET_HOURS)
    prev_ticks = get_previous_ticks(dt_now)
    for prev in prev_ticks:
        fetch_radar_snapshot(img_name="70km", dt=prev)


def ensure_normalization_stats():
    """Load saved normalization stats, computing and saving them once if missing."""
    stats_path = BASE_DIR / "models" / "normalization_stats.json"
    if stats_path.exists():
        print(f"Loaded normalization stats: {stats_path}")
        return json.loads(stats_path.read_text(encoding="utf-8"))

    print("No saved normalization stats found; computing from training split (one-off)...")
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
    dt_start = datetime_now_str(
        offset_hours=SG_OFFSET_HOURS
    )

    model_ready_queue = queue.Queue()
    file_ready_queue = queue.Queue()
    threading.Thread(
        target=run_frame_cache_worker,
        args=(file_ready_queue, model_ready_queue),
        daemon=True,
        name="frame-cache",
    ).start()

    scraper_thread = threading.Thread(
        target=run_scraper_forever,
        kwargs={
            "file_ready_queue": file_ready_queue
        },
        daemon=True,
        name="radar-scraper"
    )

    scraper_thread.start()

    print(
        f"Started radar scraper thread, time: {dt_start}"
    )

    weather_thread = threading.Thread(
        target=run_weather_scraper_forever,
        kwargs={"file_ready_queue": file_ready_queue},
        daemon=True,
        name="weather-scraper"
    )

    weather_thread.start()

    print("Started weather scraper thread")

    norm_stats = ensure_normalization_stats()
    model = load_model()

    previous_loaded = check_history(
        "70km",
        dt=dt_start
    )

    print(
        f"Past 15 mins data available: {previous_loaded}"
    )

    if not previous_loaded:
        load_missing_png()

    run_bot(
        model=model,
        folder_path=DATA_PATH,
        norm_stats=norm_stats,
        model_ready_queue=model_ready_queue
    )


if __name__ == "__main__":
    main()
