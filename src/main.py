import threading
from pathlib import Path
from scraping.rain_areas import check_history, datetime_now_str, SG_OFFSET_HOURS, run_scraper_forever, get_previous_ticks, fetch_radar_snapshot
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(SEED)
from data_processing.radar_dataset import radar_dataset
from models.convlstm import ConvLSTM

def load_model():
    model = ConvLSTM(input_dim=1,
                    hidden_dim=64,
                    kernel_size=(3, 3),
                    num_layers=2,
                    batch_first=True,
                    bias=True)


    model = model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\Jacobs laptop\rainraingoaway\models\model_best_20260623_104758.pkl"))
    model.eval()
    return model 


def load_missing_png():
    dt_now= datetime_now_str(offset_hours=SG_OFFSET_HOURS)	
    prev_ticks= get_previous_ticks(dt_now)
    for prev in prev_ticks:
        _,_,_=fetch_radar_snapshot(img_name="70km", dt = prev)
      
def main() -> None:
    dt_start= datetime_now_str(offset_hours=SG_OFFSET_HOURS)
    scraper_thread = threading.Thread(target=run_scraper_forever, daemon=True, name="radar-scraper")
    scraper_thread.start()
    print(f"Started radar scraper thread, time: {dt_start}")
    model = load_model()
    previous_loaded = check_history("70km", dt=dt_start)
    print(f"Past 15 mins data available: {previous_loaded}")
    if not previous_loaded:
        load_missing_png()
    # Keep telegram polling in the main thread. 
    run_bot(model=model, folder_path=DATA_PATH)


if __name__ == "__main__":
	main()
