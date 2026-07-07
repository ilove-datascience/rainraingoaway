
# RainRainGoAway

Simple research codebase for radar-to-rain forecasting using a ConvLSTM model.

This repository contains data preprocessing utilities, training and evaluation code, and a notebook that demonstrates forecasting the next radar frame from the latest PNG sequence.

## Repository layout

- `src/` — project source code (data loaders, models, training, utilities)
	- `data_processing/pngtojson.py` — helpers to convert radar PNG to intensity grids
	- `data_processing/data_loading.py` — radar + environment loading utilities
	- `data_processing/multimodal_radar_dataset.py` — multimodal dataset wrapper used by training
	- `models/convlstm.py` — ConvLSTM implementation used as the predictor
	- `train_model.py` — script training entry point (multimodal)
	- `visualise_model.ipynb` — notebook for loading recent images and forecasting the next frame
	- `scraping/rain_areas.py` — radar scraper for 70km and 240km PNG captures
- `data/` — example data folders (70km, 240km) and `environment/` CSVs
- `models/` — saved model checkpoints (not committed)
- `tests/` — test scripts
- `pyproject.toml` — project metadata

## Quick start

1. Create and activate a Python virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
pip install -e .
```

If you prefer `uv`, the repo also has a lockfile:

```powershell
uv sync
```

2. Prepare data

Place radar PNG sequences under `data/<resolution>/png/` with filenames formatted as `YYYYMMDDHHMM.png`.

For multimodal training (`python src/train_model.py`), place matching environment CSVs under `data/environment/` with filenames:

- `weather_YYYYMMDDHHMM.csv`

Each CSV should include:

- `longitude`, `latitude`
- `humidity`, `temperature`, `wind_dir`, `wind_speed`

Frames without a matching CSV or with incomplete environment columns are skipped during multimodal loading.

3. Run the notebook

Open and run `src/visualise_model.ipynb` with Jupyter / VS Code notebooks to load the latest 4 images, print their datetimes, and forecast the next frame.

```powershell
jupyter lab src/visualise_model.ipynb
```

## Training

You can train with either the notebook or the script:

```powershell
python src/train_model.py
```

The script trains a multimodal ConvLSTM model and saves outputs under a run-specific timestamp folder:

- `models/<timestamp>/model_<timestamp>.pkl` (best validation model)
- `models/<timestamp>/multimodal_convlstm_losses_<timestamp>.csv` (epoch, train_loss, val_loss)

Example:

- `models/20260707_153012/model_20260707_153012.pkl`
- `models/20260707_153012/multimodal_convlstm_losses_20260707_153012.csv`

If you run `src/main.py`, ensure it points to an existing checkpoint file from your `models/<timestamp>/` run folder.

## Inference & Visualization

The notebook shows how to fetch the latest 4 PNGs from `data/70km/png`, run `model(inputs)`, and plot the 4 input frames plus the predicted next frame. It also prints the timestamps of the input images before inference.

The current notebook cell does not compare against a known future target image because the next frame is unknown at prediction time.

## Scraping

`src/scraping/rain_areas.py` downloads the latest radar images for `70km` and `240km`. If a request times out, it now retries the same URL before falling back to an earlier 5-minute tick. This helps keep the scraper running through transient network issues.

## Tips
- If you need binary prediction thresholds, apply `prob > 0.5` (or another threshold) after sigmoid.
- Keep model checkpoints in `models/` and add that folder to `.gitignore` if not already excluded.

## Contributing

Issues and PRs are welcome. For changes that affect data formats or training behaviour, include reproducible notebook cells or scripts.



