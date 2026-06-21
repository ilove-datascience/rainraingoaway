
# RainRainGoAway

Simple research codebase for radar-to-rain forecasting using a ConvLSTM model.

This repository contains data preprocessing utilities, training and evaluation code, and a notebook that demonstrates forecasting the next radar frame from the latest PNG sequence.

## Repository layout

- `src/` — project source code (data loaders, models, training, utilities)
	- `pngtojson.py` — helpers to convert radar PNG to intensity grids
	- `models/convlstm.py` — ConvLSTM implementation used as the predictor
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
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If you prefer `uv`, the repo also has a lockfile:

```powershell
uv sync
```

2. Prepare data

Place PNG sequences under `data/<resolution>/png/`. The notebook expects filenames formatted like `YYYYMMDDHHMM.png` and uses the latest 4 PNGs in `data/70km/png` as input to forecast the next frame.

3. Run the notebook

Open and run `src/visualise_model.ipynb` with Jupyter / VS Code notebooks to load the latest 4 images, print their datetimes, and forecast the next frame.

```powershell
jupyter lab src/visualise_model.ipynb
```

## Training

Training is demonstrated in the notebook using the `ConvLSTM` model at `src/models/convlstm.py`. The model outputs a single-channel map produced by a 1×1 convolution and applies a sigmoid in the forward pass, so the returned values are already normalized to the 0 to 1 range.

If you need the raw pre-sigmoid values for experimentation, you would need to adjust the model code first.

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



