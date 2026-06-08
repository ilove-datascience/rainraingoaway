
# RainRainGoAway

Simple research codebase for radar-to-rain prediction using a ConvLSTM model.

This repository contains data preprocessing utilities, training and evaluation code, and a notebook that demonstrates training and visualising ConvLSTM predictions on radar PNG sequences.

## Repository layout

- `src/` — project source code (data loaders, models, training, utilities)
	- `pngtojson.py` — helpers to convert radar PNG to intensity grids
	- `models/convlstm.py` — ConvLSTM implementation used as the predictor
	- `test_convlstm.ipynb` — interactive notebook for training and visualising
- `data/` — example data folders (70km, 240km) and `environment/` CSVs
- `models/` — saved model checkpoints (not committed)
- `tests/` — test scripts
- `pyproject.toml` — project metadata

## Quick start

1. Create and activate a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the basics:

```powershell
pip install torch torchvision matplotlib pandas scipy
```

2. Prepare data

Place PNG sequences under `data/<resolution>/png/`. The notebook expects filenames formatted like `YYYYMMDDHHMM.png` and contiguous 5-minute steps to form input sequences.

3. Run the notebook

Open and run `src/test_convlstm.ipynb` with Jupyter / VS Code notebooks to train and visualise model behaviour.

```powershell
jupyter lab src/test_convlstm.ipynb
```

## Training

Training is demonstrated in the notebook using the `ConvLSTM` model at `src/models/convlstm.py`. The model outputs a single-channel map produced by a 1×1 convolution (no final activation).

Important: the model returns raw logits (real-valued outputs). To interpret values as probabilities apply a sigmoid, e.g. `prob = torch.sigmoid(output)`.

Example: a raw value of `0.03` → `sigmoid(0.03) ≈ 0.5075` (≈50.8%), not 3%.

## Inference & Visualization

The notebook shows how to fetch a batch from the `DataLoader`, run `model(inputs)`, and plot `actual`, `prediction`, and `difference`. If you prefer probabilities in plots, apply sigmoid before plotting:

```python
pred = model(inputs)
prob = torch.sigmoid(pred)
```

## Tips
- If you need binary prediction thresholds, apply `prob > 0.5` (or another threshold) after sigmoid.
- Keep model checkpoints in `models/` and add that folder to `.gitignore` if not already excluded.

## Contributing

Issues and PRs are welcome. For changes that affect data formats or training behaviour, include reproducible notebook cells or scripts.



