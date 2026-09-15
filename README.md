# RainRainGoAway

Singapore radar forecasting and a Telegram rain-alert bot, with offline tools for
training and evaluating local rain-arrival models.

The live bot uses a multimodal ConvLSTM forecast. The arrival-model framework and
v5 experiment notebooks are separate research workflows; running the bot does not
load an arrival model.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/main.py` | Starts radar/weather scrapers, frame caching, and the Telegram bot |
| `src/telegram_code/` | Commands, local forecasts, persistent rain state, notification delivery, cute mode, and Kuma heartbeats |
| `src/scraping/` | Radar downloads, GovSG weather collection, and frame caching |
| `src/data_processing/` | Radar decoding, multimodal inputs, normalization, checkpoint contracts, and training windows |
| `src/models/multi_modal_convlstm.py` | Live multimodal ConvLSTM model |
| `src/arrival/` | Rain-arrival datasets, models, training, calibration, and inference |
| `src/evaluation/` | Offline forecast and notification diagnostics |
| `scripts/` | Evaluation, replay, smoke checks, and notebook tooling |
| `tests/` | Automated regression tests |
| `models/` | Checkpoints and normalization; some existing files are tracked |
| `data/` | Local radar/weather data, caches, and bot preferences; ignored by Git |
| `reports/` | Generated evaluation outputs; ignored by Git |

## Install

Use Python **3.12 or newer**. Run these commands from the repository root.

With `uv`:

```powershell
uv sync --group dev
```

Or create a virtual environment:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest-asyncio
```

The examples below use `uv run`. With an activated virtual environment, replace
`uv run python` with `python`. Jupyter is optional and is not listed as a project
dependency; install it separately if you want to use the notebooks.

## Run the Telegram bot

### 1. Configure the environment

Create `.env` in the repository root:

```dotenv
tele_api_key=YOUR_TELEGRAM_BOT_TOKEN
gov_api_key=YOUR_GOVSG_API_KEY
MYSQLHOST=localhost
MYSQLUSER=YOUR_DATABASE_USER
MYSQL_ROOT_PASSWORD=YOUR_DATABASE_PASSWORD
MYSQL_DATABASE=WeatherBot
KUMA_PUSH_URL=
```

- `tele_api_key`: required Telegram bot token. The name is case-sensitive.
- `gov_api_key`: required by the weather scraper; `GOV_API_KEY` is also accepted.
- `MYSQLHOST`, `MYSQLUSER`, `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE`: MySQL settings.
  Despite its name, `MYSQL_ROOT_PASSWORD` supplies the password for `MYSQLUSER`.
  Code defaults are `localhost`, `root`, an empty password, and `WeatherBot`.
- `KUMA_PUSH_URL`: optional complete Uptime Kuma push URL; see monitoring below.

Keep `.env` private. It is ignored by Git. Restart the bot after changing it.

### 2. Prepare MySQL

The bot requires an existing MySQL database with the base `users` and
`user_location` tables. This repository does not include a complete fresh-database
installer: restore your existing bot database/schema before running it on a new
machine. Merely setting `MYSQL_DATABASE` does not create the database.

After the base tables exist, run the additive rain-state migration:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
uv run python -m telegram_code.rain_state_db
```

This adds missing rain-state columns and creates/updates `rain_notifications` for
durable delivery. The database user needs the corresponding schema permissions.
The migration is a separate command; normal bot startup does not invoke it.

### 3. Check the model files

`src/main.py` loads `models/model_best_latest.pkl` when present. Otherwise it
selects the most recently modified `models/model_best_*.pkl` file directly inside
`models/`. It does not search timestamp subdirectories.

Use normalization from the same model run in `models/normalization_stats.json`.
If this file is absent, startup tries to calculate it from local training data;
a fresh clone without that data cannot rely on this fallback. The loader checks
checkpoint metadata when present; unversioned checkpoints use legacy decoding.

The current entry point explicitly uses **CPU** inference. Its CUDA diagnostics
do not mean it selected the GPU.

### 4. Start

```powershell
uv run python src/main.py
```

Startup launches the radar scraper, weather scraper, and frame-cache worker,
loads the model, checks recent radar history, and starts Telegram polling. The
prediction queue is checked every 30 seconds; radar frames follow five-minute
ticks. Allow time for usable radar and weather inputs to arrive.

In a private chat with the bot:

- `/start`: register your location and choose an alert mode.
- **My forecast**: request a forecast for your saved location.
- **Current radar** or `/actual`: request the latest radar snapshot.
- **Change location**: update your saved location.
- **Alert settings** or `/setmode`: choose automatic or manual updates.

Automatic mode sends an alert when rain is predicted, then updates when rain is
predicted to end or radar shows it has ended. Manual mode pauses automatic alerts.
Unavailable or stale forecasts can fall back to an actual radar snapshot.

## Telegram cute mode

Send `/cutemode` in a private chat to toggle cat-style messages. This typed command
is not listed in the bot's buttons or command menu. Preferences are saved per chat
in `data/bot_preferences.sqlite3` and survive restarts.

Forecasts and rain alerts randomly choose from several weather-specific lines. Repeats are allowed. Examples:

- Rain expected: “Rain might be padding over—keep your paws dry, meow! 🐾”
- No rain expected: “No rain on my whiskers for now, meow! 🐾”
- Rain cleared: “The rain has padded away, meow! 🐾”

Settings prompts and confirmations also get short, context-specific cat lines. For example:

```text
Current alert setting: automatic.

A peek at your weather-cat preferences. 🐱

Select mode:

Choose my assignment, meow! 🐈
```

Cute mode does not change weather facts, alert timing, or keyboard options.

## Uptime Kuma monitoring

Create a **Push** monitor named **RainRaingoAway**, with a **60-second heartbeat
interval** and **2–3 retries**.

Copy the complete push URL into `.env` and restart:

```dotenv
KUMA_PUSH_URL=https://YOUR_KUMA_HOST/api/push/YOUR_SECRET_TOKEN
```

Keep any query parameters supplied by Kuma. Leaving the URL unset disables monitoring.
The bot sends at most one heartbeat per minute after successful queue/database
processing. It requires a successfully processed radar observation less than
15 minutes old, allowing normal radar publication delays. Failed inference or
database operations suppress the heartbeat for that cycle. Stalled radar eventually
stops heartbeats; repeated old frames cannot extend this window. Forecast display
and alert freshness remain governed by the original five-minute forecast window.
This monitor does not guarantee that every Telegram notification was delivered.

Push requests run outside the Telegram event loop, have a five-second timeout,
and do not stop the bot if Kuma is unreachable. The push URL is a secret; do not
commit it.

## Data and preprocessing

Local inputs use these paths:

```text
data/70km/png/YYYYMMDDHHMM.png
data/environment/weather_YYYYMMDDHHMM.csv
```

Weather inputs include longitude, latitude, humidity, temperature, wind direction,
and wind speed. The live pipeline builds radar/environment frames and applies
saved normalization. Offline loaders skip unusable inputs rather than treating
them as valid training examples.

`radar_codec.py` versions radar decoding. `model_contract.py` checks compatibility
between checkpoint metadata, normalization, and preprocessing. The multimodal
dataset also supports overlapping training windows with chronological frame
partitions; input cleaning avoids mutating shared target frames.

## ConvLSTM training and visualization

The script entry point is:

```powershell
uv run python src/train_model.py
```

**This script writes production asset paths:** best weights go to
`models/model_best_latest.pkl`, and normalization goes to
`models/normalization_stats.json`. Keep a copy of deployed assets before training
in the same checkout. Loss history is written to
`models/<timestamp>/multimodal_convlstm_losses_<timestamp>.csv`.

Notebook entry points:

- `src/visualise_model.ipynb`: inspect radar sequences and forecasts.
- `src/test_multimodal_convlstm_long.ipynb`: longer ConvLSTM training/evaluation workflow.

Inspect paths, data availability, and execution settings before running notebook
cells. Training and notebook execution are not required to start an already
configured bot.

## Rain-arrival research workflows

| Entry point | Purpose |
| --- | --- |
| `src/rain_arrival_model.ipynb` | Arrival-model training and evaluation |
| `src/rain_arrival_experiments.ipynb` | Architecture and experiment comparisons |
| `src/rain_arrival_v5_full_data.ipynb` | Resumable full-archive v5 training |
| `scripts/smoke_arrival_pipeline.py` | Small real-data integration check, including two training updates |
| `scripts/replay_arrival_notifications.py` | Offline replay of calibrated arrival alerts |
| `scripts/compare_arrival_alert_burden.py` | Compare alert burden across runs |
| `scripts/locked_forward_evaluation.py` | Freeze and evaluate saved finalists |

These workflows require local radar/weather archives. Replays additionally need
saved run weights and calibration files, which are generally ignored by Git and
are not supplied by a fresh clone. Full-archive training produces new weights;
previous calibration and evaluation results do not automatically validate them.

For a small integration check with the required archive available:

```powershell
uv run python scripts/smoke_arrival_pipeline.py
```

For replay options:

```powershell
uv run python scripts/replay_arrival_notifications.py --help
```

Replay accepts `--run`, `--calibration`, and `--output`. Inspect its configured data
requirements before evaluating your own run.

The locked-forward script has `freeze` and `evaluate` phases. Its finalist paths,
date range, and report directory are fixed in the script for a specific historical
comparison. Review those requirements before running either phase; it is not a
generic fresh-clone evaluation command.

Notebook generators and the historical `add_*`, `fix_*`, `update_*`, `prepare_*`,
`expand_*`, `harden_*`, and `configure_*` helpers can rewrite notebooks. Some run
at import time. They are development tools, not bot startup steps.

## Tests

```powershell
uv run python -m pytest tests -q
```

Focused bot-style and heartbeat checks:

```powershell
uv run python -m pytest tests/test_cute_mode.py tests/test_heartbeat.py tests/test_model_queue.py -q
```

If Windows denies access to pytest's default temporary folder, choose a new,
dedicated writable directory with `--basetemp`. Pytest clears that directory;
do not point it at existing project data.

## Updating an existing checkout

To update the branch you are currently running:

```powershell
git pull --ff-only
uv sync --group dev
```

Restart the bot to load code or environment changes. Preserve local edits before
switching branches. `.env`, private data, caches, generated reports, and new model
artifacts are ignored; already-tracked checkpoints remain tracked despite the
`models/` ignore rule.

## Contributing

Keep changes and tests grouped by topic. Do not commit credentials, private chat
data, or generated caches. Include data and model prerequisites with evaluation
instructions, and distinguish integration checks from forecasting-quality results.
