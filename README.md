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
| `tests/fixtures/` | Small, versioned test examples; not live weather or training archives |
| `docs/` | Deployment documentation |
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

Copy the template to `.env` in the repository root, then fill in your credentials:

```powershell
Copy-Item .env.example .env
```

Keep an existing `.env` rather than overwriting it. The required settings are:

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

The bot requires an existing MySQL database. Restore your existing bot database
when moving machines to preserve users, locations, and notification state.
Merely setting `MYSQL_DATABASE` does not create the database.

For a new empty database, `uv run python scripts/run_bot_service.py --init-db`
creates the base tables and runs the additive migration. Use a schema-administrator
account for this one-off command; it does not start polling. Docker instructions,
including the optional fresh database service, are in [the deployment guide](docs/docker-bot.md).

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

Startup launches independent 70 km and 240 km radar scrapers, the weather scraper, and the frame-cache worker,
loads the model, checks recent radar history, and starts Telegram polling. The
prediction queue is checked every 30 seconds; radar frames follow five-minute
ticks. Allow time for usable radar and weather inputs to arrive.

240 km maps are saved under `data/240km/png/` every five-minute radar tick, with
retries when unavailable. Forecasting continues to use 70 km maps; the separate
240 km worker does not feed the model queue or delay the 70 km worker.

In a private chat or group with the bot:

- `/start`: register your location and choose an alert mode.
- `/menu`: show the buttons (and leave any unfinished setup).
- `/cancel`: cancel setup and clear the buttons.
- **My forecast**: request a forecast for your saved location.
- **Current radar** or `/actual`: request the latest radar snapshot.
- `/weather`: request the official Singapore daily outlook and temperature range.
- **Weather updates** or `/weatheralerts`: turn all scheduled daily forecasts on or off.
- **Change location**: update your saved location.
- **Alert settings** or `/setmode`: choose automatic or manual updates.

Each chat has independent settings. A new group starts with `/start`, no saved
location, and manual alerts; it never inherits a member's personal settings.
Group members share the group's saved locations, alert mode, and setup conversation.
Automatic alerts go to the chat where they were enabled. Existing private-chat
settings remain valid. The legacy `userid` database columns now store Telegram
chat IDs and must support signed BIGINT values (group IDs are negative).

Private chats and groups have **Add location**, **Saved locations**, and **Remove location** buttons.
Add and Remove prompt for a name; Add then asks for a Telegram location.
Buttons clear after completed actions or cancellation. Use `/menu` to bring them
back. Automatic alerts do not open the menu. The equivalent commands also work:

- `/addlocation Office`: then share a Telegram location. Use `/cancel` to cancel.
- `/locations`: list saved names and coordinates.
- `/removelocation Office`: remove an extra location and cancel its pending alerts.
- **My forecast**: show one map with colour-coded markers and a location legend below the map for all saved locations and a short rain-status line for each.

The initial location is **Main**. **Change location** updates Main; **Current radar**
shows its marker. Main stays saved and can be changed rather than removed.
Each private chat or group can save up to **6 locations, including Main**. Remove an extra location before adding a seventh. Names must be unique within a group (up to 80 characters). Automatic alerts include
location names and track rain independently at each location; **Alert settings**
applies to the whole group. Private chats use the same location controls and combined maps. Personal locations remain separate from group locations.

Before running this version, rerun `python -m telegram_code.rain_state_db` with
`PYTHONPATH=src`. This preserves existing locations as Main and old queued alerts,
adds location IDs and labels, and extends the location and notification keys.
Do not run an older bot version once extra locations have been added.


Automatic mode sends an alert when rain is predicted or detected, then confirms
clearance after sustained dry radar. Nearby location changes are grouped into
one notice per chat. Manual mode pauses automatic alerts.
Unavailable or stale forecasts can fall back to an actual radar snapshot.

Forecast and actual-radar messages now offer **Not raining** and **Raining**
feedback for the location and time shown on each button. Reports open at that
time and close after 30 minutes; tapping the other button corrects your report.
See [feedback and private storage](docs/forecast-feedback.md) for details.

### Daily weather outlook

Open **Weather updates** from `/menu`, or send `/weatheralerts`. A single
**Daily weather updates: On/Off** switch controls all three bulletins:

| Singapore time | Bulletin |
| --- | --- |
| 5 am | Today's weather |
| 12 noon | Updated outlook for today |
| 10 pm | Tomorrow's weather |

Updates are **off by default** and start at the next scheduled time after opting
in. Each private chat or group has its own switch; only group admins can change
the group's switch. These bulletins send on dry days too, without a saved
location or a successful radar/model run. **Alert settings** still controls
local rain alerts separately. Daily outlooks are no longer appended to rain alerts.
Use `/weather` any time for an on-demand outlook, even with scheduled updates off.
No extra API key or environment setting is needed.

Bulletins and `/weather` now include **hourly rain estimates near each saved
location**, such as **1 pm–3 pm: rain possible; hourly chance up to 70%**, plus
the peak hour and forecast millimetres in that hour. This example describes the
format, not a current forecast. Times are estimates and can shift. Nearby places
can share the same forecast grid; these estimates are separate from the bot's
short-range radar alerts.

Hourly data comes from [Open-Meteo](https://open-meteo.com/en/docs), clearly
labelled separately from the official NEA outlook. Rain windows highlight hours
with at least 40% precipitation probability or at least 0.1 mm forecast. Adjacent
hours are grouped; gaps are preserved. Up to three strongest windows are shown
per location (two when there are more than three saved locations), with any
omission stated. Noon updates omit completed hours. Probabilities refer to each
hour, not the chance across the combined window. Open-Meteo timestamps describe
the preceding hour; the bot accounts for this when displaying time ranges.

Only coordinates rounded to two decimal places are sent to Open-Meteo; chat IDs
and saved names stay local. With no saved location, a clearly labelled Singapore
reference point is used. Requests for the same approximate location/day are
shared, cached for 15 minutes, and backed off for five minutes on failure. Its
[free API is for non-commercial use](https://open-meteo.com/en/pricing); no paid
service or account is configured. If hourly data cannot be fetched, the official
bulletin still sends with NEA's broader time windows when available.

Morning and noon use the [official NEA/MSS 24-hour forecast](https://data.gov.sg/datasets/d_ce2eb1e307bda31993c533285834ef2b/view).
The night bulletin selects tomorrow's exact date from the
[official four-day outlook](https://data.gov.sg/datasets/d_f131f6e343bf8168e4057a04c4326a0a/view).
Before a suitable morning issue is available, the bot can use the prior day's
four-day issue for today's date. Bulletins show the temperature range, issue time
and actual validity window, plus wind and humidity when available. Regional
time windows are also parsed from NEA and used when hourly estimates are
unavailable. These national forecasts are separate from local five-minute rain
predictions. A noon bulletin is sent even if NEA's outlook has
not changed since morning.

The scheduler checks every minute using Singapore time regardless of the host's
timezone. Forecast requests are shared across subscribers and retried after five
minutes if unavailable. Following a brief restart, a missed bulletin can send up
to 30 minutes after its scheduled time; older bulletins are skipped. If no valid
source becomes available within that window, the bulletin is skipped. `/weather`
remains available to try again.

Preferences and per-slot delivery records are created automatically in
`data/weather_schedule.sqlite3`, covered by the existing Docker data mount and
Git ignore rules. Keep this file across restarts to preserve opt-ins and avoid
duplicate bulletins. A send with an uncertain result is not retried, since it may
already have arrived. Telegram rate limits are retried within the delivery window.

## Telegram cute mode

Send `/cutemode` in a private chat to toggle cat-style messages. This typed command
is not listed in the bot's buttons or command menu. Preferences are saved per chat
in `data/bot_preferences.sqlite3` and survive restarts.

Manual forecasts randomly choose from several weather-specific lines. Repeats are allowed. Examples:

- Rain expected: “Rain might be padding over—keep your paws dry, meow! 🐾”
- No rain expected: “No rain on my whiskers for now, meow! 🐾”
- Rain cleared: “The rain has padded away, meow! 🐾”

Grouped automatic notices add at most one neutral cat line, so mixed rain and
clearance updates never imply the same conditions at every location.

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
The bot sends a heartbeat every 30 seconds from its Telegram event loop,
independently of radar availability, predictions, and database operations. The
first attempt is scheduled one second after the job scheduler starts. Use a
60-second Kuma interval with 2–3 retries to allow for network or scheduling delays.

This is a **bot-liveness monitor**, not a forecast/data-health monitor. A stopped
process, blocked event loop, or unreachable Kuma server can still cause downtime.
An Up status does not guarantee fresh radar, successful database operations, or
Telegram notification delivery. Forecast freshness rules are unchanged.

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

Notebook generators can rewrite notebooks. They are development tools, not bot
startup steps. The old one-off notebook patch scripts have been removed; their
already-applied changes remain in the notebooks and generator source.

## Tests

```powershell
uv run python -m pytest -q
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

Keep notebook code and explanatory Markdown in Git; save generated charts, model
results and executed notebook copies under ignored `reports/` or the matching
model-run directory. Committed notebooks have outputs and execution counts cleared.
Obsolete development plans and one-off patch scripts remain recoverable in Git
history. Runtime map assets stay at the root because the bot uses
those paths. Existing model checkpoints are retained with their matching normalization.

The standalone legacy PNG diagnostic lives in `scripts/inspect_legacy_png.py`:

```powershell
uv run python scripts/inspect_legacy_png.py tests/fixtures/radar/example_240km.png
```

It writes `points.json` and `grid.json` to ignored `reports/png-inspection/`.
Its legacy 0–100 colour scale is separate from the source-aware model data contract.
Automated tests are discovered only in `tests/`; local worktrees and research
notebooks are not collected as tests.

### Runtime logs

Starting the bot with `python src/main.py` writes console output and Python logs
into `logs/rainraingoaway.log` under the project directory. This includes radar,
weather, model and scheduler messages, with timestamps, severity and thread names.
The active file rotates at 10 MiB, keeping five backups (`.log.1` to `.log.5`).
Logs append across restarts and the entire `logs/` directory is ignored by Git.
Existing diagnostic output can contain user IDs and locations; review logs before sharing.

Weather collection retries failed timestamps for up to one hour, then expires them.
Slow requests do not skip intervening ticks. On restart, the previous hour is checked
for gaps; use `src/scraping/gov_api_backlog.py` to recover older missing files.

## Docker bot service

See [Docker bot setup](docs/docker-bot.md) for persistent bot, radar/weather
collection and forecasting services using your existing MySQL database, with an
optional containerized database. After configuring credentials and model files:

```powershell
docker compose -f compose.bot.yaml up -d --build
```


### Observed-rain alerts and delivery recovery

Automatic alerts also report **Rain detected** when fresh radar shows rain without
an earlier active rain alert, even if the forecast predicts dry. A separate
30-second observation job needs only one radar frame less than 10 minutes old;
it does not require model weather inputs or a successful model run. Forecast
alerts expire at their forecast time; observed notices, including clearance,
expire ten minutes after their radar observation. Superseded pending changes
are discarded. Existing active rain episodes suppress repeat starts.

Definitive delivery failures release the episode marker so new observations can
trigger an alert. Interrupted/uncertain deliveries are never blindly resent:
after 15 minutes they are marked `abandoned`, logged, and stop suppressing new
events, provided no newer event/settings change supersedes them. A new event may
therefore alert again if the original uncertain send actually reached Telegram.
Known Telegram receipts retry their database acknowledgement without resending.
The delivery-recovery change itself requires no schema migration; see the later
rain-alert stability update below for its additive migration.


### Rain alert stability (September 28 update)

The first rain warning does not wait for dry confirmation. Brief dry readings no longer close an
active rain episode: clearance needs consecutive radar observations spanning
15 minutes of dry conditions (four five-minute frames). A wet frame resets the
dry timer; a missing observation breaks the sequence. Timers are stored per saved
location in MySQL and survive restarts. Predicted-clear notices are no longer
sent automatically: both radar-first and model-first processing use confirmed
dry observations to close an episode. A forecast that never became observed
rain is also retired after sustained dry radar, even if the model is unavailable.

After closure, forecast-only warnings wait 15 minutes before starting a new
episode. Fresh observed rain bypasses that wait. This prevents a same-frame
model update from immediately contradicting a clearance, while retaining actual
new rain detection.

Changes wait 15 seconds to collect adjacent radar/model results, then the
30-second delivery jobs claim each chat's eligible locations together. They send
one notice with one labelled actual-radar map where a shared map is available.
Forecast lines explicitly give their future time. Long captions or mixed map
timestamps use one complete text notice instead. Changed/removed saved locations
invalidate an older queued map. Each location retains its own durable delivery
receipt, even when the Telegram message is shared.

Example with two locations (illustrative, not live weather):

```text
Rain update

• Main: Light to Moderate rain detected.
• Office: Rain cleared (radar dry for 15 minutes).

Radar: 28 Sep, 14:00 SGT
Intensity is relative to the radar colour scale.
```

The earlier stability update adds `rain_dry_since` and `rain_episode_wet` to `user_location`.
Grouping and daily outlooks add no further schema changes.
After pulling, rebuild and run the existing additive migration before starting:

```powershell
docker compose -f compose.bot.yaml build bot
docker compose -f compose.bot.yaml run --rm bot --init-db
docker compose -f compose.bot.yaml up -d bot
```

The one-off migration requires schema-administrator permissions. Normal bot startup
only validates the schema; it does not change it. Existing data is preserved.
