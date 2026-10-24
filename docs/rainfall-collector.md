# Rainfall archive collector

The bot now starts a separate `rainfall-archive` daemon alongside radar and general weather collection. Previously rainfall was downloaded only with the manual historical-data script; the weather collector does not collect rainfall.

The worker checks hourly for missing completed Singapore calendar days. It repairs holes within the past 14 days and catches up from the latest saved archive after longer downtime. It never downloads the current day into a completed-day file. Failures are logged and retried; they do not block radar, forecast inference or Telegram. Missing readings remain unknown, not zero.

Files are saved under `data/environment/rainfall/rainfall_YYYYMMDD.json`, using the existing archive format. Every API page is fetched and validated before atomic publication. Existing files are not overwritten. Pagination loops, invalid units, conflicting readings and empty responses are rejected. Full pagination does not guarantee every station reported every interval; downstream dataset coverage checks still apply.

The existing `gov_api_key` / `GOV_API_KEY` is used. No new secret or scheduler is required. After pulling this change, restart the bot; Docker installations must rebuild/recreate because source is copied into the image:

```powershell
docker compose -f compose.bot.yaml up -d --build
```

Use the installation's existing additional Compose files if applicable. Check logs for `Started daily rainfall archive collector` and `Rainfall archived` messages. Warning messages identify failed dates without printing API credentials.

For a one-off repair on a source checkout, without starting Telegram:

```powershell
.venv/Scripts/python.exe scripts/collect_rainfall.py --once
```

Omit `--once` to run the collector independently with hourly checks. The one-off command exits nonzero when any day failed. Downloading data does not train models or evaluate reserved test periods.
