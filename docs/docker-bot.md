# Docker bot service

The bot image runs `src/main.py`: Telegram polling, CPU ConvLSTM inference,
70 km and 240 km radar scrapers, weather collection and the frame-cache worker.
This service performs inference only; it does not start model training.

## Existing installation (recommended)

Run from the repository root with Docker Desktop running Linux containers.
Keep the existing `.env`, `data/` and `models/` directories. The image excludes
credentials, model weights, datasets and notebooks from its build context.

Required `.env` entries:

```dotenv
tele_api_key=your-token
gov_api_key=your-key
MYSQLUSER=your-existing-mysql-user
MYSQL_ROOT_PASSWORD=your-existing-mysql-password
MYSQL_DATABASE=WeatherBot
BOT_MYSQL_HOST=host.docker.internal
KUMA_PUSH_URL=
```

`BOT_MYSQL_HOST` defaults to the host machine. Set it to the existing database
server's DNS name/IP if MySQL is elsewhere. Compose overrides `MYSQLHOST` because
`localhost` inside the container means the container itself. MySQL must accept
the supplied user from the Docker network; a localhost-only grant is insufficient.

Mount the production checkpoint and its matching `normalization_stats.json` in
`models/`. The usual loader prefers `model_best_latest.pkl`, otherwise the newest
`model_best_*.pkl` directly in that directory. The directory is mounted read-only;
this service requires normalization to exist and never refits it on startup.

```powershell
docker compose -f compose.bot.yaml build
docker compose -f compose.bot.yaml run --rm --no-deps bot --check
```

The check loads the actual model, checks its metadata when supplied, and imports
the bot without database writes, scraping or Telegram polling.

Stop the old bot instance before starting the service, so only one process polls
the Telegram token. Back up the existing database before upgrading its schema.

```powershell
docker compose -f compose.bot.yaml up -d
docker compose -f compose.bot.yaml logs -f --tail 100 bot
docker compose -f compose.bot.yaml ps
```

Startup waits for MySQL and validates existing columns and location keys using
read-only queries. Normal operation needs SELECT/INSERT/UPDATE/DELETE, without
CREATE or ALTER. Existing chat ID columns must support signed BIGINT group IDs.
For a missing/outdated schema, run the following once using a database account
with CREATE/ALTER permissions, then restore the normal runtime account:

```powershell
docker compose -f compose.bot.yaml run --rm --no-deps bot --init-db
```

The service restarts after process failures and Docker restarts unless explicitly
stopped. Docker Desktop itself must start at login on Windows. No inbound ports
are published. There is no container health probe: an `Up` status alone does not
prove fresh forecasts; use logs and the optional existing Kuma heartbeat.

## Persistent files

- `data/70km/png/` and `data/240km/png/`: downloaded radar.
- `data/environment/`: weather records.
- `data/multimodal_cache/`: inference cache.
- `data/bot_preferences.sqlite3`: cute-mode preferences.
- `logs/`: rotating application logs; Docker logs are also size-limited.
- Existing MySQL: users, locations, settings and notification state.

Radar/weather archives continue accumulating as before; log rotation does not
delete those archives. Back up data, matching model/normalization and MySQL.

## Optional fresh containerized database

This creates a separate database; it does **not** move users from existing MySQL.
For migration, restore a database backup into it before starting the bot.
Add two distinct strong passwords to `.env`:

```dotenv
BOT_DB_PASSWORD=choose-a-bot-database-password
BOT_DB_ROOT_PASSWORD=choose-a-different-root-password
```

```powershell
docker compose -f compose.bot.yaml -f compose.bot.mysql.yaml up -d mysql
docker compose -f compose.bot.yaml -f compose.bot.mysql.yaml run --rm --build bot --init-db
docker compose -f compose.bot.yaml -f compose.bot.mysql.yaml up -d
```

The database uses the `bot-mysql` named volume, publishes no host port, and the bot
waits for its SQL health check. [Docker documents this dependency ordering](https://docs.docker.com/compose/how-tos/startup-order/).
Use both `-f` arguments for all subsequent commands for this installation.
Do not use `down -v` unless you intend to delete the database volume.

## Updates and stopping

```powershell
docker compose -f compose.bot.yaml up -d --build
docker compose -f compose.bot.yaml stop
```

Rebuilding preserves mounted data and logs. Recreate the service after changing
`.env`; a plain restart does not reload Compose environment configuration.

If Telegram reports `InvalidToken` / `Unauthorized`, stop the service, replace
`tele_api_key` in `.env` with a valid BotFather token, then run `up -d` again.
The Docker service cannot repair or renew Telegram credentials itself.
