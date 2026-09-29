# Forecast and radar feedback

Weather messages with a known location now include **Not raining** and **Raining**
buttons. Each button identifies the location, Singapore time, and whether the
report concerns a forecast or an actual radar observation. Report only conditions
you personally observed at that location and time.

- Forecast reports open at the forecast's target time, not when the message arrives.
- Reports close 30 minutes after the displayed target time. Request **My forecast**
  or **Current radar** for a recent update when an old button has expired.
- Tapping the other button corrects your report. Repeated taps do not create extra
  votes for that message/location. Members of a group can each give their own report;
  reports do not replace the group's shared buttons. Successful taps show a brief
  confirmation such as "Saved: Main — not raining. Thanks!" only to the reporter.
  Confirmations and errors disappear automatically, with no OK dialog to dismiss.
- Saved-location forecasts, shared-location forecasts, actual radar, radar fallbacks,
  and newly generated automatic notices support feedback. A map with no known
  location and old queued notices without location snapshots have no feedback buttons.
- Automatic notices may show an actual radar map alongside a future forecast claim.
  The **forecast** button refers to the forecast time; **radar** refers to observation
  time. Reports are stored against that specific claim.

The bot preserves the coordinates, label, observation and target times, radar and
forecast values when available, and neighbourhood radius in metres. Later changes
to saved locations do not change earlier reports. The report time is stored
separately from the weather time. Radar and model values are relative intensities,
not millimetres of rainfall.

## Storage and deployment

No new dependencies or MySQL migration are required. The bot creates
`data/forecast_feedback.sqlite3` on first use. The existing Docker `data/` mount
persists it across container rebuilds. Keep this database with your private data
backups; it includes location coordinates, chat IDs, and reporter IDs. It is ignored
by Git and should not be published.

On the machine running the bot:

```sh
git pull
docker compose -f compose.bot.yaml up -d --build bot
```

The `feedback_snapshots` table stores immutable weather context. The
`feedback_reports` table has one current vote per snapshot token and Telegram user,
including the source message ID, first reporting time and latest correction time.
Times are timezone-aware UTC ISO strings; buttons display Singapore time.

For local analysis, join `feedback_reports.token = feedback_snapshots.token`.
`raining = 0` means the person reported no rain; `raining = 1` means rain.
Deduplicate by location, weather time and reporter when comparing reports across
several messages about the same event. Several people may disagree about the same
event, so retain that disagreement rather than treating every report as ground truth.

These are unverified, point-location human observations. They do not change the
live model, suppress alerts, or enter training automatically. Compare them with
rain gauges and account for radar neighbourhood size and reporting delay before
using them for evaluation or training.

If feedback storage is unavailable, weather messages still send. Failed report
writes show a retry message rather than claiming the report was saved.
