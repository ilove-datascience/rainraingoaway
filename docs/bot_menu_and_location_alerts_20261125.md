# Compact menu and location alerts

`/menu` opens a persistent Telegram reply keyboard in the user's keyboard area. Calling it again removes the keyboard; tapping Close does the same. Navigation sends a short message and replaces the keyboard buttons. Forecasts, radar replies and ordinary bot replies leave the keyboard available. Back exits location setup. The keyboard sets `is_persistent=True`, `one_time_keyboard=False` and `resize_keyboard=True`.

The home keyboard contains five buttons in three rows:

| Row | Buttons |
| --- | --- |
| 1 | My forecast · Current radar |
| 2 | Locations & alerts |
| 3 | Help · Close |

Locations & alerts lists saved places with their current alert status. Selecting a place opens one On/Off action, its location management action, Back and Close. Extra-location removal requires confirmation. Help explains forecasts, radar, default-on alerts and cancellation. `/help`, `/locations`, `/setmode` and previous text-button labels remain supported.

New chats and newly saved locations start with automatic alerts on, as requested. Changing Main's coordinates preserves its current alert selection. Turning a location off preserves its saved coordinates and manual forecasts. Existing manual chat settings remain off during migration. If a location is added or individually enabled after a legacy global pause, only that location turns on. Other saved locations remain off. Group settings remain shared within that chat and separate from members' private settings.

Selection writes lock the user row, increment the changed location's settings version, reset its rain episode and cancel pending notifications. Repeated desired-value callbacks are idempotent. Forecast generation, queue insertion and final delivery all enforce the per-location selection. A stale removal confirmation cannot delete a replacement location with the same name.

The implementation uses the installed python-telegram-bot [reply keyboard API](https://docs.python-telegram-bot.org/en/stable/telegram.replykeyboardmarkup.html). Button texts map to the current member's displayed actions; alert buttons name their location. Group replies target the requesting member with a selective keyboard, while saved settings remain chat-wide. Old inline callbacks only retire their attached buttons and cannot change settings. No additional dependency or service is required.

## Validation

The targeted regression suite passed 117 tests and 13 subtests, including asynchronous handler and delivery tests. The 18 opt-in MySQL temporary-table integration tests were skipped. Tests use mocked Telegram I/O and isolated database fixtures; no live Telegram messages were sent.

Coverage includes persistent keyboard flags, opening/closing with /menu, Help and Back, per-member navigation, explicit location switches, default-on additions, disabled-location queue cancellation, grouped rain notices, feedback buttons, request timeouts and read-only schema validation. The publication branch preserves the latest main branch's forecast workers, radar-only notices, named feedback and database timeouts.

## Deployment

The database needs the additive `user_location.alerts_enabled` column before the new code is started. The existing explicit `scripts/run_bot_service.py --init-db` path performs the migration with a schema administrator account. Normal startup remains read-only and now validates this column. Existing subscriptions are preserved by the column's default of 1 and their existing chat mode; new application-created locations explicitly start on.

Build the updated bot image, apply that migration using the established deployment configuration, then restart the bot. Verify in Telegram that `/menu` opens and closes, Help returns to the main keyboard, new locations show alerts on, disabling one location leaves others unchanged, and a disabled location produces no automatic alert. Manual forecasts continue to include every saved location.

Keyboard navigation state is held per member in the bot's chat context. After a bot restart, use `/menu` to refresh the keyboard; stale location actions are rejected. The menu occupies Telegram's keyboard area and is not attached underneath a message. Telegram clients ultimately control keyboard visibility and allow users to switch to typing when needed.
