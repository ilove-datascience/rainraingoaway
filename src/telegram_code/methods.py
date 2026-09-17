import asyncio
import os
import queue
import sys
import threading
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from scipy import ndimage
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import ContextTypes, ConversationHandler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.data_loading import build_and_cache_frame, build_env_data, remove_small_echoes
from data_processing.pngtojson import png_to_xy_intensity, points_to_intensity_grid
from data_processing.radar_codec import decode_png, SOURCE, source_category
from masking import lat_long_to_pixel
from scraping.rain_areas import SG_OFFSET_HOURS, attempt_get_most_recent, datetime_now_str, get_previous_ticks
from telegram_code.database import add_location, add_user, get_autoupdate_users, get_location, save_mode_choice
from telegram_code.group_locations import list_locations
from telegram_code.states import WAITING_FOR_LOCATION, WAITING_FOR_MODE
from telegram_code.forecast_policy import is_fresh, forecast_text, sg_now
from telegram_code.rain_state import next_rain_state, should_notify
from telegram_code.rain_state_db import get_rain_locations, save_rain_state
from telegram_code.notification_delivery import deliver_notifications, settings_lock
from telegram_code.notification_text import alert_text, rain_notice, intensity_label
from telegram_code.local_rain import local_rain, in_coverage
from telegram_code.rain_state import ENDED
from telegram_code.database import get_user_mode


# Gated rain-prediction pipeline constants (mirror test_multimodal_convlstm.ipynb).
# threshold=0.55 chosen as the best structural operating point from the validation-only
# mask sweep: cleanest background/noise rejection with best large-component IoU among
# configs within ~0.001 CSI of the peak (see mask_extended_validation_report.csv).
SEQUENCE_LENGTH = 3
from telegram_code.forecast_mask import clean_rain_mask, RAIN_PROBABILITY_THRESHOLD, MIN_COMPONENT_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RADAR_FOLDER = PROJECT_ROOT / "data" / "70km" / "png"
ENV_FOLDER = PROJECT_ROOT / "data" / "environment"


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Channel indices (match multimodal_radar_dataset.channel_map)
CH_RADAR, CH_TEMP, CH_HUM, CH_WIND_U, CH_WIND_V, CH_STATION, CH_DIST = range(7)
ZSCORE_CHANNELS = [CH_TEMP, CH_HUM, CH_WIND_U, CH_WIND_V]


ENV_TICK_TOLERANCE_MINUTES = 15



def apply_normalization(x: torch.Tensor, norm_stats) -> torch.Tensor:
	"""Z-score env channels and scale distance channel. x: [1, T, 7, H, W]."""
	if not norm_stats:
		return x

	ordered = ["temperature", "humidity", "wind_u", "wind_v"]
	clip = float(norm_stats.get("zscore_clip", 4.0))
	means = torch.tensor([float(norm_stats[n]["mean"]) for n in ordered],
						dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
	stds = torch.tensor([max(float(norm_stats[n]["std"]), 1e-4) for n in ordered],
						dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)

	x[:, :, ZSCORE_CHANNELS] = torch.clamp(
		(x[:, :, ZSCORE_CHANNELS] - means) / stds, min=-clip, max=clip
	)

	dist_scale = float(norm_stats.get("distance", {}).get("scale", 0.0))
	if dist_scale > 0:
		x[:, :, CH_DIST] = torch.clamp(x[:, :, CH_DIST] / dist_scale, min=0.0, max=1.0)
	return x




def _find_usable_env_path(tick, height, width, tolerance_minutes=ENV_TICK_TOLERANCE_MINUTES):
	"""Return (env_path, env_stack) for `tick`, trying nearby ticks if needed.

	A CSV can exist but be unusable (e.g. the temperature API lagged and the
	scraper wrote a file with no temperature column), so file existence alone
	isn't enough â€” build the env stack and search outward in 5-minute steps
	until one succeeds.
	"""
	tick_dt = datetime.strptime(str(tick), "%Y%m%d%H%M")
	candidate_dts = [tick_dt]
	for offset in range(5, tolerance_minutes + 1, 5):
		candidate_dts.append(tick_dt - timedelta(minutes=offset))
		candidate_dts.append(tick_dt + timedelta(minutes=offset))

	saw_file = False
	for candidate_dt in candidate_dts:
		candidate_path = ENV_FOLDER / f"weather_{candidate_dt.strftime('%Y%m%d%H%M')}.csv"
		if not candidate_path.exists():
			continue
		saw_file = True
		env_stack = build_env_data(
			str(candidate_path), verbose=False, height=height, width=width
		)
		if env_stack is not None:
			return candidate_path, env_stack

	if not saw_file:
		raise FileNotFoundError(
			f"No weather CSV found for tick {tick} within {tolerance_minutes} minutes"
		)
	return None, None


def build_multimodal_input(prev_ticks, folder_path, norm_stats=None):
	"""Build a [1, T, 7, H, W] multimodal tensor from previous ticks.

	Channel order matches the training dataset:
	radar, temperature, humidity, wind_u, wind_v, station_mask, distance.
	"""
	ticks = sorted(prev_ticks)  # chronological, oldest -> newest
	if len(ticks) > SEQUENCE_LENGTH:
		ticks = ticks[-SEQUENCE_LENGTH:]

	frames = []
	for tick in ticks:
		# Cache-first: build_and_cache_frame reads cache if present, else builds+writes.
		combined = build_and_cache_frame(tick, img_name="70km", verbose=False)

		if combined is None:
			# Fallback: build inline (radar + env) without caching.
			radar_path = os.path.join(folder_path, f"{tick}.png")

			intensity_points = png_to_xy_intensity(radar_path, include_zero=True)
			intensity_grid = points_to_intensity_grid(intensity_points)
			radar_grid = remove_small_echoes(np.asarray(intensity_grid, dtype=np.float32) / 100.0)

			env_path, env_stack = _find_usable_env_path(
				tick, height=radar_grid.shape[0], width=radar_grid.shape[1]
			)
			if env_stack is None:
				raise ValueError(f"Missing/incomplete env data for tick {tick}")

			combined = np.concatenate([radar_grid[np.newaxis, :, :], env_stack], axis=0)

		frames.append(combined.astype(np.float32))

	# [T, 7, H, W]
	x = np.stack(frames, axis=0)
	x = torch.from_numpy(x).float()
	x = x.unsqueeze(0)  # [1, T, 7, H, W]
	x = apply_normalization(x, norm_stats)
	return x


def load_token(env_key: str = "tele_api_key") -> str:
    token = os.getenv(env_key)
    if token:
        return token.strip()

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_key:
                return value.strip().strip('"').strip("'")

    raise RuntimeError(
        f"Missing {env_key}. Set it in environment variables or add it to .env at repo root."
    )



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    userid = update.effective_chat.id
    if userid < 0:
        await update.message.reply_text('Group locations: /addlocation Name, /locations, /removelocation Name. My forecast checks all locations; Change location updates Main. Alert settings apply to all locations.', reply_markup=ReplyKeyboardRemove())
    location = await asyncio.to_thread(get_location, userid)
    if location:
        await update.message.reply_text("Welcome back. Use /menu for forecasts and settings.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    await asyncio.to_thread(add_user, userid)
    await update.message.reply_text("Welcome! Send your Telegram location to set up local rain forecasts.", reply_markup=ReplyKeyboardRemove())
    return WAITING_FOR_LOCATION


async def change_location(update, context):
    await update.message.reply_text("Send your new Telegram location. This will be used for automatic alerts.", reply_markup=ReplyKeyboardRemove())
    return WAITING_FOR_LOCATION


async def receive_location(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    userid = update.effective_chat.id
    location = update.message.location

    latitude = location.latitude
    longitude = location.longitude

    print(userid, latitude, longitude)
    
    if not in_coverage(latitude, longitude):
        await update.message.reply_text("That location is outside our radar coverage. Please send a location within Singapore's radar map.", reply_markup=ReplyKeyboardRemove())
        return WAITING_FOR_LOCATION
    async with settings_lock(context):
        success = await asyncio.to_thread(add_location, userid, lat=latitude, long=longitude)
    if not success:
        await update.message.reply_text("Couldn't save your location. Please send it again.", reply_markup=ReplyKeyboardRemove())
        return WAITING_FOR_LOCATION
    context.application.bot_data.setdefault("alert_history", {}).pop(userid, None)
    print("rcv lcoation called ")
    await update.message.reply_text("Location updated/saved", reply_markup=ReplyKeyboardRemove())
    
    await update.message.reply_text(
		"Select mode:",
		reply_markup=ReplyKeyboardMarkup(
			keyboard = [
				["1 - Automatic rain updates"],
				["2 - Manual updates only"]
			],
			resize_keyboard=True,
			one_time_keyboard=True
		)
	)

    return WAITING_FOR_MODE

async def update_mode(update, context):
    userid = update.effective_chat.id
    current_mode = await asyncio.to_thread(get_user_mode, userid)
    if current_mode is None:
        await update.message.reply_text("Use /start to set up your location first.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    await update.message.reply_text(f"Current alert setting: {current_mode}.", reply_markup=ReplyKeyboardRemove())
    await update.message.reply_text(
		"Select mode:",
		reply_markup=ReplyKeyboardMarkup(
			keyboard = [
				["1 - Automatic rain updates"],
				["2 - Manual updates only"]
			],
			resize_keyboard=True,
			one_time_keyboard=True
		)
	)

    return WAITING_FOR_MODE
async def receive_mode(update, context):
    userid = update.effective_chat.id
    choice = update.message.text

    if choice == "1 - Automatic rain updates":
        mode = "automatic"

    elif choice == "2 - Manual updates only":
        mode = "manual"

    else:
        await update.message.reply_text(
            "Please select one of the options below."
        , reply_markup=ReplyKeyboardRemove())
        return WAITING_FOR_MODE

    print(userid, mode)
    async with settings_lock(context):
        success = await asyncio.to_thread(save_mode_choice, userid, mode)
    if success:
        print('mode updated')
        context.application.bot_data.setdefault("alert_history", {}).pop(userid, None)
        message = ("Automatic alerts enabled: one when rain is predicted, then only when "
                   "rain is predicted to end or radar shows it has ended." if mode == "automatic"
                   else "Automatic alerts paused. Use /menu then My forecast whenever you need an update.")
        await update.message.reply_text(message, reply_markup=ReplyKeyboardRemove())
    else:
        await update.message.reply_text("Couldn't save your settings. Please try again.", reply_markup=ReplyKeyboardRemove())
        return WAITING_FOR_MODE
    

    return ConversationHandler.END


async def handle_msg(update, context, model, folder_path, norm_stats=None):
    if update.message.text == "Current radar":
        await handle_actual(update, context, folder_path)
        return
    if update.message.text != "My forecast":
        await update.message.reply_text("Use /menu to choose an option, or share a location for a forecast.", reply_markup=ReplyKeyboardRemove())
        return
    if update.effective_chat.id < 0 or len(await asyncio.to_thread(list_locations, update.effective_chat.id)) > 1:
        await handle_group_forecasts(update, context, model, folder_path, norm_stats)
        return
    location = await asyncio.to_thread(get_location, update.effective_chat.id)
    if not location:
        await update.message.reply_text("Use /start to save your location first.", reply_markup=ReplyKeyboardRemove())
        return
    if not in_coverage(float(location[0]), float(location[1])):
        await update.message.reply_text("Your saved location is outside radar coverage. Tap Change location.", reply_markup=ReplyKeyboardRemove())
        return
    prediction = context.application.bot_data.get("latest_prediction")
    if not is_fresh(prediction):
        await update.message.reply_text("Checking for a current forecast…", reply_markup=ReplyKeyboardRemove())
        prediction = await run_model(model, folder_path, norm_stats)
    if not is_fresh(prediction):
        await send_actual_fallback(update, folder_path, location)
        return
    context.application.bot_data["latest_prediction"] = prediction
    image, caption = await asyncio.to_thread(build_location_forecast, float(location[0]), float(location[1]), prediction)
    if not is_fresh(prediction):
        image.close()
        await send_actual_fallback(update, folder_path, location)
        return
    await update.message.reply_photo(photo=image, caption=caption, reply_markup=ReplyKeyboardRemove())


async def run_model(model, folder_path, norm_stats=None, tick=None):
    """
    Run the rainfall model using the latest available radar/environment data.

    Returns:
        dict containing prediction data, or None if input data is unavailable.
    """

    try:
        if tick is None:
            latest = await asyncio.to_thread(get_latest_radar_png, folder_path)
            if latest is None:
                return None
            tick = int(latest.stem)
        prev_ticks = get_previous_ticks(int(tick), count=SEQUENCE_LENGTH, most_recent_success=True)

        most_recent_tick = datetime.strptime(
            str(prev_ticks[0]),
            "%Y%m%d%H%M"
        )

        next_tick = most_recent_tick + timedelta(minutes=5)

        # Build model input
        x = await asyncio.to_thread(
            build_multimodal_input,
            prev_ticks,
            folder_path=folder_path,
            norm_stats=norm_stats
        )

        device = next(model.parameters()).device
        x = x.to(device)

        # Inference
        model.eval()

        with torch.inference_mode():
            raw_prediction, last_radar, delta, rain_logits = model(
                x,
                return_logits=True
            )

        # ----------------------------
        # Rain gating
        # ----------------------------

        raw_prediction = (
            raw_prediction
            .detach()
            .cpu()
            .squeeze(0)
            .squeeze(0)
            .clamp_min(0)
        )

        rain_probability = (
            torch.sigmoid(rain_logits)
            .detach()
            .cpu()
            .squeeze(0)
            .squeeze(0)
        )

        raw_np = raw_prediction.numpy()
        probability_np = rain_probability.numpy()

        cleaned_mask = clean_rain_mask(probability_np)

        gated_prediction = raw_np * cleaned_mask

        # Match orientation used by your plotting code
        pred_plot = np.flipud(gated_prediction)

        return {
            "prediction": pred_plot,
            "next_tick": next_tick,
            "observed_at": most_recent_tick,
            "actual_radar": np.flipud(x[0, -1, CH_RADAR].detach().cpu().numpy()),

            # Keep these if you want diagnostics later
            "raw_prediction": raw_np,
            "rain_probability": probability_np,
            "cleaned_mask": cleaned_mask,
            "gated_prediction": gated_prediction,

            "prev_ticks": prev_ticks
        }

    except (FileNotFoundError, ValueError) as exc:
        print(f"run_model failed: {exc}")
        return None

    except Exception as exc:
        print(f"Unexpected model error: {exc}")
        return None
    


async def handle_location(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    model,
    folder_path,
    norm_stats=None
):
    location = update.message.location

    latitude = location.latitude
    longitude = location.longitude
    if not in_coverage(latitude, longitude):
        await update.message.reply_text("That location is outside our radar coverage.", reply_markup=ReplyKeyboardRemove())
        return

    latest_prediction = context.application.bot_data.get(
        "latest_prediction"
    )

    if not is_fresh(latest_prediction):
        latest_prediction = await run_model(
            model,
            folder_path,
            norm_stats
        )

    if not is_fresh(latest_prediction):
        await send_actual_fallback(update, folder_path, (latitude, longitude))
        return

    plot_buffer, caption = await asyncio.to_thread(
        build_location_forecast,
        latitude,
        longitude,
        latest_prediction
    )

    if not is_fresh(latest_prediction):
        plot_buffer.close()
        await send_actual_fallback(update, folder_path, (latitude, longitude))
        return
    await update.message.reply_photo(
        photo=plot_buffer,
        caption=caption,
        reply_markup=ReplyKeyboardRemove(),
    )


def get_latest_radar_png(folder_path) -> Path | None:
    """Return the most recently captured raw radar PNG, or None if there are none."""
    pngs = sorted(Path(folder_path).glob("*.png"))
    return pngs[-1] if pngs else None


_plot_lock = threading.Lock()


def render_heatmap(grid, title, colorbar_label, marker=None):
    # Matplotlib's pyplot state is shared by manual requests and queued alerts.
    with _plot_lock:
        return _render_heatmap(grid, title, colorbar_label, marker)


def _render_heatmap(grid, title, colorbar_label, marker=None):
    """Render a phone-friendly weather card with unchanged radar coordinates."""
    sg_base_img = np.flipud(plt.imread(
        str(Path(__file__).resolve().parents[2] / "sgbaseimg_70km.png")))
    background, ink, muted = "#f3f6fa", "#172b46", "#61738b"
    fig = plt.figure(figsize=(8, 8) if isinstance(marker, list) else (8, 7), dpi=160, facecolor=background)
    try:
        heading, separator, timestamp = title.partition(" | ")
        fig.text(0.07, 0.95, "RAINRAINGOAWAY  /  SINGAPORE", fontsize=10,
                 weight="bold", color=muted)
        fig.text(0.07, 0.905, heading, fontsize=21, weight="bold", color=ink)
        if separator:
            fig.text(0.07, 0.872, timestamp, fontsize=11, color=muted)
        ax = fig.add_axes([0.055, 0.34 if isinstance(marker, list) else 0.19, 0.89,
                           0.49 if isinstance(marker, list) else 0.64])
        ax.set_facecolor("white")
        ax.imshow(sg_base_img, origin="lower",
                  extent=[0, grid.shape[1] - 1, 0, grid.shape[0] - 1], zorder=0)
        im = ax.imshow(grid, cmap="turbo", origin="lower",
                       alpha=np.where(grid < 0.003, 0.0, 0.78),
                       norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0),
                       interpolation="nearest", aspect="equal", zorder=1)
        # Preserve the source map proportions even if the model grid is resized.
        ax.set_aspect((sg_base_img.shape[0] / sg_base_img.shape[1]) *
                      ((grid.shape[1] - 1) / (grid.shape[0] - 1)))
        if isinstance(marker, list):
            fig.text(0.07, 0.29, 'SAVED LOCATIONS', fontsize=10, weight='bold', color=ink)
            fig.text(0.07, 0.265, 'Coloured dots identify places only.', fontsize=10, color=muted)
            from matplotlib.lines import Line2D
            fig.add_artist(Line2D([0.07, 0.93], [0.16, 0.16], transform=fig.transFigure,
                                  color='#cbd5e1', linewidth=1))
            for index, (label, colour, (pixel_x, pixel_y)) in enumerate(marker):
                ax.scatter(pixel_x, pixel_y, c=colour, s=95,
                           edgecolors="white", linewidths=1.8, zorder=4)
                column, row = index % 3, index // 3
                x, y = 0.07 + column * 0.30, 0.225 - row * 0.035
                fig.text(x, y, "●", color=colour, fontsize=14)
                short_label = label if len(label) <= 24 else label[:23] + '…'
                fig.text(x + 0.025, y, short_label, fontsize=10, color=ink)
        elif marker is not None:
            pixel_x, pixel_y = marker
            ax.scatter(pixel_x, pixel_y, c="#f0529c", s=230, alpha=0.2,
                       edgecolors="none", zorder=3)
            ax.scatter(pixel_x, pixel_y, c="#e83288", s=65,
                       edgecolors="white", linewidths=2, zorder=4)
            fig.text(0.07, 0.17, "●", color="#e83288", fontsize=13)
            fig.text(0.095, 0.17, "Your saved location", fontsize=10, color=muted)
        ax.set_xlim(0, grid.shape[1] - 1)
        ax.set_ylim(0, grid.shape[0] - 1)
        ax.set_axis_off()
        scale_ax = fig.add_axes([0.07, 0.09, 0.86, 0.018])
        # Use an opaque legend even though rain is blended over the map.
        from matplotlib.cm import ScalarMappable
        cbar = fig.colorbar(ScalarMappable(norm=im.norm, cmap=im.cmap),
                            cax=scale_ax, orientation="horizontal")
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=9, colors=muted, length=0, pad=6)
        fig.text(0.07, 0.125, 'RAIN INTENSITY · shaded map colours' if isinstance(marker, list) else colorbar_label,
                 fontsize=10, color=ink, weight='bold' if isinstance(marker, list) else 'normal')
        fig.text(0.93, 0.125, "LOW → HIGH", fontsize=9, color=muted, ha="right")
        plot_buffer = BytesIO()
        fig.savefig(plot_buffer, format="png", facecolor=background)
        plot_buffer.seek(0)
        return plot_buffer
    finally:
        plt.close(fig)



def build_radar_snapshot_plot(png_path, location=None):
    """Render the raw radar PNG at `png_path` using the same style as predictions."""
    radar_grid = remove_small_echoes(decode_png(png_path, SOURCE))
    radar_plot = np.flipud(radar_grid)
    marker = None
    if location and in_coverage(*map(float, location)):
        latitude, longitude = map(float, location)
        value, marker, radius = local_rain(radar_plot, latitude, longitude)

    tick = png_path.stem
    plot_buffer = render_heatmap(
        radar_plot,
        title=f"Raw Radar - {tick}",
        colorbar_label="Radar intensity",
        marker=marker,
    )

    observed = datetime.strptime(tick, '%Y%m%d%H%M')
    caption = rain_notice('RADAR SNAPSHOT', observed)
    if marker is not None:
        title = 'RAIN DETECTED' if value > 0.01 else 'NO RAIN DETECTED'
        caption = rain_notice(title, observed, radius=radius, observed_category=source_category(value))
        caption += '\nPink marker: your saved location'
    else:
        caption += "\nUse /start to save your location and show it on the map."
    return plot_buffer, caption


async def send_actual_fallback(update, folder_path, location):
    """Respond with a dated observation when no future +5 forecast is usable."""
    latest_png = await asyncio.to_thread(get_latest_radar_png, folder_path)
    if latest_png is None:
        await update.message.reply_text(
            "FORECAST DELAYED\n\nNo radar image is available yet. Please try again shortly.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    try:
        image, caption = await asyncio.to_thread(build_radar_snapshot_plot, latest_png, location)
    except (OSError, ValueError) as exc:
        print(f"Radar fallback unavailable: {exc}")
        await update.message.reply_text(
            "FORECAST DELAYED\n\nThe latest radar image could not be loaded. Please try again shortly.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    caption = "FORECAST DELAYED — SHOWING ACTUAL RADAR\n\n" + caption
    try:
        await update.message.reply_photo(photo=image, caption=caption, reply_markup=ReplyKeyboardRemove())
    finally:
        image.close()


async def handle_actual(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    folder_path
):
    latest_png = await asyncio.to_thread(get_latest_radar_png, folder_path)

    if latest_png is None:
        await update.message.reply_text("No radar images available yet.", reply_markup=ReplyKeyboardRemove())
        return

    location = await asyncio.to_thread(get_location, update.effective_chat.id)
    try:
        plot_buffer, caption = await asyncio.to_thread(build_radar_snapshot_plot, latest_png, location)
    except (OSError, ValueError) as exc:
        print(f"Actual radar unavailable: {exc}")
        await update.message.reply_text(
            "ACTUAL RADAR UNAVAILABLE\n\nThe latest radar image could not be decoded. Please try again shortly.",
            reply_markup=ReplyKeyboardRemove())
        return
    try:
        await update.message.reply_photo(photo=plot_buffer, caption=caption, reply_markup=ReplyKeyboardRemove())
    finally:
        plot_buffer.close()


def build_location_forecast(latitude, longitude, latest_prediction):
    pred_plot = latest_prediction["prediction"]
    next_tick = latest_prediction["next_tick"]

    rain_value_at_location, (pixel_x, pixel_y), radius = local_rain(pred_plot, latitude, longitude)

    plot_buffer = render_heatmap(
        pred_plot,
        title=f"Rain Prediction Heatmap - {next_tick}",
        colorbar_label="Prediction intensity",
        marker=(pixel_x, pixel_y),
    )

    # -----------------------------
    # Caption
    # -----------------------------

    rain_value_at_location_str = (
        f"{rain_value_at_location:.3g}"
    )

    caption = forecast_text(float(rain_value_at_location), latest_prediction, radius)

    return plot_buffer, caption


async def process_new_timestamp(
    context,
    timestamp,
    model,
    folder_path,
    norm_stats
):
    # Prevent processing the same timestamp twice
    last_timestamp = context.application.bot_data.get(
        "last_model_timestamp"
    )

    if timestamp == last_timestamp:
        print(f"{timestamp} already processed")
        return False

    print(f"New timestamp detected: {timestamp}")
    print("Running model...")

    result = await run_model(
        model,
        folder_path,
        norm_stats, tick=timestamp
    )

    if result is None:
        print("Model run failed")
        return False

    # Save latest prediction
    context.application.bot_data["latest_prediction"] = result

    # Mark timestamp as processed
    context.application.bot_data["last_model_timestamp"] = timestamp

    print(
        f"Prediction generated for "
        f"{result['next_tick']}"
    )

    # Send alerts to automatic users
    await send_auto_update(
        context,
        result
    )

    return True


async def check_model_queue(context, model, folder_path, norm_stats, model_ready_queue):
    pending = context.application.bot_data.setdefault("pending_model_ticks", set())
    while True:
        try:
            pending.add(int(model_ready_queue.get_nowait()))
        except queue.Empty:
            break
    for tick in sorted(pending.copy(), reverse=True):
        observed = datetime.strptime(str(tick), "%Y%m%d%H%M")
        deadline = observed + timedelta(minutes=5)
        if sg_now() >= deadline:
            pending.discard(tick)
            print(f"Expired model tick {tick}: forecast window has ended")
            continue
        if observed > sg_now():
            continue
        current = context.application.bot_data.get("latest_prediction")
        if current is not None and observed <= current["observed_at"]:
            pending.discard(tick)
            continue
        required = [observed - timedelta(minutes=5 * i) for i in range(SEQUENCE_LENGTH)]
        if not all((Path(folder_path) / f"{dt:%Y%m%d%H%M}.png").is_file() for dt in required):
            continue  # Wait for recent gaps to be repaired, only until the deadline.
        result = await run_model(model, folder_path, norm_stats, tick=tick)
        if sg_now() >= deadline:
            pending.discard(tick)
            print(f"Expired model tick {tick} during inference; not sending alerts")
            continue
        if result is None:
            print(f"Model run failed for {tick}; retained for retry")
            continue
        await send_auto_update(context, result)
        pending.discard(tick)
        current = context.application.bot_data.get("latest_prediction")
        if current is None or result["observed_at"] > current["observed_at"]:
            context.application.bot_data["latest_prediction"] = result
    # Retry durable notifications even when there are no newly arrived radar frames.
    await deliver_notifications(context)


async def send_auto_update(context, latest_prediction):
    actual = latest_prediction.get("actual_radar")
    if actual is None:
        return False
    rows = await asyncio.to_thread(get_rain_locations)
    grid = latest_prediction["prediction"]
    failed = False
    for row in rows:
        try:
            latitude, longitude = float(row["latitude"]), float(row["longitude"])
            if not in_coverage(latitude, longitude):
                continue
            forecast, marker, radius = local_rain(grid, latitude, longitude)
            observed, _, _ = local_rain(actual, latitude, longitude)
            result = next_rain_state(row, forecast, observed, latest_prediction["observed_at"], latest_prediction["next_tick"])
            if result is None:
                continue
            message = None
            photo = None
            if row['mode'] == 'automatic' and should_notify(row, result) and (result["reason"] == ENDED or is_fresh(latest_prediction)):
                is_actual = result['reason'] == ENDED
                value = observed if is_actual else forecast
                message = alert_text(result["reason"], result["rain_observed_at"], result["rain_forecast_at"], radius, value)
                message = f"{row.get('label', 'Main')}\n{message}"
                image_grid = actual if is_actual else grid
                image_time = result['rain_observed_at'] if is_actual else result['rain_forecast_at']
                source = 'Actual radar' if is_actual else 'Forecast radar (+5 min)'
                image = await asyncio.to_thread(render_heatmap, image_grid,
                                               f'{source} | {image_time:%d %b %H:%M} SGT',
                                               'Relative radar intensity', marker=marker)
                photo = image.getvalue()
                image.close()
            await asyncio.to_thread(save_rain_state, row, result, message, sg_now(), photo)
        except Exception as exc:
            failed = True
            print(f"Rain-state update failed for {row['userid']}: {exc}")
    if failed:
        raise RuntimeError("Some location states failed; keeping timestamp for retry")
    return True


def build_group_map(rows, grid, timestamp, actual=False):
    markers, lines = [], []
    colours = ("#e83288", "#0072b2", "#009e73", "#e69f00", "#7b3294", "#333333")
    for index, row in enumerate(rows):
        latitude, longitude = float(row['latitude']), float(row['longitude'])
        label = ' '.join(row['label'].split())
        if not in_coverage(latitude, longitude):
            lines.append(f'{label}: outside coverage')
            continue
        value, point, _ = local_rain(grid, latitude, longitude)
        markers.append((label, colours[index % len(colours)], point))
        raining = value > 0.01 if actual else value >= 0.003
        status = ('Rain detected' if raining else 'No rain detected') if actual else ('Rain expected' if raining else 'No rain expected')
        if raining:
            intensity = source_category(value) if actual else intensity_label(value)
            status += f' — {intensity.lower()} (radar scale)'
        lines.append(f'{label}: {status}')
    heading = 'Actual radar' if actual else 'Forecast (+5 min)'
    caption = f'{heading} | {timestamp:%d %b, %H:%M} SGT\n' + '\n'.join(lines)
    image = render_heatmap(grid, f'{heading} | {timestamp:%d %b %H:%M} SGT',
                           'Radar intensity' if actual else 'Prediction intensity', marker=markers)
    return image, caption


async def send_group_actual(update, folder_path, rows):
    latest = await asyncio.to_thread(get_latest_radar_png, folder_path)
    if latest is None:
        await update.message.reply_text('Forecast delayed. No radar image available yet.', reply_markup=ReplyKeyboardRemove())
        return
    try:
        image, caption = await asyncio.to_thread(build_group_actual, rows, latest)
    except (OSError, ValueError):
        await update.message.reply_text('Forecast delayed. Radar image unavailable; try again shortly.', reply_markup=ReplyKeyboardRemove())
        return
    try:
        await update.message.reply_photo(photo=image, caption='Forecast delayed — showing observations\n'+caption,
                                         reply_markup=ReplyKeyboardRemove())
    finally:
        image.close()


def build_group_actual(rows, path):
    grid = np.flipud(remove_small_echoes(decode_png(path, SOURCE)))
    return build_group_map(rows, grid, datetime.strptime(path.stem, '%Y%m%d%H%M'), actual=True)


async def handle_group_forecasts(update, context, model, folder_path, norm_stats=None):
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if not rows:
        await update.message.reply_text('Use /start to save your Main location first.', reply_markup=ReplyKeyboardRemove())
        return
    if len(rows) > 6:
        await update.message.reply_text('This chat has more than 6 locations. Use /removelocation to reduce it to 6 before requesting a map.', reply_markup=ReplyKeyboardRemove())
        return
    prediction = context.application.bot_data.get('latest_prediction')
    if not is_fresh(prediction):
        prediction = await run_model(model, folder_path, norm_stats)
    if not is_fresh(prediction):
        await send_group_actual(update, folder_path, rows)
        return
    context.application.bot_data['latest_prediction'] = prediction
    image, caption = await asyncio.to_thread(build_group_map, rows, prediction['prediction'], prediction['next_tick'])
    try:
        if is_fresh(prediction):
            await update.message.reply_photo(photo=image, caption=caption, reply_markup=ReplyKeyboardRemove())
        else:
            await send_group_actual(update, folder_path, rows)
    finally:
        image.close()
