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
from masking import lat_long_to_pixel
from scraping.rain_areas import SG_OFFSET_HOURS, attempt_get_most_recent, datetime_now_str, get_previous_ticks
from telegram_code.database import add_location, add_user, get_autoupdate_users, get_location, save_mode_choice
from telegram_code.states import WAITING_FOR_LOCATION, WAITING_FOR_MODE
from telegram_code.forecast_policy import is_fresh, forecast_text, sg_now
from telegram_code.rain_state import next_rain_state, should_notify
from telegram_code.rain_state_db import get_rain_locations, save_rain_state
from telegram_code.notification_delivery import deliver_notifications, settings_lock
from telegram_code.notification_text import alert_text, rain_notice
from telegram_code.local_rain import local_rain, in_coverage
from telegram_code.rain_state import ENDED
from telegram_code.database import get_user_mode


def main_menu():
    return ReplyKeyboardMarkup(
        [["My forecast", "Current radar"], ["Change location", "Alert settings"]],
        resize_keyboard=True,
    )
# Gated rain-prediction pipeline constants (mirror test_multimodal_convlstm.ipynb).
# threshold=0.55 chosen as the best structural operating point from the validation-only
# mask sweep: cleanest background/noise rejection with best large-component IoU among
# configs within ~0.001 CSI of the peak (see mask_extended_validation_report.csv).
SEQUENCE_LENGTH = 3
RAIN_PROBABILITY_THRESHOLD = 0.55
MIN_COMPONENT_SIZE = 25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RADAR_FOLDER = PROJECT_ROOT / "data" / "70km" / "png"
ENV_FOLDER = PROJECT_ROOT / "data" / "environment"


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Channel indices (match multimodal_radar_dataset.channel_map)
CH_RADAR, CH_TEMP, CH_HUM, CH_WIND_U, CH_WIND_V, CH_STATION, CH_DIST = range(7)
ZSCORE_CHANNELS = [CH_TEMP, CH_HUM, CH_WIND_U, CH_WIND_V]


ENV_TICK_TOLERANCE_MINUTES = 15

def clean_rain_mask(probability: np.ndarray) -> np.ndarray:
	"""Threshold the rain-probability map and drop small components."""
	hard_mask = probability >= RAIN_PROBABILITY_THRESHOLD
	labels, _ = ndimage.label(hard_mask, structure=np.ones((3, 3), dtype=np.uint8))
	component_sizes = np.bincount(labels.ravel())
	keep = component_sizes >= MIN_COMPONENT_SIZE
	keep[0] = False
	return keep[labels]


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
    userid = update.effective_user.id
    location = await asyncio.to_thread(get_location, userid)
    if location:
        await update.message.reply_text("Welcome back. Choose a forecast or update your settings.", reply_markup=main_menu())
        return ConversationHandler.END
    await asyncio.to_thread(add_user, userid)
    await update.message.reply_text("Welcome! Send your Telegram location to set up local rain forecasts.")
    return WAITING_FOR_LOCATION


async def change_location(update, context):
    await update.message.reply_text("Send your new Telegram location. This will be used for automatic alerts.")
    return WAITING_FOR_LOCATION


async def receive_location(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    userid = update.effective_user.id
    location = update.message.location

    latitude = location.latitude
    longitude = location.longitude

    print(userid, latitude, longitude)
    
    if not in_coverage(latitude, longitude):
        await update.message.reply_text("That location is outside our radar coverage. Please send a location within Singapore's radar map.")
        return WAITING_FOR_LOCATION
    async with settings_lock(context):
        success = await asyncio.to_thread(add_location, userid, lat=latitude, long=longitude)
    if not success:
        await update.message.reply_text("Couldn't save your location. Please send it again.")
        return WAITING_FOR_LOCATION
    context.application.bot_data.setdefault("alert_history", {}).pop(userid, None)
    print("rcv lcoation called ")
    await update.message.reply_text("Location updated/saved")
    
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
    userid = update.effective_user.id
    current_mode = await asyncio.to_thread(get_user_mode, userid)
    if current_mode is None:
        await update.message.reply_text("Use /start to set up your location first.")
        return ConversationHandler.END
    await update.message.reply_text(f"Current alert setting: {current_mode}.")
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
    userid = update.effective_user.id
    choice = update.message.text

    if choice == "1 - Automatic rain updates":
        mode = "automatic"

    elif choice == "2 - Manual updates only":
        mode = "manual"

    else:
        await update.message.reply_text(
            "Please select one of the options below."
        )
        return WAITING_FOR_MODE

    print(userid, mode)
    async with settings_lock(context):
        success = await asyncio.to_thread(save_mode_choice, userid, mode)
    if success:
        print('mode updated')
        context.application.bot_data.setdefault("alert_history", {}).pop(userid, None)
        message = ("Automatic alerts enabled: one when rain is predicted, then only when "
                   "rain is predicted to end or radar shows it has ended." if mode == "automatic"
                   else "Automatic alerts paused. Tap My forecast whenever you need an update.")
        await update.message.reply_text(message, reply_markup=main_menu())
    else:
        await update.message.reply_text("Couldn't save your settings. Please try again.")
        return WAITING_FOR_MODE
    

    return ConversationHandler.END


async def handle_msg(update, context, model, folder_path, norm_stats=None):
    if update.message.text == "Current radar":
        await handle_actual(update, context, folder_path)
        return
    if update.message.text != "My forecast":
        await update.message.reply_text("Choose an option below, or share a location for a forecast.", reply_markup=main_menu())
        return
    location = await asyncio.to_thread(get_location, update.effective_user.id)
    if not location:
        await update.message.reply_text("Use /start to save your location first.", reply_markup=main_menu())
        return
    if not in_coverage(float(location[0]), float(location[1])):
        await update.message.reply_text("Your saved location is outside radar coverage. Tap Change location.", reply_markup=main_menu())
        return
    prediction = context.application.bot_data.get("latest_prediction")
    if not is_fresh(prediction):
        await update.message.reply_text("Checking for a current forecast…")
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
    await update.message.reply_photo(photo=image, caption=caption, reply_markup=main_menu())


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
        await update.message.reply_text("That location is outside our radar coverage.", reply_markup=main_menu())
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
        reply_markup=main_menu(),
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
    """Render `grid` (values in [0, 1]) over the Singapore base map.

    Shared plot style for both model predictions and raw radar snapshots.
    """
    sg_base_img = np.flipud(
        plt.imread(
            str(Path(__file__).resolve().parents[2] / "sgbaseimg_70km.png")
        )
    )

    clear_mask = grid < 0.003
    alpha = np.where(clear_mask, 0.0, 0.78)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    ax.set_facecolor("white")

    ax.imshow(
        sg_base_img,
        origin="lower",
        extent=[0, grid.shape[1] - 1, 0, grid.shape[0] - 1],
        zorder=0
    )

    im = ax.imshow(
        grid,
        cmap="turbo",
        origin="lower",
        alpha=alpha,
        norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0),
        interpolation="nearest",
        aspect="equal",
        zorder=1
    )

    if marker is not None:
        pixel_x, pixel_y = marker
        ax.scatter(
            pixel_x,
            pixel_y,
            c="hotpink",
            s=110,
            edgecolors="white",
            linewidths=2,
            zorder=3,
            label="User location"
        )

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_xlim(0, grid.shape[1] - 1)
    ax.set_ylim(0, grid.shape[0] - 1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(colorbar_label, rotation=270, labelpad=18)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    if marker is not None:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False)

    plt.tight_layout()

    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png", bbox_inches="tight", facecolor="white")
    plot_buffer.seek(0)
    plt.close(fig)

    return plot_buffer


def build_radar_snapshot_plot(png_path, location=None):
    """Render the raw radar PNG at `png_path` using the same style as predictions."""
    intensity_points = png_to_xy_intensity(str(png_path), include_zero=True)
    intensity_grid = points_to_intensity_grid(intensity_points)
    radar_grid = remove_small_echoes(np.asarray(intensity_grid, dtype=np.float32) / 100.0)
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
        caption = rain_notice(title, observed, radius=radius, intensity=value)
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
            reply_markup=main_menu(),
        )
        return
    try:
        image, caption = await asyncio.to_thread(build_radar_snapshot_plot, latest_png, location)
    except (OSError, ValueError) as exc:
        print(f"Radar fallback unavailable: {exc}")
        await update.message.reply_text(
            "FORECAST DELAYED\n\nThe latest radar image could not be loaded. Please try again shortly.",
            reply_markup=main_menu(),
        )
        return
    caption = "FORECAST DELAYED — SHOWING ACTUAL RADAR\n\n" + caption
    try:
        await update.message.reply_photo(photo=image, caption=caption, reply_markup=main_menu())
    finally:
        image.close()


async def handle_actual(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    folder_path
):
    latest_png = await asyncio.to_thread(get_latest_radar_png, folder_path)

    if latest_png is None:
        await update.message.reply_text("No radar images available yet.")
        return

    location = await asyncio.to_thread(get_location, update.effective_user.id)
    plot_buffer, caption = await asyncio.to_thread(build_radar_snapshot_plot, latest_png, location)

    await update.message.reply_photo(photo=plot_buffer, caption=caption, reply_markup=main_menu())


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
    for tick in sorted(pending.copy()):
        result = await run_model(model, folder_path, norm_stats, tick=tick)
        if result is None:
            print(f"Model run failed for {tick}; retained for retry")
            continue
        await send_auto_update(context, result)
        pending.discard(tick)
        current = context.application.bot_data.get("latest_prediction")
        if current is None or result["observed_at"] > current["observed_at"]:
            context.application.bot_data["latest_prediction"] = result
    # Retry durable notifications even when there are no newly arrived radar frames.
    await deliver_notifications(context, reply_markup=main_menu())


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
