import asyncio
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
from telegram_code.notification_text import encode_alert, rain_notice, intensity_label
from telegram_code.local_rain import local_rain, in_coverage
from telegram_code.rain_state import ENDED, OBSERVED, START
from telegram_code.database import get_user_mode
from telegram_code.feedback_context import (
    forecast_feedback, radar_feedback, location_row, reply_weather_photo, snapshot,
)


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
        message = ("Automatic alerts enabled: I'll let you know when rain is expected or detected, "
                   "then when radar confirms it has cleared. Location updates are grouped together." if mode == "automatic"
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
        await send_actual_fallback(update, folder_path, location)
        return
    context.application.bot_data["latest_prediction"] = prediction
    image, caption = await asyncio.to_thread(build_location_forecast, float(location[0]), float(location[1]), prediction)
    markup = await forecast_feedback(update.effective_chat.id, location_row(location), prediction)
    if not is_fresh(prediction):
        image.close()
        await send_actual_fallback(update, folder_path, location)
        return
    await reply_weather_photo(update, image, caption, markup)


# Keep inference off Telegram's loop and short I/O workers; serialize model access.
_model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="radar-model")


async def run_model(model, folder_path, norm_stats=None, tick=None):
    return await asyncio.get_running_loop().run_in_executor(
        _model_executor, partial(_run_model_sync, model, folder_path, norm_stats, tick)
    )


def _run_model_sync(model, folder_path, norm_stats=None, tick=None):
    """
    Run the rainfall model using the latest available radar/environment data.

    Returns:
        dict containing prediction data, or None if input data is unavailable.
    """

    try:
        if tick is None:
            latest = get_latest_radar_png(folder_path)
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
        x = build_multimodal_input(
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
            # Observations must use exact source levels, not resized model inputs.
            "actual_radar": np.flipud(remove_small_echoes(decode_png(
                Path(folder_path) / f"{prev_ticks[0]}.png", SOURCE))),

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
        await send_actual_fallback(update, folder_path, (latitude, longitude), label='Shared location')
        return

    plot_buffer, caption = await asyncio.to_thread(
        build_location_forecast,
        latitude,
        longitude,
        latest_prediction
    )
    markup = await forecast_feedback(update.effective_chat.id,
                                     location_row((latitude, longitude), 'Shared location'), latest_prediction)

    if not is_fresh(latest_prediction):
        plot_buffer.close()
        await send_actual_fallback(update, folder_path, (latitude, longitude), label='Shared location')
        return
    await reply_weather_photo(update, plot_buffer, caption, markup)


def get_latest_radar_png(folder_path) -> Path | None:
    """Return the most recently captured raw radar PNG, or None if there are none."""
    return max(Path(folder_path).glob("*.png"), default=None)


_plot_lock = threading.Lock()


def render_heatmap(grid, title, colorbar_label, marker=None, marker_numbers=None):
    # Matplotlib's pyplot state is shared by manual requests and queued alerts.
    with _plot_lock:
        return _render_heatmap(grid, title, colorbar_label, marker, marker_numbers)


def _render_heatmap(grid, title, colorbar_label, marker=None, marker_numbers=None):
    """Render a phone-friendly weather card with unchanged radar coordinates."""
    sg_base_img = np.flipud(plt.imread(
        str(Path(__file__).resolve().parents[2] / "sgbaseimg_70km.png")))
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MarkerPath
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.text import Text
    background, ink, muted = "#f4f7fb", "#17304c", "#60748b"
    multiple = isinstance(marker, list)
    # A 1600px-wide card keeps phone text sharp while limiting upload size.
    # Radar cells retain their original values and nearest-neighbour display.
    fig = plt.figure(figsize=(8, 9) if multiple else (8, 7), dpi=200, facecolor=background)
    try:
        heading, separator, timestamp = title.partition(" | ")
        fig.text(0.06, 0.96, "RAINRAINGOAWAY", fontsize=10,
                 weight="bold", color=muted)
        forecast = 'forecast' in heading.lower() or 'prediction' in heading.lower()
        badge = 'FORECAST' if forecast else 'RADAR OBSERVATION'
        badge_colour = '#5640a0' if forecast else '#11675e'
        fig.text(0.94, 0.96, badge, ha='right', fontsize=8, weight='bold',
                 color=badge_colour,
                 bbox=dict(boxstyle='round,pad=0.55', facecolor='#eee9fa' if forecast else '#dff2ed',
                           edgecolor='none'))
        fig.text(0.06, 0.905, heading, fontsize=24, weight="bold", color=ink)
        if separator:
            fig.text(0.06, 0.873, timestamp, fontsize=12, color=muted)
        ax = fig.add_axes([0.045, 0.40 if multiple else 0.20, 0.91,
                           0.45 if multiple else 0.64])
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
        if multiple:
            fig.text(0.06, 0.365, 'YOUR SAVED PLACES', fontsize=11, weight='bold', color=ink)
            fig.text(0.94, 0.365, f'{len(marker)} on map', fontsize=10, color=muted, ha='right')
            numbers = marker_numbers if marker_numbers is not None else range(1, len(marker) + 1)
            if len(numbers) != len(marker):
                raise ValueError('Each location marker needs a matching display number')
            for index, (label, colour, (pixel_x, pixel_y)) in enumerate(marker):
                number = numbers[index]
                # The tip, rather than the centre of the number, marks the
                # exact radar coordinate. Point-sized vector artwork stays crisp.
                pin = DrawingArea(22, 28, 0, 0)
                outline = MarkerPath(
                    [(11, 0), (7, 6), (5, 6), (0, 6), (0, 11),
                     (0, 23), (0, 28), (5, 28), (17, 28),
                     (22, 28), (22, 23), (22, 11), (22, 6), (17, 6),
                     (15, 6), (11, 0), (11, 0)],
                    [MarkerPath.MOVETO, MarkerPath.LINETO, MarkerPath.LINETO,
                     MarkerPath.CURVE3, MarkerPath.CURVE3, MarkerPath.LINETO,
                     MarkerPath.CURVE3, MarkerPath.CURVE3, MarkerPath.LINETO,
                     MarkerPath.CURVE3, MarkerPath.CURVE3, MarkerPath.LINETO,
                     MarkerPath.CURVE3, MarkerPath.CURVE3, MarkerPath.LINETO,
                     MarkerPath.LINETO, MarkerPath.CLOSEPOLY])
                pin.add_artist(PathPatch(outline, facecolor='#155c65', edgecolor='white',
                                         linewidth=1.1, antialiased=True, joinstyle='round'))
                pin.add_artist(Text(11, 17, str(number), color='white', ha='center',
                                    va='center', fontsize=10, weight='medium'))
                ax.add_artist(AnnotationBbox(pin, (pixel_x, pixel_y), xycoords='data',
                    box_alignment=(0.5, 0), frameon=False, pad=0,
                    annotation_clip=False, zorder=5))
                column, row = index % 2, index // 2
                x, y = 0.06 + column * 0.46, 0.315 - row * 0.062
                fig.add_artist(FancyBboxPatch((x, y - 0.0225), 0.42, 0.045,
                    boxstyle='round,pad=0.005,rounding_size=0.009',
                    transform=fig.transFigure, facecolor='white', edgecolor='#e1e8f0', linewidth=0.8))
                tile_w, tile_h = 18 / (72 * fig.get_figwidth()), 18 / (72 * fig.get_figheight())
                fig.add_artist(FancyBboxPatch((x + 0.025 - tile_w / 2, y - tile_h / 2),
                    tile_w, tile_h, boxstyle='round,pad=0,rounding_size=0.007',
                    transform=fig.transFigure, facecolor='#e6f2ef', edgecolor='none'))
                fig.text(x + 0.025, y, str(number), ha='center', va='center', fontsize=9.5,
                         weight='medium', color='#155c65')
                short_label = label if len(label) <= 27 else label[:26] + '…'
                fig.text(x + 0.06, y, short_label, va='center', fontsize=11, color=ink)
            fig.text(0.06, 0.141, 'Numbers identify places. Map shading shows rain.', fontsize=10, color=muted)
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
        scale_ax = fig.add_axes([0.06, 0.065 if multiple else 0.08, 0.88, 0.014])
        # Use an opaque legend even though rain is blended over the map.
        from matplotlib.cm import ScalarMappable
        cbar = fig.colorbar(ScalarMappable(norm=im.norm, cmap=im.cmap),
                            cax=scale_ax, orientation="horizontal")
        cbar.set_ticks([0.0, 1.0], labels=['Lower', 'Higher'])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=9, colors=muted, length=0, pad=6)
        fig.text(0.06, 0.104 if multiple else 0.12, 'RELATIVE RADAR INTENSITY',
                 fontsize=9, color=ink, weight='bold')
        fig.text(0.94, 0.104 if multiple else 0.12, 'Radar colour scale', fontsize=9, color=muted, ha='right')
        plot_buffer = BytesIO()
        fig.savefig(plot_buffer, format="jpeg", facecolor=background,
                    pil_kwargs={"quality": 92, "subsampling": 0, "optimize": True})
        plot_buffer.name = "radar.jpg"
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
    observed = datetime.strptime(tick, '%Y%m%d%H%M')
    plot_buffer = render_heatmap(
        radar_plot,
        title=f"Actual radar | {observed:%d %b %H:%M} SGT",
        colorbar_label="Radar intensity",
        marker=marker,
    )

    caption = rain_notice('RADAR SNAPSHOT', observed)
    if marker is not None:
        title = 'RAIN DETECTED' if value > 0.01 else 'NO RAIN DETECTED'
        caption = rain_notice(title, observed, radius=radius, observed_category=source_category(value))
        caption += '\nPink marker: your saved location'
    else:
        caption += "\nUse /start to save your location and show it on the map."
    return plot_buffer, caption


async def send_actual_fallback(update, folder_path, location, label='Main'):
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
    markup = await radar_feedback(update.effective_chat.id, location_row(location, label), latest_png)
    try:
        await reply_weather_photo(update, image, caption, markup)
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

    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if len(rows) > 6:
        await update.message.reply_text('This chat has more than 6 locations. Use /removelocation to reduce it to 6 before requesting a map.', reply_markup=ReplyKeyboardRemove())
        return
    try:
        if rows:
            plot_buffer, caption = await asyncio.to_thread(build_group_actual, rows, latest_png)
        else:
            plot_buffer, caption = await asyncio.to_thread(build_radar_snapshot_plot, latest_png, None)
    except (OSError, ValueError) as exc:
        print(f"Actual radar unavailable: {exc}")
        await update.message.reply_text(
            "ACTUAL RADAR UNAVAILABLE\n\nThe latest radar image could not be decoded. Please try again shortly.",
            reply_markup=ReplyKeyboardRemove())
        return
    try:
        markup = await radar_feedback(update.effective_chat.id, rows, latest_png)
        await reply_weather_photo(update, plot_buffer, caption, markup)
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
        pending.discard(tick)
        current = context.application.bot_data.get("latest_prediction")
        if current is None or result["observed_at"] > current["observed_at"]:
            context.application.bot_data["latest_prediction"] = result
        await send_auto_update(context, result)
    # Retry durable notifications even when there are no newly arrived radar frames.
    await deliver_notifications(context)


async def send_auto_update(context, latest_prediction):
    lock = context.application.bot_data.setdefault("rain_state_lock", asyncio.Lock())
    async with lock:
        return await _send_auto_update(context, latest_prediction)


async def _send_auto_update(context, latest_prediction):
    actual = latest_prediction.get("actual_radar")
    if actual is None:
        return False
    rows = await asyncio.to_thread(get_rain_locations)
    grid = latest_prediction.get("prediction")
    failed = False
    updates = []
    photos = {}
    for row in rows:
        try:
            latitude, longitude = float(row["latitude"]), float(row["longitude"])
            if not in_coverage(latitude, longitude):
                continue
            observed, marker, radius = local_rain(actual, latitude, longitude)
            forecast = local_rain(grid, latitude, longitude)[0] if grid is not None else None
            result = next_rain_state(row, forecast, observed, latest_prediction["observed_at"], latest_prediction["next_tick"])
            if result is None:
                continue
            message = None
            now = sg_now()
            fresh_actual = result['rain_observed_at'] <= now < result['rain_observed_at'] + timedelta(minutes=10)
            fresh = is_fresh(latest_prediction) if result['reason'] == START else fresh_actual
            if row['mode'] == 'automatic' and should_notify(row, result) and fresh:
                category = (source_category(observed) if result['reason'] == OBSERVED else
                            intensity_label(forecast) if result['reason'] == START else None)
                scope = sorted([saved.get('location_id', 0), saved['rain_settings_version']]
                               for saved in rows if saved['userid'] == row['userid'])
                message = encode_alert(row.get('label', 'Main'), result['reason'], category,
                                       photo_locations=scope,
                                       feedback=snapshot(
                                           row, 'forecast' if result['reason'] == START else 'radar',
                                           result['rain_observed_at'],
                                           result['rain_forecast_at'] if result['reason'] == START else result['rain_observed_at'],
                                           radar_value=observed, forecast_value=forecast, radius_m=radius))
            updates.append((row, result, message))
        except Exception as exc:
            failed = True
            print(f"Rain-state update failed for {row['userid']}: {exc}")
    # One observation map per chat, with every saved location labelled. Sharing
    # the image lets delivery combine only the events still eligible to send.
    for row, result, message in updates:
        if not message or row['userid'] in photos:
            continue
        try:
            chat_rows = sorted([saved for saved in rows if saved['userid'] == row['userid']],
                               key=lambda saved: saved.get('location_id', 0))
            image, _ = await asyncio.to_thread(build_group_map, chat_rows, actual,
                                                result['rain_observed_at'], actual=True)
            try:
                photos[row['userid']] = image.getvalue()
            finally:
                image.close()
        except Exception as exc:
            failed = True
            photos[row['userid']] = None
            print(f"Rain-map rendering failed for {row['userid']}: {type(exc).__name__}")
    # Start the shared collection window after rendering, which can be slow.
    available_at = sg_now() + timedelta(seconds=15)
    for row, result, message in updates:
        if message and photos.get(row['userid']) is None:
            continue  # Keep this observation retryable rather than losing its alert.
        try:
            await asyncio.to_thread(save_rain_state, row, result, message, available_at,
                                    photos.get(row['userid']) if message else None)
        except Exception as exc:
            failed = True
            print(f"Rain-state save failed for {row['userid']}: {type(exc).__name__}")
    if failed:
        raise RuntimeError("Some location states failed; keeping timestamp for retry")
    return True


def build_group_map(rows, grid, timestamp, actual=False):
    markers, lines, marker_numbers = [], [], []
    colours = ("#e83288", "#0072b2", "#009e73", "#e69f00", "#7b3294", "#333333")
    for index, row in enumerate(rows):
        latitude, longitude = float(row['latitude']), float(row['longitude'])
        label = ' '.join(row['label'].split())
        if not in_coverage(latitude, longitude):
            lines.append(f'{index + 1}. {label}: outside coverage')
            continue
        value, point, _ = local_rain(grid, latitude, longitude)
        markers.append((label, colours[index % len(colours)], point))
        marker_numbers.append(index + 1)
        raining = value > 0.01 if actual else value >= 0.003
        status = ('Rain detected' if raining else 'No rain detected') if actual else ('Rain expected' if raining else 'No rain expected')
        if raining:
            intensity = source_category(value) if actual else intensity_label(value)
            status += f' — {intensity.lower()} (radar scale)'
        lines.append(f'{index + 1}. {label}: {status}')
    heading = 'Actual radar' if actual else 'Forecast (+5 min)'
    caption = f'{heading} | {timestamp:%d %b, %H:%M} SGT\n\n' + '\n'.join(lines)
    image = render_heatmap(grid, f'{heading} | {timestamp:%d %b %H:%M} SGT',
                           'Radar intensity' if actual else 'Prediction intensity',
                           marker=markers, marker_numbers=marker_numbers)
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
    markup = await radar_feedback(update.effective_chat.id, rows, latest)
    try:
        await reply_weather_photo(update, image, 'Forecast delayed — showing observations\n'+caption, markup)
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
        await send_group_actual(update, folder_path, rows)
        return
    context.application.bot_data['latest_prediction'] = prediction
    image, caption = await asyncio.to_thread(build_group_map, rows, prediction['prediction'], prediction['next_tick'])
    markup = await forecast_feedback(update.effective_chat.id, rows, prediction)
    try:
        if is_fresh(prediction):
            await reply_weather_photo(update, image, caption, markup)
        else:
            await send_group_actual(update, folder_path, rows)
    finally:
        image.close()


async def check_radar_notifications(context, folder_path):
    """Observed rain needs one fresh frame, independent of model/weather success."""
    try:
        latest = await asyncio.to_thread(get_latest_radar_png, folder_path)
        if latest is None:
            return
        observed = datetime.strptime(latest.stem, '%Y%m%d%H%M')
        if not observed <= sg_now() < observed + timedelta(minutes=10):
            return
        radar = await asyncio.to_thread(decode_png, latest, SOURCE)
        radar = np.flipud(remove_small_echoes(radar))
        await send_auto_update(context, {
            'actual_radar': radar, 'observed_at': observed,
            'next_tick': observed + timedelta(minutes=10),
        })
    except Exception as exc:
        print(f'Observed-rain notification check failed: {type(exc).__name__}')
    finally:
        await deliver_notifications(context)
