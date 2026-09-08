
from telegram import Update
from pathlib import Path
from telegram.ext import ContextTypes, ConversationHandler
from io import BytesIO
import os
import sys
import csv
from datetime import datetime, timedelta

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.data_loading import build_env_data, remove_small_echoes, build_and_cache_frame
from data_processing.pngtojson import points_to_intensity_grid, png_to_xy_intensity
from scraping.rain_areas import datetime_now_str, get_previous_ticks, SG_OFFSET_HOURS, attempt_get_most_recent
from masking import lat_long_to_pixel
from telegram_code.database import add_user
import pandas as pd
import numpy as np 
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy import ndimage

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


ENV_TICK_TOLERANCE_MINUTES = 15


def _find_usable_env_path(tick, height, width, tolerance_minutes=ENV_TICK_TOLERANCE_MINUTES):
	"""Return (env_path, env_stack) for `tick`, trying nearby ticks if needed.

	A CSV can exist but be unusable (e.g. the temperature API lagged and the
	scraper wrote a file with no temperature column), so file existence alone
	isn't enough — build the env stack and search outward in 5-minute steps
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
	user = update.effective_user.id
 
	success = add_user(user)	
	#success = True 
	print(success)
	if success:
		await update.message.reply_text("Please send current location")
	
	
	
	
	
async def handle_msg(update: Update , context: ContextTypes.DEFAULT_TYPE):
	user_id = update.effective_user.id
	user_name= update.effective_user.name
	text = update.message.text
	print(f"{user_id}-{user_name}, {text}")
	#await update.effective_sender.send_message("hfhfifhehfew")
	await update.message.reply_text("Fuck you mans calling..... i got bad news...")

async def handle_location(update: Update , context: ContextTypes.DEFAULT_TYPE,model,folder_path,norm_stats=None):
	user_id = update.effective_user.id
	user_name= update.effective_user.name
	location = update.message.location
	long= location.longitude
	lat = location.latitude
	print(f"Lat:{lat}, Long:{long}")
	success_most_recent = attempt_get_most_recent()
	dt_now= datetime_now_str(offset_hours=SG_OFFSET_HOURS)	
	prev_ticks= get_previous_ticks(dt_now, count=SEQUENCE_LENGTH, most_recent_success=success_most_recent)
	most_recent_tick = datetime.strptime(str(prev_ticks[0]), "%Y%m%d%H%M")
	next_tick = most_recent_tick + timedelta(minutes=5)
	#with open(r"rainraingoaway\userdata.csv", "r", newline="") as file:
	#	repr(file.read())
	# Build [1, T, 7, H, W] multimodal input for the gated ConvLSTM_MM model
	try:
		x = build_multimodal_input(prev_ticks, folder_path=folder_path, norm_stats=norm_stats)
	except (FileNotFoundError, ValueError) as exc:
		print(f"build_multimodal_input failed: {exc}")
		await update.message.reply_text(
			"Weather/radar data for the latest ticks isn't available yet — please try again in a few minutes."
		)
		return
 
	x = x.to(next(model.parameters()).device)
	with torch.no_grad():
		raw_prediction, last_radar, delta, rain_logits = model(x, return_logits=True)

	# Gated rain-prediction pipeline (rain prob used as a gate)
	raw_pred_plot = raw_prediction.detach().cpu().squeeze(0).squeeze(0).clamp_min(0)
	rain_probability = torch.sigmoid(rain_logits).detach().cpu().squeeze(0).squeeze(0)

	raw_np = raw_pred_plot.numpy()
	prob_np = rain_probability.numpy()

	cleaned_mask = clean_rain_mask(prob_np)              # boolean rain mask gate
	gated_prediction = raw_np * cleaned_mask             # final gated output

	# Gating images exposed as variables
	gating_rain_probability = prob_np
	gating_cleaned_mask = cleaned_mask
	gating_gated_prediction = gated_prediction

	pred_plot = np.flipud(gated_prediction)

	pixel_x, pixel_y = lat_long_to_pixel(
		lat=lat,
		long=long,
		width=pred_plot.shape[1],
		height=pred_plot.shape[0]
	)

	pixel_y = pred_plot.shape[0] - 1 - pixel_y

	print(f"x pixel: {pixel_x}")
	print(f"y pixel: {pixel_y}")
	rain_value_at_location = pred_plot[pixel_y, pixel_x]
	sg_base_img = np.flipud(plt.imread(str(Path(__file__).resolve().parents[2] / "sgbaseimg_70km.png")))
	clear_mask = pred_plot < 0.003
	print(f"Clear pixels below threshold: {np.count_nonzero(clear_mask)}")
	pred_alpha = np.where(clear_mask, 0.0, 0.78)
	# nicer plot
	fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
	ax.set_facecolor("white")
	ax.imshow(
		sg_base_img,
		origin="lower",
		extent=[0, pred_plot.shape[1] - 1, 0, pred_plot.shape[0] - 1],
		zorder=0
	)

	im = ax.imshow(
		pred_plot,
		 cmap="turbo",
		origin="lower",
		alpha=pred_alpha,
		norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0),
		interpolation="nearest",
		aspect="equal",
		zorder=1
	)

	# user location marker
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

	# title and labels
	ax.set_title(
		f"Rain Prediction Heatmap - {next_tick}",
		fontsize=13,
		weight="bold",
		pad=12
	)

	ax.set_xlabel("Pixel X")
	ax.set_ylabel("Pixel Y")

	# keep full image bounds
	ax.set_xlim(0, pred_plot.shape[1] - 1)
	ax.set_ylim(0, pred_plot.shape[0] - 1)

	# colorbar
	cbar = fig.colorbar(
		im,
		ax=ax,
		fraction=0.035,
		pad=0.025
	)

	cbar.set_label("Prediction intensity", rotation=270, labelpad=18)
	cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

	# legend outside the plot
	ax.legend(
		loc="lower left",
		bbox_to_anchor=(0.0, 1.02),
		frameon=False
	)

	plt.tight_layout()

	plot_buffer = BytesIO()
	plt.savefig(plot_buffer, format="png", bbox_inches="tight", facecolor="white")
	plot_buffer.seek(0)
	plt.close(fig)
	rain_value_at_location_str = f"{rain_value_at_location:.3g}"

	await update.message.reply_photo(
		photo=plot_buffer,
		caption=f"Rain prediction heatmap, predicted rain value at location: {rain_value_at_location_str}"
	)
		
 
    


async def set_loc(update: Update , context: ContextTypes.DEFAULT_TYPE):
    print("set loc attempted.")
    user_id= update.effective_user.id
    await update.message.reply_text("Send new current location")
    #time.sleep(1000)
    
    return 0 