
from telegram import Update, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton, PhotoSize
from pathlib import Path
from telegram.ext import ContextTypes
from io import BytesIO
import os
from datetime import datetime, timedelta, timezone
from data_processing.data_loading import load_specific_data
from scraping.rain_areas import datetime_now_str,get_previous_ticks ,SG_OFFSET_HOURS, attempt_get_most_recent
import pandas as pd
import numpy as np 
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from masking import lat_long_to_pixel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
	
	
async def handle_msg(update: Update , context: ContextTypes.DEFAULT_TYPE):
	user_id = update.effective_user.id
	user_name= update.effective_user.name
	text = update.message.text
	print(f"{user_id}-{user_name}, {text}")
	#await update.effective_sender.send_message("hfhfifhehfew")
	await update.message.reply_text("Fuck you mans calling..... i got bad news...")

async def handle_location(update: Update , context: ContextTypes.DEFAULT_TYPE,model,folder_path,):
	user_id = update.effective_user.id
	user_name= update.effective_user.name
	location = update.message.location
	long= location.longitude
	lat = location.latitude
	print(f"Lat:{lat}, Long:{long}")
	success_most_recent = attempt_get_most_recent()
	dt_now= datetime_now_str(offset_hours=SG_OFFSET_HOURS)	
	prev_ticks= get_previous_ticks(dt_now, most_recent_success=success_most_recent)
	most_recent_tick = datetime.strptime(str(prev_ticks[0]), "%Y%m%d%H%M")
	next_tick = most_recent_tick + timedelta(minutes=5)
	
	data = load_specific_data(file_names = prev_ticks, folder_path= folder_path)
	print(type(data))
	frames = []
	for frame in data:
		if isinstance(frame, pd.DataFrame):
			arr = frame.to_numpy(dtype=np.float32)
		else:
			arr = np.asarray(frame, dtype=np.float32)

		#print("frame shape:", arr.shape)

		# Optional safety check
		if arr.shape != (120, 217):
			raise ValueError(f"Bad frame shape: {arr.shape}, expected (120, 217)")

		frames.append(arr)
  
	# [T, H, W] = [4, 120, 217]
	x = np.stack(frames, axis=0)

	# Convert to torch tensor
	x = torch.from_numpy(x).float().to(device)
	x = x.unsqueeze(1)
	x = x.unsqueeze(0)
	#print(x.shape)
 
	with torch.no_grad():
		pred = model(x)
	

	pred_plot = pred.detach().cpu().squeeze()
	if pred_plot.ndim > 2:
		pred_plot = pred_plot[0]

	# flip plot on horizontal axis 
	pred_plot = np.flipud(pred_plot.numpy())

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
		
 
    

