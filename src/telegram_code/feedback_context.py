"""Freeze the location and weather shown to a user before collecting a report."""
import asyncio
from datetime import datetime
import json
import logging

import numpy as np

from telegram import ReplyKeyboardRemove
from telegram_code.feedback import FEEDBACK_PROMPT, prepare_feedback
from telegram_code.local_rain import in_coverage, local_rain

logger = logging.getLogger(__name__)


def location_row(location, label='Main'):
    return [{'latitude': float(location[0]), 'longitude': float(location[1]), 'label': label}] if location else []


def snapshot(row, kind, observed_at, target_at, radar_value=None, forecast_value=None,
             radius_m=None, source='automatic_notice'):
    """JSON-safe facts, including coordinates that later location edits cannot change."""
    return dict(kind=kind, latitude=float(row['latitude']), longitude=float(row['longitude']),
                label=row.get('label', 'Main'), location_id=row.get('location_id'),
                settings_version=row.get('rain_settings_version'),
                observed_at=observed_at.isoformat(), target_at=target_at.isoformat(),
                radar_value=None if radar_value is None else float(radar_value),
                forecast_value=None if forecast_value is None else float(forecast_value),
                radius_m=radius_m, source=source)


def _dated(item):
    return dict(item, observed_at=datetime.fromisoformat(item['observed_at']),
                target_at=datetime.fromisoformat(item['target_at']))


def _grid_snapshots(rows, kind, observed_at, target_at, radar=None, forecast=None):
    snapshots = []
    for index, row in enumerate(rows, 1):
        lat, lon = float(row['latitude']), float(row['longitude'])
        if not in_coverage(lat, lon):
            continue
        actual = local_rain(radar, lat, lon) if radar is not None else None
        predicted = local_rain(forecast, lat, lon) if forecast is not None else None
        value = predicted if kind == 'forecast' else actual
        snapshots.append(_dated(snapshot(
            row, kind, observed_at, target_at,
            radar_value=actual[0] if actual else None,
            forecast_value=predicted[0] if predicted else None,
            radius_m=value[2], source='manual_' + kind)))
        snapshots[-1]['display_index'] = index
    return snapshots


async def forecast_feedback(chat_id, rows, prediction):
    try:
        snapshots = await asyncio.to_thread(
            _grid_snapshots, rows, 'forecast', prediction['observed_at'], prediction['next_tick'],
            prediction.get('actual_radar'), prediction['prediction'])
        return await prepare_feedback(chat_id, snapshots)
    except Exception as exc:
        logger.warning('Forecast feedback unavailable: %s', type(exc).__name__)
        return None


def _radar_snapshots(rows, path):
    # Same decode, cleaning and display orientation used by the shown radar map.
    from data_processing.data_loading import remove_small_echoes
    from data_processing.radar_codec import decode_png, SOURCE
    grid = np.flipud(remove_small_echoes(decode_png(path, SOURCE)))
    observed = datetime.strptime(path.stem, '%Y%m%d%H%M')
    return _grid_snapshots(rows, 'radar', observed, observed, radar=grid)


async def radar_feedback(chat_id, rows, path):
    if not rows:
        return None  # A national map alone does not identify where the user is.
    try:
        snapshots = await asyncio.to_thread(_radar_snapshots, rows, path)
        return await prepare_feedback(chat_id, snapshots)
    except Exception as exc:
        logger.warning('Radar feedback unavailable: %s', type(exc).__name__)
        return None


async def notification_feedback(chat_id, items):
    snapshots = []
    for index, item in enumerate(items, 1):
        try:
            payload = json.loads(item.get('message') or '')
            if isinstance(payload, dict) and payload.get('rain_alert') == 1 and payload.get('feedback'):
                snapshots.append(_dated(payload['feedback']))
                snapshots[-1]['display_index'] = index
        except (ValueError, KeyError, TypeError):
            continue  # Legacy notices keep working without guessing a location.
    return await prepare_feedback(chat_id, snapshots) if snapshots else None


def with_feedback_prompt(caption, markup):
    return caption + '\n\n' + FEEDBACK_PROMPT if markup is not None else caption


async def reply_weather_photo(update, photo, caption, markup=None):
    """Keep long location labels intact and put buttons beneath their full text."""
    caption = with_feedback_prompt(caption, markup)
    keyboard = markup if markup is not None else ReplyKeyboardRemove()
    try:
        if len(caption.encode('utf-16-le')) // 2 <= 1024:
            return await update.message.reply_photo(photo=photo, caption=caption, reply_markup=keyboard)
        await update.message.reply_photo(photo=photo)
        return await update.message.reply_text(caption, reply_markup=keyboard)
    finally:
        photo.close()
