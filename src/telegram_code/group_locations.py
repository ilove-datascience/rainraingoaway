"""Named extra locations for groups; location 0 remains the existing Main location."""
import asyncio
import secrets
from telegram import ForceReply, ReplyKeyboardRemove
from telegram.ext import ConversationHandler
from telegram_code.database import _get_db_connection
from telegram_code.local_rain import in_coverage
from telegram_code.notification_delivery import settings_lock

WAITING_FOR_EXTRA_LOCATION = 3
WAITING_FOR_LOCATION_NAME = 4
WAITING_FOR_REMOVAL_NAME = 5


def list_locations(chat_id):
    conn = _get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute('SELECT location_id,label,latitude,longitude FROM user_location WHERE userid=%s ORDER BY location_id', (chat_id,))
            return cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()


def add_named_location(chat_id, label, latitude, longitude):
    if chat_id >= 0:
        raise ValueError('Multiple locations are available in groups only.')
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SELECT userid FROM users WHERE userid=%s FOR UPDATE', (chat_id,))
            if not cur.fetchone():
                return False
            cur.execute('SELECT location_id FROM user_location WHERE userid=%s AND label=%s', (chat_id, label))
            if cur.fetchone():
                return False
            # IDs are never deliberately reused, so old queued events cannot match a re-added location.
            cur.execute('INSERT INTO user_location (userid,location_id,label,latitude,longitude) VALUES (%s,%s,%s,%s,%s)',
                        (chat_id, secrets.randbelow(2**63 - 1) + 1, label, latitude, longitude))
            conn.commit()
            return True
        finally:
            cur.close()
    finally:
        conn.close()


def remove_named_location(chat_id, label):
    if chat_id >= 0:
        return False
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SELECT userid FROM users WHERE userid=%s FOR UPDATE', (chat_id,))
            if not cur.fetchone():
                return False
            cur.execute('SELECT location_id FROM user_location WHERE userid=%s AND label=%s AND location_id<>0', (chat_id, label))
            row = cur.fetchone()
            if not row:
                return False
            cur.execute("UPDATE rain_notifications SET status='cancelled' WHERE userid=%s AND location_id=%s AND status='pending'", (chat_id, row[0]))
            cur.execute('DELETE FROM user_location WHERE userid=%s AND location_id=%s', (chat_id, row[0]))
            conn.commit()
            return True
        finally:
            cur.close()
    finally:
        conn.close()


async def add_location_command(update, context):
    return await begin_named_location(update, context, ' '.join(context.args or []).strip())


async def begin_named_location(update, context, label):
    if update.effective_chat.type not in ('group', 'supergroup'):
        await update.message.reply_text('Multiple locations are available in groups only.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    if not label or len(label) > 80:
        await update.message.reply_text('Use /addlocation followed by a name (up to 80 characters), for example /addlocation Office.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if not rows:
        await update.message.reply_text('Use /start to set up the group’s Main location first.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    if any(row['label'].casefold() == label.casefold() for row in rows):
        await update.message.reply_text('That location name is already saved. Choose another name.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    context.chat_data['new_location_label'] = label
    await update.message.reply_text(f'Send a Telegram location for {label}. Use /cancel to cancel.', reply_markup=ForceReply())
    return WAITING_FOR_EXTRA_LOCATION


async def receive_extra_location(update, context):
    label = context.chat_data.get('new_location_label')
    if not label:
        await update.message.reply_text('Use /addlocation followed by a name to begin again.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    location = update.message.location
    if not in_coverage(location.latitude, location.longitude):
        await update.message.reply_text('That location is outside radar coverage. Send another location or /cancel.', reply_markup=ReplyKeyboardRemove())
        return WAITING_FOR_EXTRA_LOCATION
    async with settings_lock(context):
        saved = await asyncio.to_thread(add_named_location, update.effective_chat.id, label, location.latitude, location.longitude)
    context.chat_data.pop('new_location_label', None)
    await update.message.reply_text(f'Saved {label}. It uses this group’s alert setting. Tap My forecast for all saved locations.' if saved
                                    else 'Could not add that location. Check /locations and try again with a unique name.',
                                    reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def locations_command(update, context):
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if not rows:
        await update.message.reply_text('No saved locations. Use /start to set up this chat.', reply_markup=ReplyKeyboardRemove())
        return
    # Separate short messages avoid Telegram’s message length limit.
    for row in rows:
        await update.message.reply_text(f"{row['label']}: {row['latitude']}, {row['longitude']}", reply_markup=ReplyKeyboardRemove())
    if update.effective_chat.type in ('group', 'supergroup'):
        await update.message.reply_text('Use /menu for Add location or Remove location. Change location updates Main.',
                                        reply_markup=ReplyKeyboardRemove())


async def remove_location_command(update, context):
    await remove_location_by_name(update, context, ' '.join(context.args or []).strip())


async def remove_location_by_name(update, context, label):
    if not label:
        await update.message.reply_text('Use /removelocation followed by the saved name.', reply_markup=ReplyKeyboardRemove())
        return
    async with settings_lock(context):
        removed = await asyncio.to_thread(remove_named_location, update.effective_chat.id, label)
    await update.message.reply_text(f'Removed {label} and cancelled its pending alerts.' if removed
                                    else 'Extra location not found. Use Saved locations. Main can be changed with Change location.',
                                    reply_markup=ReplyKeyboardRemove())


async def cancel_location(update, context):
    context.chat_data.pop('new_location_label', None)
    await update.message.reply_text('Setup cancelled.', reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def add_location_button(update, context):
    if update.effective_chat.type not in ('group', 'supergroup'):
        await update.message.reply_text('Multiple locations are available in groups only.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if not rows:
        await update.message.reply_text('Use /start to set up Main first.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    await update.message.reply_text('What should this location be called? Send a name, such as Office, or /cancel.',
                                    reply_markup=ForceReply())
    return WAITING_FOR_LOCATION_NAME


async def receive_location_name(update, context):
    result = await begin_named_location(update, context, update.message.text.strip())
    return WAITING_FOR_LOCATION_NAME if result == ConversationHandler.END else result


async def remove_location_button(update, context):
    if update.effective_chat.type not in ('group', 'supergroup'):
        await update.message.reply_text('Multiple locations are available in groups only.', reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    extras = [row for row in rows if row['location_id'] != 0]
    if not extras:
        await update.message.reply_text('No extra locations to remove. Change location updates Main.',
                                        reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    for row in extras:
        await update.message.reply_text(row['label'], reply_markup=ReplyKeyboardRemove())
    await update.message.reply_text('Send the name of the location to remove, or /cancel.', reply_markup=ForceReply())
    return WAITING_FOR_REMOVAL_NAME


async def receive_removal_name(update, context):
    await remove_location_by_name(update, context, update.message.text.strip())
    return ConversationHandler.END
