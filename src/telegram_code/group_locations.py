"""Named extra locations for private and group chats; location 0 remains the existing Main location."""
import asyncio
import secrets
from telegram import ForceReply
from telegram.ext import ConversationHandler
from telegram_code.database import _get_db_connection
from telegram_code.local_rain import in_coverage
from telegram_code.notification_delivery import settings_lock

MAX_CHAT_LOCATIONS = 6

WAITING_FOR_EXTRA_LOCATION = 3
WAITING_FOR_LOCATION_NAME = 4
WAITING_FOR_REMOVAL_NAME = 5


def list_locations(chat_id):
    conn = _get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute('SELECT location_id,label,latitude,longitude,alerts_enabled FROM user_location WHERE userid=%s ORDER BY location_id', (chat_id,))
            return cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()


def add_named_location(chat_id, label, latitude, longitude):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SELECT mode FROM users WHERE userid=%s FOR UPDATE', (chat_id,))
            user = cur.fetchone()
            if not user:
                return False
            cur.execute('SELECT location_id FROM user_location WHERE userid=%s AND label=%s', (chat_id, label))
            if cur.fetchone():
                return False
            cur.execute('SELECT COUNT(*) FROM user_location WHERE userid=%s', (chat_id,))
            if cur.fetchone()[0] >= MAX_CHAT_LOCATIONS:
                return False
            # A legacy global pause stays off for existing locations. Only the new one starts on.
            if user[0] != 'automatic':
                cur.execute('''UPDATE user_location SET alerts_enabled=0,
                    rain_settings_version=rain_settings_version+1,
                    rain_episode_reason=NULL,rain_alert_reason=NULL,rain_dry_since=NULL,rain_episode_wet=0 WHERE userid=%s''', (chat_id,))
                cur.execute("UPDATE rain_notifications SET status='cancelled' WHERE userid=%s AND status='pending'", (chat_id,))
                cur.execute("UPDATE users SET mode='automatic' WHERE userid=%s", (chat_id,))
            # IDs are never deliberately reused, so old queued events cannot match a re-added location.
            cur.execute('INSERT INTO user_location (userid,location_id,label,latitude,longitude,alerts_enabled) VALUES (%s,%s,%s,%s,%s,1)',
                        (chat_id, secrets.randbelow(2**63 - 1) + 1, label, latitude, longitude))
            conn.commit()
            return True
        finally:
            cur.close()
    finally:
        conn.close()


def remove_named_location(chat_id, label, expected_location_id=None):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute('SELECT userid FROM users WHERE userid=%s FOR UPDATE', (chat_id,))
            if not cur.fetchone():
                return False
            cur.execute('SELECT location_id FROM user_location WHERE userid=%s AND label=%s AND location_id<>0', (chat_id, label))
            row = cur.fetchone()
            if not row or (expected_location_id is not None and row[0] != expected_location_id):
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
    if update.effective_chat.type not in ('private', 'group', 'supergroup'):
        await update.message.reply_text('Locations are available in private chats and groups.', reply_markup=None)
        return ConversationHandler.END
    if not label or len(label) > 80:
        await update.message.reply_text('Use /addlocation followed by a name (up to 80 characters), for example /addlocation Office.', reply_markup=None)
        return ConversationHandler.END
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if not rows:
        await update.message.reply_text('Use /start to set up the Main location first.', reply_markup=None)
        return ConversationHandler.END
    if len(rows) >= MAX_CHAT_LOCATIONS:
        await update.message.reply_text('Each chat can save up to 6 locations, including Main. Remove one before adding another.', reply_markup=None)
        return ConversationHandler.END
    if any(row['label'].casefold() == label.casefold() for row in rows):
        await update.message.reply_text('That location name is already saved. Choose another name.', reply_markup=None)
        return ConversationHandler.END
    context.chat_data['new_location_label'] = label
    await update.message.reply_text(f'Send a Telegram location for {label}. Use /cancel to cancel.', reply_markup=ForceReply())
    return WAITING_FOR_EXTRA_LOCATION


async def receive_extra_location(update, context):
    label = context.chat_data.get('new_location_label')
    if not label:
        await update.message.reply_text('Use /addlocation followed by a name to begin again.', reply_markup=None)
        return ConversationHandler.END
    location = update.message.location
    if not in_coverage(location.latitude, location.longitude):
        await update.message.reply_text('That location is outside radar coverage. Send another location or /cancel.', reply_markup=None)
        return WAITING_FOR_EXTRA_LOCATION
    async with settings_lock(context):
        saved = await asyncio.to_thread(add_named_location, update.effective_chat.id, label, location.latitude, location.longitude)
    context.chat_data.pop('new_location_label', None)
    await update.message.reply_text(f'Saved {label}. Rain alerts are on. You can change this in Locations & alerts.' if saved
                                    else 'Could not add that location. Each chat allows 6 locations including Main, with unique names. Check /locations.',
                                    reply_markup=None)
    if saved:
        await location_menu(update, context, 'locations')
    return ConversationHandler.END


async def location_menu(update, context, page):
    from telegram_code.menus import open_menu_page
    return await open_menu_page(update, context, page)


async def locations_command(update, context):
    context.chat_data.pop('new_location_label', None)
    return await location_menu(update, context, 'locations')


async def remove_location_command(update, context):
    await remove_location_by_name(update, context, ' '.join(context.args or []).strip())


async def remove_location_by_name(update, context, label):
    if not label:
        await update.message.reply_text('Use /removelocation followed by the saved name.', reply_markup=None)
        return
    async with settings_lock(context):
        removed = await asyncio.to_thread(remove_named_location, update.effective_chat.id, label)
    await update.message.reply_text(f'Removed {label} and cancelled its pending alerts.' if removed
                                    else 'Extra location not found. Use Saved locations. Main can be changed with Change location.',
                                    reply_markup=None)


async def cancel_location(update, context):
    context.chat_data.pop('new_location_label', None)
    await update.message.reply_text('Setup cancelled.', reply_markup=None)
    await location_menu(update, context, 'home')
    return ConversationHandler.END


async def add_location_button(update, context):
    if update.effective_chat.type not in ('private', 'group', 'supergroup'):
        await update.message.reply_text('Locations are available in private chats and groups.', reply_markup=None)
        return ConversationHandler.END
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    if not rows:
        await update.message.reply_text('Use /start to set up Main first.', reply_markup=None)
        return ConversationHandler.END
    if len(rows) >= MAX_CHAT_LOCATIONS:
        await update.message.reply_text('Each chat can save up to 6 locations, including Main. Remove one first.', reply_markup=None)
        return ConversationHandler.END
    await update.message.reply_text('What should this location be called? Send a name, such as Office, or /cancel.',
                                    reply_markup=ForceReply())
    return WAITING_FOR_LOCATION_NAME


async def receive_location_name(update, context):
    result = await begin_named_location(update, context, update.message.text.strip())
    return WAITING_FOR_LOCATION_NAME if result == ConversationHandler.END else result


async def remove_location_button(update, context):
    if update.effective_chat.type not in ('private', 'group', 'supergroup'):
        await update.message.reply_text('Locations are available in private chats and groups.', reply_markup=None)
        return ConversationHandler.END
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    extras = [row for row in rows if row['location_id'] != 0]
    if not extras:
        await update.message.reply_text('No extra locations to remove. Change location updates Main.',
                                        reply_markup=None)
        return ConversationHandler.END
    for row in extras:
        await update.message.reply_text(row['label'], reply_markup=None)
    await update.message.reply_text('Send the name of the location to remove, or /cancel.', reply_markup=ForceReply())
    return WAITING_FOR_REMOVAL_NAME


async def receive_removal_name(update, context):
    await remove_location_by_name(update, context, update.message.text.strip())
    return ConversationHandler.END
