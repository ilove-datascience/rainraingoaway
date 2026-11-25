"""Persistent Telegram reply keyboards with explicit location subscriptions."""
import asyncio
import logging
from dataclasses import dataclass
from functools import partial

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import BadRequest
from telegram.ext import ConversationHandler

from telegram_code.database import get_user_mode
from telegram_code.group_locations import list_locations, remove_named_location
from telegram_code.location_alerts import set_location_alert, turn_off_all
from telegram_code.notification_delivery import settings_lock
from telegram_code.request_budget import bounded_weather_reply

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MenuButton:
    text: str
    action: str


@dataclass(frozen=True)
class MenuLayout:
    rows: list


def button(text, action):
    return MenuButton(text, action)


def main_menu(chat_id=None):
    return MenuLayout([
        [button('My forecast', 'forecast'), button('Current radar', 'radar')],
        [button('Locations & alerts', 'locations')],
        [button('Help', 'help'), button('Close', 'close')],
    ])


def navigation(back='home'):
    return [button('‹ Back', back), button('Close', 'close')]


def menu_state(update, context):
    # Settings are shared by the chat, but a member's navigation is their own.
    actor = str(update.effective_user.id)
    return context.chat_data.setdefault('reply_menus', {}).setdefault(actor, {})


def reply_options(update):
    if update.effective_chat.type in ('group', 'supergroup'):
        return {'reply_to_message_id': update.effective_message.message_id}
    return {}


async def render_menu(update, context, text, markup):
    rows = markup.rows
    keyboard = ReplyKeyboardMarkup([[key.text for key in row] for row in rows],
        resize_keyboard=True, one_time_keyboard=False, is_persistent=True, selective=True)
    await update.effective_message.reply_text(text, reply_markup=keyboard, **reply_options(update))
    state = menu_state(update, context)
    state.update(open=True, actions={key.text: key.action
                                    for row in rows for key in row})


async def close_menu(update, context):
    await update.effective_message.reply_text('Menu closed. Send /menu to open it again.',
        reply_markup=ReplyKeyboardRemove(selective=True), **reply_options(update))
    menu_state(update, context).clear()
    context.chat_data.pop('new_location_label', None)
    return ConversationHandler.END


async def open_menu_page(update, context, page='home', note=None):
    if page == 'home':
        text = 'What would you like to check?\nSend /menu again or tap Close to hide this menu.'
        markup = main_menu(update.effective_chat.id)
    elif page == 'help':
        text = ('How to use this bot\n\n'
                'My forecast — check all saved locations now.\n'
                'Current radar — see the latest observed rain.\n'
                'Locations & alerts — add a place, then tap its name to manage it or switch rain alerts on/off. '
                'New locations start with alerts on.\n\n'
                'Automatic alerts notify you when rain is expected and when it clears. '
                'Switching them off does not remove the location.\n\n'
                'Use /weather for the daily outlook and /weatheralerts for scheduled weather updates.\n\n'
                'Send /cancel to stop location setup. Send /menu again or tap Close to hide the menu.')
        markup = MenuLayout([navigation()])
    else:
        rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
        if not rows:
            text = 'Save your first location to get forecasts and choose rain alerts.'
            markup = MenuLayout([[button('Set up Main location', 'setup')], navigation()])
        elif page == 'alerts':
            mode = await asyncio.to_thread(get_user_mode, update.effective_chat.id)
            enabled = lambda row: mode == 'automatic' and bool(row['alerts_enabled'])
            count = sum(enabled(row) for row in rows)
            text = (f'Rain alerts · {count} of {len(rows)} locations on\n'
                    'Tap a location to turn its automatic rain alerts on or off. Changes save immediately.\n'
                    'You can still request a forecast for any saved location.')
            buttons = [[button(f"{'✓ On' if enabled(row) else '○ Off'} · {row['label']}",
                f"alert:{row['location_id']}:{0 if enabled(row) else 1}")] for row in rows]
            if count:
                buttons.append([button('Turn off all alerts', 'off_all')])
            markup = MenuLayout(buttons + [navigation()])
        else:
            mode = await asyncio.to_thread(get_user_mode, update.effective_chat.id)
            text = 'Locations & alerts\nTap a location to manage it. ✓ Alerts on · ○ Alerts off'
            buttons = [[button(f"{'✓' if mode == 'automatic' and row['alerts_enabled'] else '○'} {row['label']}",
                               f"location:{row['location_id']}")] for row in rows]
            if len(rows) < 6:
                buttons.append([button('+ Add location', 'add')])
            markup = MenuLayout(buttons + [navigation()])
    if update.effective_chat.type in ('group', 'supergroup'):
        text += '\nSettings are shared by this chat.'
    if note:
        text = note + '\n\n' + text
    await render_menu(update, context, text, markup)
    return ConversationHandler.END


async def open_location(update, context, identifier):
    rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
    row = next((row for row in rows if str(row['location_id']) == identifier), None)
    if not row:
        return await open_menu_page(update, context, 'locations', 'That location is no longer saved.')
    mode = await asyncio.to_thread(get_user_mode, update.effective_chat.id)
    enabled = mode == 'automatic' and bool(row['alerts_enabled'])
    text = f"{row['label']}\nRain alerts are {'on' if enabled else 'off'}.\n{row['latitude']}, {row['longitude']}"
    markup = MenuLayout([
        [button(f"Turn alerts {'off' if enabled else 'on'} · {row['label']}", f'location_alert:{identifier}:{0 if enabled else 1}')],
        [button('Change location', 'change') if row['location_id'] == 0
         else button('Remove location', f'remove:{identifier}')], navigation('locations')])
    await render_menu(update, context, text, markup)
    return ConversationHandler.END


async def show_menu(update, context):
    context.chat_data.pop('new_location_label', None)
    if menu_state(update, context).get('open'):
        return await close_menu(update, context)
    return await open_menu_page(update, context)


async def show_help(update, context):
    context.chat_data.pop('new_location_label', None)
    return await open_menu_page(update, context, 'help')


async def menu_callback(update, context, **unused):
    # Retire buttons on old message-attached menus after this UI change.
    query = update.callback_query
    await query.answer('Use /menu to open the keyboard menu.')
    if query.message:
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except BadRequest:
            pass
    return None


async def menu_text(update, context, model=None, folder_path=None, norm_stats=None):
    text = update.effective_message.text
    action = menu_state(update, context).get('actions', {}).get(text)
    if action is None:
        action = {'My forecast': 'forecast', 'Current radar': 'radar',
                  'Locations & alerts': 'locations', 'Help': 'help', 'Close': 'close'}.get(text)
    if action is None:
        await update.effective_message.reply_text('That option is no longer active. Use /menu to choose again.')
        return None
    context.chat_data.pop('new_location_label', None)
    return await dispatch_menu_action(update, context, action, model, folder_path, norm_stats)


async def dispatch_menu_action(update, context, action, model=None, folder_path=None, norm_stats=None):
    message_update = update
    if action == 'close':
        return await close_menu(update, context)
    if action in ('home', 'locations', 'alerts', 'help'):
        return await open_menu_page(update, context, action)
    if action in ('setup', 'change', 'add'):
        from telegram_code.methods import start, change_location
        from telegram_code.group_locations import add_location_button
        await render_menu(update, context, 'Follow the location prompt below, or tap Back to cancel.',
                          MenuLayout([navigation('locations')]))
        return await {'setup': start, 'change': change_location, 'add': add_location_button}[action](message_update, context)
    if action in ('forecast', 'radar'):
        from telegram_code.methods import handle_group_forecasts, handle_actual
        if action == 'forecast':
            await bounded_weather_reply(partial(handle_group_forecasts, model=model, folder_path=folder_path, norm_stats=norm_stats))(message_update, context)
        else:
            await bounded_weather_reply(partial(handle_actual, folder_path=folder_path))(message_update, context)
        return ConversationHandler.END
    if action == 'off_all' or action.startswith(('alert:', 'location_alert:')):
        try:
            async with settings_lock(context):
                if action == 'off_all':
                    saved = await asyncio.to_thread(turn_off_all, update.effective_chat.id)
                else:
                    _, location, value = action.split(':')
                    if value not in ('0', '1') or not location.isdecimal():
                        raise ValueError('Invalid location selection')
                    saved = await asyncio.to_thread(set_location_alert, update.effective_chat.id, int(location), value == '1')
        except Exception:
            logger.exception('Could not save location alert selection')
            await update.effective_message.reply_text('Could not save your alert selection. Please try again.')
            return ConversationHandler.END
        if action.startswith('location_alert:') and saved:
            return await open_location(update, context, location)
        return await open_menu_page(update, context, 'alerts', None if saved else 'That location is no longer saved.')
    if action.startswith(('location:', 'remove:', 'confirm_remove:')):
        operation, identifier = action.split(':', 1)
        rows = await asyncio.to_thread(list_locations, update.effective_chat.id)
        row = next((row for row in rows if str(row['location_id']) == identifier), None)
        if not row:
            return await open_menu_page(update, context, 'locations', 'That location is no longer saved.')
        if operation == 'confirm_remove' and row['location_id'] != 0:
            async with settings_lock(context):
                removed = await asyncio.to_thread(remove_named_location, update.effective_chat.id, row['label'], int(identifier))
            return await open_menu_page(update, context, 'locations',
                f"Removed {row['label']} and cancelled its alerts." if removed else 'That location is no longer saved.')
        if operation == 'remove' and row['location_id'] != 0:
            text = f"Remove {row['label']}? Its automatic alerts will also stop."
            markup = MenuLayout([[button(f"Confirm remove · {row['label']}", f'confirm_remove:{identifier}')],
                                           navigation(f'location:{identifier}')])
        else:
            return await open_location(update, context, identifier)
        await render_menu(update, context, text, markup)
        return ConversationHandler.END
    return await open_menu_page(update, context)
