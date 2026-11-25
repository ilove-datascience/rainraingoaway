
from functools import partial
from telegram.ext import CallbackQueryHandler, CommandHandler, ConversationHandler, MessageHandler, filters

from telegram_code.methods import receive_location, receive_mode, start, update_mode, change_location
from telegram_code.group_locations import (add_location_command, receive_extra_location,
    cancel_location, WAITING_FOR_EXTRA_LOCATION, add_location_button, receive_location_name,
    remove_location_button, receive_removal_name, WAITING_FOR_LOCATION_NAME, WAITING_FOR_REMOVAL_NAME,
    locations_command)
from telegram_code.menus import show_menu, show_help, menu_callback, menu_text
from telegram_code.states import WAITING_FOR_LOCATION, WAITING_FOR_MODE


def get_conversation_handler(model=None, folder_path=None, norm_stats=None):
    return ConversationHandler(
        per_user=False,  # One shared setup conversation per chat.
        entry_points=[
            MessageHandler(filters.Regex(r'^(My forecast|Current radar|Locations & alerts|Help|Close|‹ Back|\+ Add location|Set up Main location|Change location|Remove location|Turn off all alerts|Turn alerts (?:on|off) · .+|Confirm remove · .+|[✓○] .+)$'),
                           partial(menu_text, model=model, folder_path=folder_path, norm_stats=norm_stats)),
            CallbackQueryHandler(partial(menu_callback, model=model, folder_path=folder_path,
                                         norm_stats=norm_stats), pattern=r'^menu:'),
            CommandHandler("start", start),
            CommandHandler("menu", show_menu),
            CommandHandler("help", show_help),
            CommandHandler("locations", locations_command),
            MessageHandler(filters.Regex("^Saved locations$"), locations_command),
            CommandHandler("cancel", cancel_location),
            CommandHandler("addlocation", add_location_command),
            MessageHandler(filters.Regex("^Add location$"), add_location_button),
            MessageHandler(filters.Regex("^Remove location$"), remove_location_button),
            MessageHandler(filters.Regex("^Change location$"), change_location),
            CommandHandler("setmode", update_mode),
            MessageHandler(filters.Regex("^Alert settings$"), update_mode),
            MessageHandler(filters.Regex("^(1 - Automatic rain updates|2 - Manual updates only)$"), receive_mode),
        ],
        states={
            WAITING_FOR_LOCATION_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_location_name)],
            WAITING_FOR_REMOVAL_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_removal_name)],
            WAITING_FOR_EXTRA_LOCATION: [MessageHandler(filters.LOCATION, receive_extra_location)],
            WAITING_FOR_LOCATION: [
                MessageHandler(filters.LOCATION, receive_location)
            ],
            WAITING_FOR_MODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_mode)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel_location)],
        allow_reentry=True,
    )
    
def get_conversation_handler2():
    return ConversationHandler (
            per_user=False,
            entry_points= [CommandHandler("setmode", update_mode),
                           MessageHandler(filters.Regex("^Alert settings$"), update_mode)]
            ,
            states={
                    WAITING_FOR_MODE: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, receive_mode)
                    ]},
            fallbacks=[]
    )
