
from telegram.ext import CommandHandler, ConversationHandler, MessageHandler, filters

from telegram_code.methods import receive_location, receive_mode, start, update_mode, change_location
from telegram_code.states import WAITING_FOR_LOCATION, WAITING_FOR_MODE


def get_conversation_handler():
    return ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            MessageHandler(filters.Regex("^Change location$"), change_location),
            CommandHandler("setmode", update_mode),
            MessageHandler(filters.Regex("^Alert settings$"), update_mode),
        ],
        states={
            WAITING_FOR_LOCATION: [
                MessageHandler(filters.LOCATION, receive_location)
            ],
            WAITING_FOR_MODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_mode)
            ]
        },
        fallbacks=[],
        allow_reentry=True,
    )
    
def get_conversation_handler2():
    return ConversationHandler (
            entry_points= [CommandHandler("setmode", update_mode),
                           MessageHandler(filters.Regex("^Alert settings$"), update_mode)]
            ,
            states={
                    WAITING_FOR_MODE: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, receive_mode)
                    ]},
            fallbacks=[]
    )
