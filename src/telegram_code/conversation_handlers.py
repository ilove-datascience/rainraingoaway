

from telegram.ext import ContextTypes, ConversationHandler

from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram_code.methods import start, receive_location, update_mode
# conversation_handlers.py

from telegram.ext import (
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from telegram_code.methods import start, receive_location, receive_mode


from telegram_code.states import WAITING_FOR_LOCATION, WAITING_FOR_MODE


def get_conversation_handler():
    return ConversationHandler(
        entry_points=[
            CommandHandler("start", start)
        ],
        states={
            WAITING_FOR_LOCATION: [
                MessageHandler(filters.LOCATION, receive_location)
            ],
            WAITING_FOR_MODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_mode)
            ]
        },
        fallbacks=[]
    )
    
def get_conversation_handler2():
    return ConversationHandler (
            entry_points= [CommandHandler("setmode", update_mode)]
            ,
            states={
                    WAITING_FOR_MODE: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, receive_mode)
                    ]},
            fallbacks=[]
    )