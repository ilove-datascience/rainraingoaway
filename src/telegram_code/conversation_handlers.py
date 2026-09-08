

from telegram.ext import ContextTypes, ConversationHandler

from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram_code.methods import start, receive_location
# conversation_handlers.py

from telegram.ext import (
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from telegram_code.methods import start, receive_location


from telegram_code.states import WAITING_FOR_LOCATION


def get_conversation_handler():
    return ConversationHandler(
        entry_points=[
            CommandHandler("start", start)
        ],
        states={
            WAITING_FOR_LOCATION: [
                MessageHandler(filters.LOCATION, receive_location)
            ]
        },
        fallbacks=[]
    )
