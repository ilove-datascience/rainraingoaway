from telegram.telegram import Bot, Update, ForceReply, InlineKeyboardMarkup, InlineKeyboardButton, PhotoSize
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.ext.filters import BaseFilter
import os
from methods import start 


TOKEN = os.getenv("tele_api_key")
app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))