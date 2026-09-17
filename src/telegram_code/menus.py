"""Reply keyboards scoped to the destination chat."""
from telegram import ReplyKeyboardMarkup


def main_menu(chat_id=None):
    rows = [["My forecast", "Current radar"], ["Change location", "Alert settings"]]
    if chat_id is not None and chat_id < 0:
        rows += [["Add location", "Saved locations"], ["Remove location"]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)
