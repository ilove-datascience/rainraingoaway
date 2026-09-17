"""Reply keyboards scoped to the destination chat."""
from telegram import ReplyKeyboardMarkup
from telegram.ext import ConversationHandler


def main_menu(chat_id=None):
    rows = [["My forecast", "Current radar"], ["Change location", "Alert settings"]]
    rows += [["Add location", "Saved locations"], ["Remove location"]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, one_time_keyboard=True)


async def show_menu(update, context):
    context.chat_data.pop('new_location_label', None)
    await update.message.reply_text('Choose an option. Use /menu to show these buttons again.',
                                    reply_markup=main_menu(update.effective_chat.id))
    return ConversationHandler.END
