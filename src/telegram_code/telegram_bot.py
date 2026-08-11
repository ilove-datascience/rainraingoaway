from telegram.ext import Application, CommandHandler, MessageHandler, filters
from functools import partial
import logging

try:
    from .methods import start, handle_msg, handle_location, load_token
except ImportError:
    from methods import start, handle_msg, handle_location, load_token


logger = logging.getLogger(__name__)


async def on_bot_error(update, context) -> None:
    # Logs full traceback and a compact update payload for debugging.
    logger.exception("Unhandled telegram error: %s", context.error)
    if update is not None:
        logger.error("Update that caused error: %s", update)


def run_bot(model, folder_path) -> None:
    token = load_token()
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    app = Application.builder().token(token).connect_timeout(30).read_timeout(30).write_timeout(30).pool_timeout(30).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    app.add_handler(
        MessageHandler(
            filters.LOCATION,
            partial(handle_location, model=model, folder_path=folder_path),
        )
    )
    app.add_error_handler(on_bot_error)
    app.run_polling()




