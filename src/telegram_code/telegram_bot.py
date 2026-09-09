import logging
from functools import partial

from telegram.ext import Application, CommandHandler, MessageHandler, filters

try:
    from .methods import check_model_queue, handle_actual, handle_location, handle_msg, load_token, start
except ImportError:
    from methods import check_model_queue, handle_actual, handle_location, handle_msg, load_token, start

from telegram_code.conversation_handlers import get_conversation_handler, get_conversation_handler2

logger = logging.getLogger(__name__)


async def on_bot_error(update, context) -> None:
    # Logs full traceback and a compact update payload for debugging.
    logger.exception("Unhandled telegram error: %s", context.error)
    if update is not None:
        logger.error("Update that caused error: %s", update)


def run_bot(model, folder_path, model_ready_queue, norm_stats=None) -> None:
    token = load_token()
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )
    app.add_handler(get_conversation_handler())
    app.add_handler(get_conversation_handler2())

    app.job_queue.run_repeating(
        partial(
            check_model_queue,
            model=model,
            folder_path=folder_path,
            norm_stats=norm_stats,
            model_ready_queue=model_ready_queue,
        ),
        interval=30,
        first=0,
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        CommandHandler(
            "actual",
            partial(handle_actual, folder_path=folder_path),
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    app.add_handler(
        MessageHandler(
            filters.LOCATION,
            partial(
                handle_location,
                model=model,
                folder_path=folder_path,
                norm_stats=norm_stats,
            ),
        )
    )

    app.add_error_handler(on_bot_error)
    app.run_polling()




