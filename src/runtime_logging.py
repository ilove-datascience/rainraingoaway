"""Console and bounded on-disk logging for the bot and its worker threads."""
import io
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading


class SecretSafeFormatter(logging.Formatter):
    """Redact credentials in both messages and formatted exception tracebacks."""

    def format(self, record):
        text = super().format(record)
        for name in ('tele_api_key', 'gov_api_key', 'GOV_API_KEY',
                     'MYSQL_ROOT_PASSWORD', 'KUMA_PUSH_URL'):
            value = os.getenv(name)
            if value:
                text = text.replace(value, '[REDACTED]')
        return re.sub(r'\b\d{6,}:[A-Za-z0-9_-]{20,}', '[REDACTED]', text)


class LoggedStream(io.TextIOBase):
    """Turn legacy print output into timestamped records, one buffer per thread."""

    def __init__(self, logger, level, original):
        self.logger = logger
        self.level = level
        self.original = original
        self.local = threading.local()

    @property
    def encoding(self):
        return self.original.encoding or "utf-8"

    def writable(self):
        return True

    def isatty(self):
        return self.original.isatty()

    def write(self, text):
        pending = getattr(self.local, "pending", "") + text
        lines = pending.split("\n")
        self.local.pending = lines.pop()
        for line in lines:
            if line.strip():
                self.logger.log(self.level, "%s", line.rstrip("\r"))
        return len(text)

    def flush(self):
        pending = getattr(self.local, "pending", "")
        self.local.pending = ""
        if pending.strip():
            self.logger.log(self.level, "%s", pending)


def configure_logging(log_dir=None, *, max_bytes=10 * 1024 * 1024, backup_count=5):
    """Keep the active log plus five backups; preserve console output."""
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "rain_bot_log", False):
            return Path(handler.baseFilename)
    directory = Path(log_dir) if log_dir else Path(__file__).resolve().parents[1] / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rainraingoaway.log"
    formatter = SecretSafeFormatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    )
    file_handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.rain_bot_log = True
    console = logging.StreamHandler(sys.stderr)
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    # Application entry point owns root logging. Repeated setup is a no-op above.
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    sys.stdout = LoggedStream(logging.getLogger("stdout"), logging.INFO, sys.stdout)
    sys.stderr = LoggedStream(logging.getLogger("stderr"), logging.ERROR, sys.stderr)
    logging.getLogger(__name__).info("Logging to %s", path)
    return path
