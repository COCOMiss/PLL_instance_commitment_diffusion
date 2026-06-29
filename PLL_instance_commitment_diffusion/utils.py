import logging
import os
import sys


_LOGGING_INITIALIZED = False


def init_logging(log_file=None, force=True):
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED and not force:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError:
                pass

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"Log file set to: {log_file}")

    _LOGGING_INITIALIZED = True


def get_logger(name):
    if not _LOGGING_INITIALIZED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logging.getLogger(name)
