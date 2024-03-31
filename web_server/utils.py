import logging
from typing import TextIO


def init_logger(logger_name: str, log_file: None | str = None, log_stream: None | TextIO = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    # Make the logger be able to handle all levels of logs
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_log = logging.FileHandler(log_file)
    console_log = logging.StreamHandler(log_stream)
    # Set the level of the handlers, so that the logger can handle different levels of logs
    console_log.setLevel(logging.INFO)
    file_log.setLevel(logging.INFO)
    console_log.setFormatter(formatter)
    file_log.setFormatter(formatter)
    logger.addHandler(console_log)
    logger.addHandler(file_log)
    return logger


def reset_logger_level(logger: logging.Logger, level: int):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
