"""Logging configuration for nxtint."""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

from .config import LogConfig

INFO = logging.INFO
DEBUG = logging.DEBUG

FMT = "%(asctime)s - %(levelname)s - %(module)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int | None = None) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level for the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    # Use default level if none provided
    level = level if level is not None else LogConfig.level

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # Allow logs to propagate up to root logger

    # Only add handlers if none exist
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # File handler
        log_dir = LogConfig.dir
        log_file = LogConfig.file
        log_path = log_dir / log_file
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def log_io(logger: logging.Logger, level: int = DEBUG) -> Callable:
    """Log function calls with args and return values.

    Args:
        logger: Logger instance to use
        level: Logging level for the messages

    Returns:
        Callable: Decorator function that preserves the wrapped function's metadata
    """

    def get_name(arg):
        if type(arg) is type:
            return arg.__name__
        return type(arg).__name__

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            arg_names = ", ".join(get_name(a) for a in args)
            kwarg_names = ", ".join(f"{k}=..." for k in kwargs)
            result = func(*args, **kwargs)
            logger.log(
                level,
                f"\n>>> {func_name}({arg_names}, {kwarg_names})" + f" -> {type(result).__name__}",
            )
            return result

        return wrapper

    return decorator
