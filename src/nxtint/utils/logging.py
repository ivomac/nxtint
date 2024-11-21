"""Logging configuration for nxtint."""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

# Default format includes timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

INFO = logging.INFO
DEBUG = logging.DEBUG


def setup_logger(
    name: str,
    level: int = INFO,
    fmt: str = DEFAULT_FORMAT,
    date_fmt: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level
        fmt: Log message format
        date_fmt: Date format in log messages
        log_file: Optional path to log file

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # Allow logs to propagate up to root logger

    # Only add handlers if none exist
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log(logger: logging.Logger, level: int = DEBUG) -> Callable:
    """Log function calls with args and return values.

    Args:
        logger: Logger instance to use
        level: Logging level for the messages

    Returns:
        Callable: Decorator function that preserves the wrapped function's metadata
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.log(level, f"\n>>> {func_name}(\n{args=},\n{kwargs=}\n)")
            result = func(*args, **kwargs)
            logger.log(level, f"\n>>> {func_name}()\n{result!r}")
            return result

        return wrapper

    return decorator
