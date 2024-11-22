"""Logging configuration for nxtint."""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

INFO = logging.INFO
DEBUG = logging.DEBUG

FMT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(DEBUG)
    logger.propagate = True  # Allow logs to propagate up to root logger

    # Only add handlers if none exist
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_io(logger: logging.Logger) -> Callable:
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
            logger.log(DEBUG, f"\n>>> {func_name}(\n{args=},\n{kwargs=}\n)")
            result = func(*args, **kwargs)
            logger.log(DEBUG, f"\n>>> {func_name}()\n{result!r}")
            return result

        return wrapper

    return decorator
