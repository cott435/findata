"""Timing helpers for pipeline logging.

``timed`` wraps a code block, ``log_timing`` wraps a function; both emit one
log line with the elapsed wall time. Runs under 0.1s log at DEBUG so hot
loops don't flood the log; slower ones log at INFO.
"""

import functools
import logging
import time
from contextlib import contextmanager

_DEBUG_UNDER = 0.1  # seconds


@contextmanager
def timed(logger: logging.Logger, label: str, level: int | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if level is None:
            level = logging.DEBUG if elapsed < _DEBUG_UNDER else logging.INFO
        logger.log(level, '%s done in %.2fs', label, elapsed)


def log_timing(logger: logging.Logger = None, label: str = None):
    """Decorator form of ``timed``. Defaults to the wrapped function's module
    logger and qualified name."""
    def decorate(func):
        log = logger or logging.getLogger(func.__module__)
        name = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timed(log, name):
                return func(*args, **kwargs)
        return wrapper
    return decorate
