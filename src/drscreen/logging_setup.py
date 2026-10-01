"""Process-wide logging configuration.

Library modules only ever call ``logging.getLogger(__name__)``. Configuring
handlers and levels is the job of entry points (scripts, the web app), which
call ``configure_logging`` once at startup.
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def configure_logging(level: str) -> None:
    """Configure the root logger with a single stderr handler.

    Args:
        level: One of ``LOG_LEVELS``, case-insensitive.

    Raises:
        ValueError: If ``level`` is not a recognised level name.
    """
    normalised = level.upper()
    if normalised not in LOG_LEVELS:
        raise ValueError(f"Unknown log level {level!r}; choose one of {', '.join(LOG_LEVELS)}.")
    logging.basicConfig(level=normalised, format=LOG_FORMAT, force=True)
