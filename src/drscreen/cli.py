"""Shared plumbing for the command-line entry points in ``scripts/``.

Every script accepts the same ``--config``, ``--override`` and ``--log-level``
options and handles expected failures the same way: a ``DrScreenError`` is
logged as a one-line actionable message with exit status 1, and anything else
is treated as a bug and propagates with its traceback.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from drscreen.exceptions import DrScreenError
from drscreen.logging_setup import LOG_LEVELS, configure_logging

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def build_parser(description: str | None, default_config: Path) -> argparse.ArgumentParser:
    """Create an argument parser with the options shared by all scripts.

    Args:
        description: Help text, usually the script's module docstring.
        default_config: Config file used when ``--config`` is not given.

    Returns:
        A parser that scripts may extend with their own arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=default_config, help="YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. split.seed=7. Repeatable.",
    )
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS)
    return parser


def run(main: Callable[[argparse.Namespace], None], args: argparse.Namespace) -> int:
    """Configure logging, run an entry point and map expected errors to exit codes.

    Args:
        main: The script body, taking the parsed arguments.
        args: Arguments parsed by a parser from ``build_parser``.

    Returns:
        Process exit status.
    """
    configure_logging(args.log_level)
    try:
        main(args)
    except DrScreenError as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE
    return EXIT_SUCCESS
