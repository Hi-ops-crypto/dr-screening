"""Start the screening web app on http://127.0.0.1:7860 (classic Gradio UI at /classic)."""

import argparse
import sys
from pathlib import Path

import uvicorn

from drscreen.app.ui import AppConfig
from drscreen.app.web import create_web
from drscreen.cli import build_parser, run
from drscreen.config import load_config

DEFAULT_CONFIG = Path("configs/app.yaml")


def main(args: argparse.Namespace) -> None:
    """Load the app config, the models, and serve until interrupted.

    Args:
        args: Parsed command-line arguments.
    """
    uvicorn.run(create_web(load_config(args.config, AppConfig, args.override)), port=7860)


if __name__ == "__main__":
    sys.exit(run(main, build_parser(__doc__, DEFAULT_CONFIG).parse_args()))
