"""Hugging Face Space entry point; assembled into build/hf_space by `make space`."""

from pathlib import Path

from drscreen.app.ui import AppConfig, create_app, launch
from drscreen.config import load_config
from drscreen.logging_setup import configure_logging

configure_logging("INFO")
# Spaces set GRADIO_SERVER_NAME, so the app listens on all interfaces there.
launch(create_app(load_config(Path("configs/app.yaml"), AppConfig)))
