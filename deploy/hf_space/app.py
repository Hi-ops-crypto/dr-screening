"""Hugging Face Space entry point; assembled into build/hf_space by `make space`."""

from pathlib import Path

import uvicorn

from drscreen.app.ui import AppConfig
from drscreen.app.web import create_web
from drscreen.config import load_config
from drscreen.logging_setup import configure_logging

configure_logging("INFO")
uvicorn.run(create_web(load_config(Path("configs/app.yaml"), AppConfig)), host="0.0.0.0", port=7860)
