"""Host the web app on Modal: `modal deploy deploy/modal_app.py` from the repo root.

The image carries the package, the configs and the two checkpoints that
configs/app.yaml lists, so the container needs nothing from outside to start.
It scales to zero when idle and wakes on the next visit.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(
        "requirements.txt", extra_index_url="https://download.pytorch.org/whl/cpu"
    )
    .add_local_dir("src/drscreen", "/root/drscreen")
    .add_local_dir("configs", "/root/configs")
    .add_local_file("models/quality.pt", "/root/models/quality.pt")
    .add_local_file("models/grading-v4.pt", "/root/models/grading-v4.pt")
)
app = modal.App("dr-screening", image=image)


@app.function(cpu=2, memory=2048, scaledown_window=600, max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    """Build the ASGI app inside the container; Modal wraps and serves it."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    os.chdir("/root")
    from drscreen.app.ui import AppConfig  # noqa: PLC0415
    from drscreen.app.web import create_web  # noqa: PLC0415
    from drscreen.config import load_config  # noqa: PLC0415

    return create_web(load_config(Path("configs/app.yaml"), AppConfig))
