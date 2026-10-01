"""Reading and writing RGB images on disk.

OpenCV stores channels as BGR; everything else in drscreen uses RGB. This
module is the only place that converts between the two.
"""

from pathlib import Path

import cv2

from drscreen.data.preprocessing import Image, validate_rgb
from drscreen.exceptions import DataError


def read_rgb(path: Path) -> Image:
    """Read an image file as an RGB uint8 array.

    Args:
        path: Path to a PNG, JPEG or other OpenCV-readable image.

    Returns:
        The image with shape ``(H, W, 3)``.

    Raises:
        DataError: If the file does not exist or cannot be decoded.
    """
    if not path.is_file():
        raise DataError(f"Image file not found: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise DataError(f"Image file could not be decoded (corrupt or unsupported): {path}")
    return validate_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def write_rgb(image: Image, path: Path) -> None:
    """Write an RGB uint8 array to an image file, creating parent directories.

    Args:
        image: RGB uint8 array.
        path: Destination; the extension selects the format.

    Raises:
        DataError: If OpenCV fails to encode or write the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise DataError(f"Failed to write image to {path}")
