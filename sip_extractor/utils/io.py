"""File IO and image display helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


PREVIEW_WIDTH = 3000


def save_preview(image: np.ndarray, path: Path, width: int = PREVIEW_WIDTH) -> Path:
    """Downsample a wide engineering scan for inline notebook display.

    Full-resolution binaries at 300 DPI are too wide to render usefully inline
    and matplotlib OOMs on Colab when given them. Save a fixed-width preview
    and display via IPython.display.Image instead.
    """
    h, w = image.shape[:2]
    if w <= width:
        cv2.imwrite(str(path), image)
        return path
    target_h = int(h * width / w)
    preview = cv2.resize(image, (width, target_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), preview)
    return path


def write_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
