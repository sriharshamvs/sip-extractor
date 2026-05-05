"""Stages 1-5: PDF render, color filter, binarize, edge crop, save.

Pipeline:
    1. Render PDF page as RGB at target DPI (PyMuPDF).
    2. Drop colored markup by zeroing pixels with chroma > 20.
    3. Sauvola binarize with auto-derived window size.
    4. Crop title blocks and edge tables off the X-axis.
    5. Save binary.png plus a downsampled preview.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

from .schema import PreprocessResult
from .utils.io import save_preview

Image.MAX_IMAGE_PIXELS = None


# Sauvola is computed at the target DPI; the window must scale with it. Strokes
# are roughly DPI/30 px wide, the window should span ~5x that. (DPI // 6) | 1
# gives an odd window of the right magnitude across 150-600 DPI.
def sauvola_window_for_dpi(dpi: int) -> int:
    return (dpi // 6) | 1


def render_pdf_page(pdf_path: Path, page_index: int, target_dpi: int) -> np.ndarray:
    """Render a single PDF page at target DPI as an RGB array."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        zoom = target_dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            rgb = rgb[:, :, :3].copy()
        return rgb
    finally:
        doc.close()


def drop_colored_markup(rgb: np.ndarray, chroma_thresh: int = 20) -> np.ndarray:
    """Convert to grayscale and blank out chromatic pixels.

    Handwritten red/blue/green annotations have non-trivial chroma; black
    structural ink has none. One filter does the whole 'color separation'
    elaborate dance.
    """
    rgb_max = rgb.max(axis=2)
    rgb_min = rgb.min(axis=2)
    chroma = cv2.subtract(rgb_max, rgb_min)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray[chroma > chroma_thresh] = 255
    return gray


def sauvola_binarize(gray: np.ndarray, window: int, k: float = 0.4) -> np.ndarray:
    """Adaptive thresholding. Sauvola handles uneven illumination and stroke
    width variation, so no separate background normalization is needed.
    """
    thr = threshold_sauvola(gray, window_size=window, k=k)
    return ((gray < thr).astype(np.uint8)) * 255


def crop_to_diagram(
    binary: np.ndarray,
    smooth_window: int = 200,
    high_density_factor: float = 2.0,
    bridge_gap: int = 2000,
    edge_margin: int = 200,
) -> tuple[int, int]:
    """Find the X-axis bounds of the diagram region, dropping title blocks
    and side legends.

    Title blocks fill columns with much higher ink density than the diagram's
    track region. The method is:

    1. Smoothed column-ink-density profile (edge-padded so border columns
       aren't artificially low).
    2. Threshold against high_density_factor * median of the central 50%.
    3. Morphologically close gaps so table grids (low-density rows between
       borders) don't stop the inward walk.
    4. Walk inward from each edge through any high-density region.

    Tune high_density_factor: raise (e.g., 2.5) if the crop is too aggressive,
    lower (e.g., 1.5) if too loose. Simpler approaches (morphological opening,
    Hough-line-based crops) have been tried and failed; do not replace this
    with one of those.
    """
    h, w = binary.shape
    col_ink = (binary > 0).sum(axis=0).astype(float)
    pad = smooth_window // 2
    padded = np.pad(col_ink, pad, mode="edge")
    smooth = np.convolve(padded, np.ones(smooth_window) / smooth_window, mode="valid")[:w]

    diagram_median = np.median(smooth[w // 4 : 3 * w // 4])
    is_high = (smooth >= diagram_median * high_density_factor).astype(np.uint8)

    kernel = np.ones((1, max(3, bridge_gap | 1)), np.uint8)
    closed = cv2.morphologyEx(is_high.reshape(1, -1) * 255, cv2.MORPH_CLOSE, kernel).ravel() > 0

    x_min = 0
    if closed[0]:
        while x_min < w - 1 and closed[x_min]:
            x_min += 1
    x_max = w - 1
    if closed[-1]:
        while x_max > 0 and closed[x_max]:
            x_max -= 1

    return max(0, x_min - edge_margin), min(w, x_max + edge_margin)


def run(
    pdf_path: Path,
    out_dir: Path,
    target_dpi: int = 300,
    page_index: int = 0,
    sauvola_k: float = 0.4,
) -> PreprocessResult:
    """Run Stages 1-5 end to end. Returns a PreprocessResult holding both the
    saved file paths and the in-memory cropped arrays for downstream stages.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = render_pdf_page(pdf_path, page_index=page_index, target_dpi=target_dpi)
    gray = drop_colored_markup(rgb)
    del rgb

    window = sauvola_window_for_dpi(target_dpi)
    binary = sauvola_binarize(gray, window=window, k=sauvola_k)

    x_min, x_max = crop_to_diagram(binary)
    binary_cropped = binary[:, x_min:x_max]
    gray_cropped = gray[:, x_min:x_max]

    binary_path = out_dir / "binary.png"
    cv2.imwrite(str(binary_path), binary_cropped)
    preview_path = out_dir / "binary_preview.png"
    save_preview(binary_cropped, preview_path)

    return PreprocessResult(
        rgb_shape=(binary.shape[0], binary.shape[1], 3),
        binary_path=str(binary_path),
        binary_preview_path=str(preview_path),
        crop_x_min=x_min,
        crop_x_max=x_max,
        target_dpi=target_dpi,
        binary_cropped=binary_cropped,
        gray_cropped=gray_cropped,
    )
