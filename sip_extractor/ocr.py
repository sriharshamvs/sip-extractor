"""Stage 6: PaddleOCR + dedup + normalize.

Tile-based OCR over the cleaned grayscale (NOT the binary; Sauvola fragments
thin character strokes). Detections from overlapping tiles are deduped by
normalized text and bbox IoU. Outputs text.json with entries:

    {text, text_normalized, score, bbox: [x, y, w, h]}

Category is added later by classify.run().
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .schema import TextEntity
from .utils.geometry import bbox_iou
from .utils.io import write_json


TILE = 2000
OVERLAP = 200
DEFAULT_REC_SCORE_THRESH = 0.5
DEDUPE_IOU = 0.3


_ocr_singleton = None


def _get_ocr(rec_score_thresh: float):
    """Lazy-load PaddleOCR. First call triggers a model download (a few hundred
    MB) cached under ~/.paddleocr/ (or PADDLE_HOME on Colab).
    """
    global _ocr_singleton
    if _ocr_singleton is not None:
        return _ocr_singleton
    from paddleocr import PaddleOCR

    _ocr_singleton = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang="en",
        text_rec_score_thresh=rec_score_thresh,
    )
    return _ocr_singleton


def normalize_text(s: str) -> str:
    """Collapse OCR whitespace artifacts so 'S 19' == 'S19'.

    Strips internal whitespace from short (<=8 chars) alphanumeric-ish tokens
    only, keeping longer text intact. Note that 'UP MAIN' (7 chars) does
    collapse to 'UPMAIN' under this rule; the track_label regex in classify.py
    matches both forms.
    """
    s = s.strip()
    if len(s) <= 8 and re.match(r"^[A-Z0-9./\s]+$", s, re.IGNORECASE):
        s = re.sub(r"\s+", "", s)
    return s


def dedupe(entities: list[TextEntity], iou_thr: float = DEDUPE_IOU) -> list[TextEntity]:
    """Drop near-duplicate detections produced by tile overlap.

    Two detections collapse if their normalized text matches AND their bboxes
    overlap above iou_thr. Highest-confidence detection wins.
    """
    entities = sorted(entities, key=lambda e: -e["score"])
    kept: list[TextEntity] = []
    for e in entities:
        if not any(
            bbox_iou(e["bbox"], k["bbox"]) > iou_thr
            and e["text_normalized"] == k["text_normalized"]
            for k in kept
        ):
            kept.append(e)
    return kept


def _iter_tiles(h: int, w: int, tile: int, overlap: int) -> Iterable[tuple[int, int, int, int]]:
    for y0 in range(0, h, tile - overlap):
        for x0 in range(0, w, tile - overlap):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            if y1 <= y0 or x1 <= x0:
                continue
            yield x0, y0, x1, y1


def detect(
    gray_cropped: np.ndarray,
    tile: int = TILE,
    overlap: int = OVERLAP,
    rec_score_thresh: float = DEFAULT_REC_SCORE_THRESH,
) -> list[TextEntity]:
    """Run tile-based PaddleOCR over a single-channel grayscale image.

    Tile-based detection bounds memory and works better than feeding one
    giant image. Coordinates are returned in the source image's frame.
    """
    ocr = _get_ocr(rec_score_thresh)
    rgb = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]

    entities: list[TextEntity] = []
    for x0, y0, x1, y1 in _iter_tiles(h, w, tile, overlap):
        results = ocr.predict(rgb[y0:y1, x0:x1])
        for r in results:
            for poly, text, score in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
                xs = [int(p[0]) + x0 for p in poly]
                ys = [int(p[1]) + y0 for p in poly]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                entities.append(
                    {
                        "text": text,
                        "text_normalized": normalize_text(text),
                        "score": float(score),
                        "bbox": bbox,
                    }
                )

    return dedupe(entities)


def save_overlay(
    gray_cropped: np.ndarray,
    entities: list[TextEntity],
    out_path: Path,
    color: tuple[int, int, int] = (0, 200, 255),
) -> Path:
    overlay = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2BGR)
    for t in entities:
        x, y, w, h = t["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            overlay,
            t["text"][:20],
            (x, max(15, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), overlay)
    return out_path


def run(
    gray_cropped: np.ndarray,
    out_dir: Path,
    tile: int = TILE,
    overlap: int = OVERLAP,
    rec_score_thresh: float = DEFAULT_REC_SCORE_THRESH,
    write_overlay: bool = True,
) -> list[TextEntity]:
    """Run Stage 6 end to end. Writes text.json (and optionally an overlay
    PNG) to out_dir; returns the entity list for downstream stages.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entities = detect(
        gray_cropped, tile=tile, overlap=overlap, rec_score_thresh=rec_score_thresh
    )
    write_json(entities, out_dir / "text.json")

    if write_overlay:
        from .utils.io import save_preview

        overlay_path = out_dir / "text_overlay.png"
        save_overlay(gray_cropped, entities, overlay_path)
        save_preview(cv2.imread(str(overlay_path)), out_dir / "text_overlay_preview.png")

    return entities
