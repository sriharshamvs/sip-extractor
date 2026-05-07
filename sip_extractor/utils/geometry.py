"""Bbox and polyline helpers."""

from __future__ import annotations

from typing import Sequence


Bbox = Sequence[int]  # [x, y, w, h]


def bbox_iou(a: Bbox, b: Bbox) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def bbox_center(b: Bbox) -> tuple[int, int]:
    return b[0] + b[2] // 2, b[1] + b[3] // 2


def bbox_overlaps_any(b: Bbox, others: Sequence[Bbox], iou_thr: float = 0.0) -> bool:
    """True if b overlaps any bbox in others above iou_thr (default: any overlap)."""
    return any(bbox_iou(b, o) > iou_thr for o in others)
