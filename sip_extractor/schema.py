"""JSON schemas / dataclasses for pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


Category = Literal[
    "signal_id",
    "track_circuit",
    "km_marker",
    "track_label",
    "point_id",
    "dimension",
    "note",
]


class TextEntity(TypedDict, total=False):
    text: str
    text_normalized: str
    score: float
    bbox: list[int]  # [x, y, w, h]
    category: Category
    engine: str  # "paddleocr" | "qwen-vl" — set by Stage 9.5


SymbolMethod = Literal["template_match", "dinov2_nn"]


class Symbol(TypedDict, total=False):
    id: str                # "symbol_001"
    type: Literal["symbol"]
    class_name: str        # slug from library/index.json, e.g. "MainHomeWithoutJR"
    bbox: list[int]        # [x, y, w, h] in cropped-binary frame
    confidence: float
    method: SymbolMethod
    anchored_text_id: str  # set later by Stage 10 composition


@dataclass
class PreprocessResult:
    """What preprocessing produces. The arrays stay in memory for downstream
    stages; the saved files are for inspection and downstream notebook reuse.
    """
    rgb_shape: tuple[int, int, int]
    binary_path: str
    binary_preview_path: str
    crop_x_min: int
    crop_x_max: int
    target_dpi: int
    # Working arrays (not serialized): held on the result so OCR can reuse
    # them without re-rendering.
    binary_cropped: object = field(repr=False, default=None)
    gray_cropped: object = field(repr=False, default=None)
