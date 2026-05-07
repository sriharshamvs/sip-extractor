"""Stage 9: Symbol detection.

Two detectors against the cropped binary:

- Template matching for distinctive simple shapes (KM marker, S/B box, BSLB,
  GL gate-lodge box). Multi-scale OpenCV matchTemplate.
- DINOv2 nearest-neighbour for chained-ellipse signal posts (Distant, Home,
  Starter, Shunt, Calling-on). Added in a follow-up commit.

Output schema is sip_extractor.schema.Symbol. Detections that fall inside an
OCR text bbox are dropped to suppress matches on dense label clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .schema import Symbol, TextEntity
from .utils.geometry import bbox_overlaps_any
from .utils.io import read_json, save_preview, write_json


# Keyword sets used to map the user's library slugs onto the priority classes
# in CLAUDE.md. The build_library notebook slugifies type names from the
# reference PDFs, so exact slugs vary; we match against the human-readable
# type_name from index.json. First match wins.
PRIORITY_CLASSES: dict[str, dict[str, list[str]]] = {
    "template_match": {
        "km_marker": ["km marker", "kilometer marker", "km/hm"],
        "sb_box": ["station building", "s/b", "sb box"],
        "bslb": ["bslb", "block section limit"],
        "gl": ["gate lodge", "gl ", "level crossing"],
    },
    "dinov2_nn": {
        "distant": ["distant"],
        "home": ["home"],
        "starter": ["starter"],
        "shunt": ["shunt"],
        "calling_on": ["calling-on", "calling on", "co signal"],
    },
}

# Multi-scale template match. Reference PDFs render at unknown DPI; the SIP
# at 300 DPI typically wants a 0.5-1.5x rescale of the exemplar. Wide range
# costs little since each scale is one matchTemplate call.
DEFAULT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_TEMPLATE_THRESHOLD = 0.7
DEFAULT_NMS_IOU = 0.3


@dataclass
class Library:
    """In-memory view of library/index.json plus on-demand exemplar loads."""

    root: Path
    classes: list[str]
    exemplars_by_class: dict[str, list[dict]]
    type_names: dict[str, str]
    _cache: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_path(cls, root: Path) -> "Library":
        root = Path(root)
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"library/index.json not found at {index_path}. "
                "Run notebooks/02_build_library.ipynb to generate it."
            )
        idx = read_json(index_path)
        return cls(
            root=root,
            classes=idx["classes"],
            exemplars_by_class=idx["exemplars_by_class"],
            type_names=idx.get("type_names", {}),
        )

    def exemplars(self, class_name: str) -> list[np.ndarray]:
        """Lazy-load exemplar PNGs for a class as grayscale uint8."""
        if class_name in self._cache:
            return self._cache[class_name]
        out: list[np.ndarray] = []
        for entry in self.exemplars_by_class.get(class_name, []):
            path = self.root / entry["path"]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[symbols] failed to read exemplar {path}; skipping", flush=True)
                continue
            out.append(img)
        self._cache[class_name] = out
        return out

    def classes_for_method(self, method: str) -> list[str]:
        """Return the library slugs matching the priority classes for a given
        detection method. Slugs are resolved by keyword-matching against each
        class's human-readable type_name. Missing priorities are silently
        dropped (logged via print) so the pipeline still runs partially.
        """
        groups = PRIORITY_CLASSES.get(method, {})
        resolved: list[str] = []
        for priority_name, keywords in groups.items():
            slug = self._find_slug(keywords)
            if slug is None:
                print(f"[symbols] no library class matched priority '{priority_name}' "
                      f"(keywords: {keywords}); skipping", flush=True)
                continue
            resolved.append(slug)
        return resolved

    def _find_slug(self, keywords: list[str]) -> str | None:
        kw_lower = [k.lower() for k in keywords]
        for slug in self.classes:
            label = self.type_names.get(slug, slug).lower()
            if any(kw in label for kw in kw_lower):
                return slug
        return None


def _multi_scale_match(
    haystack: np.ndarray,
    needle: np.ndarray,
    scales: Iterable[float],
    threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    """Run cv2.matchTemplate at each scale, return (x, y, w, h, score) tuples."""
    H, W = haystack.shape[:2]
    matches: list[tuple[int, int, int, int, float]] = []
    for s in scales:
        nh, nw = int(needle.shape[0] * s), int(needle.shape[1] * s)
        if nh < 8 or nw < 8 or nh >= H or nw >= W:
            continue
        scaled = cv2.resize(needle, (nw, nh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(haystack, scaled, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for y, x in zip(ys, xs):
            matches.append((int(x), int(y), nw, nh, float(res[y, x])))
    return matches


def _nms(
    boxes: list[tuple[int, int, int, int, float]],
    iou_thr: float,
) -> list[tuple[int, int, int, int, float]]:
    if not boxes:
        return []
    rects = [(b[0], b[1], b[2], b[3]) for b in boxes]
    scores = [b[4] for b in boxes]
    keep = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0, nms_threshold=iou_thr)
    if keep is None or len(keep) == 0:
        return []
    keep = keep.flatten() if hasattr(keep, "flatten") else list(keep)
    return [boxes[i] for i in keep]


def template_match(
    binary_cropped: np.ndarray,
    library: Library,
    text_bboxes: list[list[int]],
    scales: Iterable[float] = DEFAULT_SCALES,
    threshold: float = DEFAULT_TEMPLATE_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> list[Symbol]:
    """Detect simple-shape symbols (km marker, S/B, BSLB, GL) via multi-scale
    template matching. Drops detections that overlap any text bbox.
    """
    haystack = binary_cropped
    out: list[Symbol] = []
    next_id = 1
    for slug in library.classes_for_method("template_match"):
        all_matches: list[tuple[int, int, int, int, float]] = []
        for ex in library.exemplars(slug):
            all_matches.extend(_multi_scale_match(haystack, ex, scales, threshold))
        kept = _nms(all_matches, nms_iou)
        for x, y, w, h, score in kept:
            bbox = [x, y, w, h]
            if bbox_overlaps_any(bbox, text_bboxes):
                continue
            out.append({
                "id": f"symbol_{next_id:03d}",
                "type": "symbol",
                "class_name": slug,
                "bbox": bbox,
                "confidence": score,
                "method": "template_match",
            })
            next_id += 1
        print(f"[symbols] template_match {slug}: {len(kept)} pre-filter -> "
              f"{sum(1 for s in out if s['class_name'] == slug)} kept", flush=True)
    return out


def save_overlay(
    binary_cropped: np.ndarray,
    symbols: list[Symbol],
    out_path: Path,
) -> Path:
    """Render symbol bboxes on the binary, color-coded by method."""
    overlay = cv2.cvtColor(binary_cropped, cv2.COLOR_GRAY2BGR)
    palette = {
        "template_match": (0, 200, 255),  # yellow-orange
        "dinov2_nn": (255, 100, 0),       # blue
    }
    for s in symbols:
        x, y, w, h = s["bbox"]
        color = palette.get(s.get("method", ""), (0, 255, 0))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        label = f"{s['class_name'][:18]} {s['confidence']:.2f}"
        cv2.putText(overlay, label, (x, max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def run(
    binary_cropped: np.ndarray,
    text_entities: list[TextEntity],
    library: Library,
    out_dir: Path,
    methods: tuple[str, ...] = ("template_match",),
    write_overlay: bool = True,
) -> list[Symbol]:
    """Run Stage 9 end to end. Writes symbols.json (and optionally an overlay
    PNG) to out_dir; returns the symbol list for Stage 10 composition.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_bboxes = [t["bbox"] for t in text_entities if "bbox" in t]
    symbols: list[Symbol] = []
    if "template_match" in methods:
        symbols.extend(template_match(binary_cropped, library, text_bboxes))
    # dinov2_nn lands in the next commit.

    write_json(symbols, out_dir / "symbols.json")
    if write_overlay:
        overlay_path = out_dir / "symbols_overlay.png"
        save_overlay(binary_cropped, symbols, overlay_path)
        save_preview(cv2.imread(str(overlay_path)), out_dir / "symbols_overlay_preview.png")
    print(f"[symbols] total: {len(symbols)} symbols", flush=True)
    return symbols
