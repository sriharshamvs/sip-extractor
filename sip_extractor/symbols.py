"""Stage 9: Symbol detection.

Two detectors against the cropped binary:

- Template matching for distinctive simple shapes (KM marker, S/B box, BSLB,
  GL gate-lodge box). Multi-scale OpenCV matchTemplate.
- DINOv2 nearest-neighbour for chained-ellipse signal posts (Distant, Home,
  Starter, Shunt, Calling-on). Connected components on the binary -> letterbox
  to 224x224 -> DINOv2-base CLS embedding -> cosine NN against exemplars.

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


DINOV2_MODEL_ID = "facebook/dinov2-base"
DINOV2_INPUT_SIZE = 224
DEFAULT_NN_COSINE_THRESHOLD = 0.45  # cosine sim, not distance; cross-modal exemplars need a permissive bar
DEFAULT_BATCH_SIZE = 16
DEFAULT_CC_AREA_MIN = 200
DEFAULT_CC_AREA_MAX = 50_000
DEFAULT_CC_ASPECT_MIN = 0.2
DEFAULT_CC_ASPECT_MAX = 5.0


# Keyword sets used to map the user's library slugs onto the priority classes
# in CLAUDE.md. The build_library notebook slugifies type names from the
# reference PDFs, so exact slugs vary; we match against the human-readable
# type_name from index.json. First match wins.
PRIORITY_CLASSES: dict[str, dict[str, list[str]]] = {
    "template_match": {
        "km_marker": ["km marker", "kilometer marker", "kilometre", "km/hm", "km stone", "km_stone"],
        "sb_box": ["station building", "s/b", "sb box", "stn building", "stn_bldg"],
        "bslb": ["bslb", "block section limit", "block_section", "block limit"],
        "gl": ["gate lodge", "gl ", "level crossing", "gate_lodge", "boards_gate"],
    },
    "dinov2_nn": {
        "distant": ["distant"],
        "home": ["home"],
        "starter": ["starter"],
        "shunt": ["shunt"],
        "calling_on": ["calling-on", "calling on", "calling_on", "co signal", "co_signal"],
    },
}

# Multi-scale template match. Reference PDFs render at unknown DPI; the SIP
# at 300 DPI typically wants a 0.5-1.5x rescale of the exemplar. Wide range
# costs little since each scale is one matchTemplate call.
DEFAULT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_TEMPLATE_THRESHOLD = 0.55  # cross-modal (clean PDF exemplar vs scan binary)
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
    starting_id: int = 1,
) -> list[Symbol]:
    """Detect simple-shape symbols (km marker, S/B, BSLB, GL) via multi-scale
    template matching. Drops detections that overlap any text bbox.
    """
    haystack = binary_cropped
    out: list[Symbol] = []
    next_id = starting_id
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


_embedder_singleton: dict | None = None


def _get_embedder():
    """Lazy-load DINOv2 model + image processor + device. Cached as a singleton.

    First call downloads ~340MB of weights, cached under HF_HOME (which the
    notebook setup cell points at .model_cache/ on Drive so it persists).
    """
    global _embedder_singleton
    if _embedder_singleton is not None:
        return _embedder_singleton

    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 ({DINOV2_MODEL_ID}, device={device}; first run downloads ~340 MB)...", flush=True)
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    print("DINOv2 ready.", flush=True)
    _embedder_singleton = {"model": model, "processor": processor, "device": device, "torch": torch}
    return _embedder_singleton


def _letterbox_to_rgb(img: np.ndarray, size: int = DINOV2_INPUT_SIZE) -> np.ndarray:
    """Letterbox-pad a single-channel image to a square RGB (white background),
    preserving aspect ratio. Output is HxWx3 uint8.
    """
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def _embed_batch(images: list[np.ndarray]) -> np.ndarray:
    """Embed a list of letterboxed RGB uint8 images. Returns L2-normalized
    (N, 768) float32 ndarray.
    """
    emb = _get_embedder()
    torch = emb["torch"]
    inputs = emb["processor"](images=images, return_tensors="pt").to(emb["device"])
    with torch.no_grad():
        out = emb["model"](**inputs)
    cls = out.last_hidden_state[:, 0, :]  # CLS token
    cls = torch.nn.functional.normalize(cls, dim=-1)
    return cls.cpu().numpy().astype(np.float32)


def _embed(images: list[np.ndarray], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Embed an arbitrary-length list, batching internally."""
    if not images:
        return np.zeros((0, 768), dtype=np.float32)
    out: list[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        out.append(_embed_batch(images[i:i + batch_size]))
    return np.concatenate(out, axis=0)


def _candidate_regions(
    binary_cropped: np.ndarray,
    text_bboxes: list[list[int]],
    area_min: int = DEFAULT_CC_AREA_MIN,
    area_max: int = DEFAULT_CC_AREA_MAX,
    aspect_min: float = DEFAULT_CC_ASPECT_MIN,
    aspect_max: float = DEFAULT_CC_ASPECT_MAX,
) -> list[tuple[int, int, int, int]]:
    """Connected components on inverted binary (ink as foreground). Filter by
    area, aspect ratio, and OCR text bbox overlap. Returns [(x, y, w, h), ...].
    """
    inv = cv2.bitwise_not(binary_cropped)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep: list[tuple[int, int, int, int]] = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < area_min or area > area_max:
            continue
        if w == 0 or h == 0:
            continue
        ar = w / h
        if ar < aspect_min or ar > aspect_max:
            continue
        if bbox_overlaps_any([int(x), int(y), int(w), int(h)], text_bboxes):
            continue
        keep.append((int(x), int(y), int(w), int(h)))
    return keep


def dinov2_nn(
    binary_cropped: np.ndarray,
    library: Library,
    text_bboxes: list[list[int]],
    starting_id: int = 1,
    cosine_threshold: float = DEFAULT_NN_COSINE_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> list[Symbol]:
    """Detect chained-ellipse signal classes via DINOv2 nearest-neighbor.

    Pipeline: connected components -> filter -> letterbox 224x224 -> DINOv2
    CLS embedding -> cosine NN against exemplar embeddings -> per-class NMS.
    """
    slugs = library.classes_for_method("dinov2_nn")
    if not slugs:
        print("[symbols] dinov2_nn: no library classes resolved; skipping", flush=True)
        return []

    # Embed exemplars once. Track which slug each row belongs to.
    exemplar_imgs: list[np.ndarray] = []
    exemplar_slugs: list[str] = []
    for slug in slugs:
        for ex in library.exemplars(slug):
            exemplar_imgs.append(_letterbox_to_rgb(ex))
            exemplar_slugs.append(slug)
    if not exemplar_imgs:
        print(f"[symbols] dinov2_nn: 0 exemplars for {slugs}; skipping", flush=True)
        return []
    print(f"[symbols] dinov2_nn: embedding {len(exemplar_imgs)} exemplars across {len(slugs)} classes...", flush=True)
    exemplar_emb = _embed(exemplar_imgs)

    candidates = _candidate_regions(binary_cropped, text_bboxes)
    print(f"[symbols] dinov2_nn: {len(candidates)} candidate regions after CC filter", flush=True)
    if not candidates:
        return []

    # Crop and embed each candidate.
    candidate_imgs = [
        _letterbox_to_rgb(binary_cropped[y:y + h, x:x + w])
        for x, y, w, h in candidates
    ]
    print(f"[symbols] dinov2_nn: embedding {len(candidate_imgs)} candidates...", flush=True)
    cand_emb = _embed(candidate_imgs)

    # Cosine similarity since both sides are L2-normalized.
    sims = cand_emb @ exemplar_emb.T  # (N_candidates, N_exemplars)
    best_ex = sims.argmax(axis=1)
    best_score = sims.max(axis=1)

    # Group by class for NMS.
    by_class: dict[str, list[tuple[int, int, int, int, float]]] = {slug: [] for slug in slugs}
    for i, ((x, y, w, h), score) in enumerate(zip(candidates, best_score)):
        if score < cosine_threshold:
            continue
        slug = exemplar_slugs[best_ex[i]]
        by_class[slug].append((x, y, w, h, float(score)))

    out: list[Symbol] = []
    next_id = starting_id
    for slug, hits in by_class.items():
        kept = _nms(hits, nms_iou)
        for x, y, w, h, score in kept:
            out.append({
                "id": f"symbol_{next_id:03d}",
                "type": "symbol",
                "class_name": slug,
                "bbox": [x, y, w, h],
                "confidence": score,
                "method": "dinov2_nn",
            })
            next_id += 1
        print(f"[symbols] dinov2_nn {slug}: {len(hits)} pre-NMS -> {len(kept)} kept", flush=True)
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
    methods: tuple[str, ...] = ("template_match", "dinov2_nn"),
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
        symbols.extend(template_match(binary_cropped, library, text_bboxes, starting_id=len(symbols) + 1))
    if "dinov2_nn" in methods:
        symbols.extend(dinov2_nn(binary_cropped, library, text_bboxes, starting_id=len(symbols) + 1))

    write_json(symbols, out_dir / "symbols.json")
    if write_overlay:
        overlay_path = out_dir / "symbols_overlay.png"
        save_overlay(binary_cropped, symbols, overlay_path)
        save_preview(cv2.imread(str(overlay_path)), out_dir / "symbols_overlay_preview.png")
    print(f"[symbols] total: {len(symbols)} symbols", flush=True)
    return symbols
