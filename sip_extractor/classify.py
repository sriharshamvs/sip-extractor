"""Stage 7: regex classification of OCR text into useful buckets.

Categories:
    signal_id      letter prefix + digits + optional aspect suffix  (S63, SH6, C02, S2(i), SH 14 (ii))
    track_circuit  digits + optional A + T                          (116AT, 305T)
    km_marker      KM/JKM + digits                                   (KM 429)
    track_label    UP/DN/DOWN + MAIN/LOOP/etc, length-capped         (UP MAIN, DN LOOP)
    point_id       100-119 with optional /partner                    (103, 113/114)
    dimension      digits + optional decimal + optional m            (180m, 709.5m)
    note           everything else

point_id is restricted to 100-119 to avoid eating dimensions like 200 or 850.
Station-name detection is intentionally NOT done here; spatial position works
better than text patterns (an "all caps + 4+ chars" heuristic false-positives
on STARTER, BSLB, etc.).

Signal IDs with parenthetical aspect suffixes (`SH 14 (i)`, `S2 (ii)`, `CO 20 (ii)`)
appear on real SIPs to disambiguate aspects of the same signal. The regex matches
the aspect suffix as an optional trailing group; classification ignores
whitespace inside the head. text_normalized may collapse spaces in short tokens
(`S 19` -> `S19`) but leaves longer aspect-suffix tokens alone, so the regex
operates on the raw text variant when normalization is too aggressive.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import cv2

from .schema import Category, TextEntity
from .utils.io import save_preview, write_json


PATTERNS: dict[Category, re.Pattern] = {
    "signal_id": re.compile(
        r"^(S|SH|C|CO|H|D|AS|IB|GL|BSLB)\s?\d{1,3}[A-Z]?(\s*\([ivx]+\))?$",
        re.IGNORECASE,
    ),
    "track_circuit": re.compile(r"^\d{1,4}A?T$"),
    "km_marker": re.compile(r"^J?K\.?\s?M\.?\s?\d{2,4}", re.IGNORECASE),
    "track_label": re.compile(
        r"^(UP|DN|DOWN)\.?\s*(MAIN|LOOP|LINE|SIDING|YARD|SHUNT)"
        r"(\s+(HOME|STARTER|ADV\.?\s*STARTER|DIST\.?))?\s*\.?$",
        re.IGNORECASE,
    ),
    "point_id": re.compile(r"^1[01]\d(\s?/\s?1[01]\d)?$"),
    "dimension": re.compile(r"^\d{1,5}(\.\d+)?\s*m\.?$", re.IGNORECASE),
}


CATEGORY_COLORS: dict[Category, tuple[int, int, int]] = {
    "signal_id": (255, 100, 100),
    "track_circuit": (100, 255, 100),
    "km_marker": (255, 255, 100),
    "track_label": (100, 200, 255),
    "dimension": (255, 100, 255),
    "point_id": (200, 200, 200),
    "note": (80, 80, 80),
}


def classify_one(text: str) -> Category:
    """Return the first matching category, or 'note' if nothing matches."""
    s = text.strip()
    for cat, pat in PATTERNS.items():
        if pat.search(s):
            return cat
    return "note"


def classify(entities: list[TextEntity]) -> list[TextEntity]:
    """Add a 'category' field to each entity in place. Returns the same list
    for chaining.
    """
    for t in entities:
        t["category"] = classify_one(t["text_normalized"])
    return entities


def category_counts(entities: list[TextEntity]) -> Counter:
    return Counter(t.get("category", "note") for t in entities)


def save_overlay(
    gray_cropped,
    entities: list[TextEntity],
    out_path: Path,
) -> Path:
    overlay = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2BGR)
    for t in entities:
        x, y, w, h = t["bbox"]
        color = CATEGORY_COLORS.get(t.get("category", "note"), (200, 200, 200))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite(str(out_path), overlay)
    return out_path


def run(
    entities: list[TextEntity],
    out_dir: Path,
    gray_cropped=None,
    write_overlay: bool = True,
) -> list[TextEntity]:
    """Classify, rewrite text.json with the category field, and (optionally)
    save a category-colored overlay.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classify(entities)
    write_json(entities, out_dir / "text.json")

    if write_overlay and gray_cropped is not None:
        overlay_path = out_dir / "text_categorized_overlay.png"
        save_overlay(gray_cropped, entities, overlay_path)
        save_preview(cv2.imread(str(overlay_path)), out_dir / "text_categorized_preview.png")

    return entities
