# sip-extractor — Project Handoff

## Goal

Build a reusable Python package that takes an Indian Railways Signaling Interlocking Plan (SIP) PDF and produces structured JSON describing tracks, text labels, and signal symbols. Output should be reusable across many SIPs from different stations.

## Constraints

- Personal project — cannot use Penn Medicine Databricks, no enterprise GPU access.
- User has Colab Pro (100 compute units/month) and develops via VS Code with the Colab plugin (kernel runs on Colab, code lives locally).
- Local CPU for development and Stages 1-7. Colab GPU (T4 or A100) for Stages 9+ which need ML inference.
- No em dashes in any writing or comments. Direct simple language. No "I'd love" type AI-sounding phrases.
- All written deliverables (READMEs, comments) should sound like a senior engineer wrote them, not a chatbot.

## Repository Layout

```
sip-extractor/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml                    # package metadata
│
├── sip_extractor/                    # the package
│   ├── __init__.py
│   ├── preprocessing.py              # Stages 1-5: PDF render, color filter, binarize, crop
│   ├── ocr.py                        # Stage 6: PaddleOCR + dedup + normalize
│   ├── classify.py                   # Stage 7: regex classification of text
│   ├── symbols.py                    # Stage 9: template match + DINOv2 NN (TO BUILD)
│   ├── tracks.py                     # Stage 8: text+symbol-anchored tracks (TO REBUILD)
│   ├── compose.py                    # Stage 10: signal composition rules (TO BUILD)
│   ├── schema.py                     # JSON schemas / dataclasses
│   └── utils/
│       ├── geometry.py               # bbox/polyline helpers
│       └── io.py                     # file IO, image display helpers
│
├── notebooks/                        # entry points and exploration
│   ├── 01_preprocess.ipynb           # local CPU, Stages 1-7
│   ├── 02_build_library.ipynb        # local CPU, one-time symbol library extraction
│   ├── 03_detect_symbols.ipynb       # Colab GPU, Stage 9 (DINOv2)
│   ├── 04_full_pipeline.ipynb        # end-to-end on a single SIP
│   └── batch_corpus.ipynb            # Colab GPU, runs the pipeline over many SIPs
│
├── data/
│   ├── refs/                         # reference PDFs (gitignored, kept on Drive)
│   │   ├── Types_of_signals.pdf
│   │   ├── Types_of_signals_ncr.pdf
│   │   ├── Types_of_signals_from_vassar.pdf
│   │   └── NCR_DIMENTIONS.pdf
│   └── sips/                         # SIPs to process (gitignored)
│       └── MUNDEWADI-SIP.pdf
│
├── library/                          # output of build_library.ipynb (gitignored, regenerable)
│   ├── index.json
│   └── <class_name>/exemplar_NN.png
│
├── outputs/                          # per-SIP outputs (gitignored)
│   └── <sip_name>/
│       ├── binary.png
│       ├── text.json
│       ├── tracks.json
│       └── symbols.json
│
└── .model_cache/                     # gitignored, persisted on Drive when on Colab
```

## Colab + VS Code Setup

### Drive layout

The user uses Drive as the persistent filesystem. Project root sits at `/content/drive/MyDrive/sip-extractor/` when mounted. Local development uses the same relative paths.

### Setup cell template (top of every notebook)

```python
import os
from pathlib import Path

# Detect whether we're on Colab or local
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/drive/MyDrive/sip-extractor")

    # Cache models on Drive so we don't redownload every session.
    # Saves ~30-60s on PaddleOCR/DINOv2, several minutes on Qwen2.5-VL.
    cache = PROJECT_ROOT / ".model_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["PADDLE_HOME"] = str(cache / "paddle")
else:
    PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd()

os.chdir(PROJECT_ROOT)
print(f"Working dir: {PROJECT_ROOT}")
```

### .gitignore

```
# Data
data/sips/*
data/refs/*
!data/refs/README.md
!data/sips/README.md

# Outputs (regenerable)
outputs/
library/

# Model cache
.model_cache/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.venv/
venv/

# OS
.DS_Store
```

### requirements.txt

```
# Preprocessing + OCR (CPU, runs anywhere)
pymupdf
opencv-python
scikit-image
numpy
pillow
paddleocr==3.5.0
paddlepaddle

# Symbol detection (GPU recommended, runs on Colab)
transformers
torch
faiss-cpu  # use faiss-gpu on Colab if needed

# Notebooks
ipykernel
ipython
```

## Pipeline Stages

### Stage 1-5: Preprocessing (working, do not change without reason)

1. Render PDF as RGB at 300 DPI using PyMuPDF.
2. Drop colored markup (handwritten red/blue/green annotations). One filter: `gray[chroma > 20] = 255` where `chroma = max(R,G,B) - min(R,G,B)`.
3. Sauvola binarize with auto-derived window size `(TARGET_DPI // 6) | 1`.
4. Crop title blocks/tables off the X-axis edges. Uses column ink-density profile thresholding with `high_density_factor=2.0` and `bridge_gap=2000`. Method: convolve column ink count with edge-padded smoothing window, threshold relative to median of central 50%, morphologically close gaps with kernel `(1, bridge_gap)`, walk inward from each edge through high-density zones. This step is critical and well-tuned; do not replace it with simpler approaches. We tried morphological opening and Hough-line-based crops and they failed.
5. Save outputs: `binary.png` (full resolution) and `binary_preview.png` (3000px wide downsample for inline display). Use `IPython.display.Image` for inline display, NOT matplotlib (matplotlib OOMs on Colab and holds figure state).

### Stage 6: OCR (working with known issues)

Uses PaddleOCR 3.5.0:
- `text_rec_score_thresh=0.5` to drop garbage like `WoooL` (low-confidence misreads of `1000m`)
- `use_textline_orientation=True` for rotated KM markers
- Tile-based: 2000px tiles with 200px overlap
- Inputs: cleaned grayscale (NOT binary; Sauvola fragments thin character strokes)

Dedup by `text_normalized` (whitespace stripped from short alphanumerics) so `S 19` and `S19` collapse.

Output: `text.json` with entries `{text, text_normalized, score, bbox: [x,y,w,h], category}`.

Known OCR failure rate ~5%: things like `1000m → WoooL`, `KM 431.230 → KM 431230` (dropped period), `KM 433.170 → JKM 433.170`. These survive the score threshold sometimes. Plan: VLM fallback (Qwen2.5-VL on Colab GPU) for crops where confidence < 0.6 OR text fails regex validation. Not yet built.

### Stage 7: Text Classification (working)

Regex-based bucketing into:
- `signal_id` — `S\d+`, `SH\d+`, `C\d+`, etc.
- `track_circuit` — `\d+A?T` (e.g., `116AT`, `213T`, `241T`)
- `km_marker` — `KM\s?\d+...`
- `track_label` — `(UP|DN|DOWN).*(MAIN|LOOP|...)` with length cap (don't match full sentences)
- `point_id` — `1[01]\d` (restricted to 100-119; `200`/`850` are dimensions, not points)
- `dimension` — `\d+(\.\d+)?\s?m\.?$` (e.g., `180m`, `709.5m`)
- `note` — fallback

Do NOT use an "all caps + 4+ chars" heuristic for `station_name`; it false-positives on `STARTER`, `BSLB`, etc. Station name detection should be done by spatial position (leftmost/rightmost text in diagram band), not text patterns.

### Stage 8: Track Detection (BROKEN, deferred until after Stage 9)

Current state: produces wrong output. Multiple iterations all failed. Skeleton + connected-components anchored to text labels, but labels are above-or-below ambiguous and the skeleton fragments at every text/symbol crossing.

Failure modes encountered (do not repeat these):

1. Hough lines + Y clustering caught page frame, table borders, missed turnout curves
2. Skeleton + connected components fused parallel tracks at platforms/buildings into mega-components
3. Skeleton with horizontal `MORPH_CLOSE` to bridge gaps fused parallel tracks (DN MAIN + UP MAIN became one component)
4. Anchor-by-Y-distance suffered from labels above/below ambiguity causing wrong assignments
5. Anchor-by-horizontal-containment was better but skeleton fragmentation meant many JSON entries per logical track and names propagated inconsistently across fragments

The structural issue: SIPs are dense with overlapping horizontal structures (tracks, text, dimension lines, signals, station buildings, platform borders). Classical CV cannot reliably separate them.

Recommended path forward: skip track detection until symbol detection works. Once we have signal-post bboxes from symbol detection, tracks can be defined as the lines passing through Y-positions where signals attach. Signals are localized and reliable; tracks are the harder geometric problem. Re-attempt tracks AFTER symbols are working.

If symbols give us 50+ spatial anchors per SIP, track detection becomes "find horizontal lines that pass through these anchor Y positions" — a much easier problem.

### Stage 9: Symbol Detection (WORKING, untuned)

Step 1: Library extraction (DONE in `notebooks/02_build_library.ipynb`)

Extracts symbol exemplars from reference PDFs in `data/refs/`:
- `Types_of_signals.pdf` — 21 pages, 3-column table, ~37 rows, 23 with images, 33 exemplars total
- `Types_of_signals_ncr.pdf` — 9 pages, free-flowing format, real-SIP-quality scans, 12 rows, 27 exemplars total (includes the S/B station-building box)

Produces `library/<class_name>/exemplar_NN.png` plus `library/index.json` with 31 distinct classes, 60 exemplars total.

Step 2: Detection methods (DONE in `sip_extractor/symbols.py`, `notebooks/03_detect_symbols.ipynb`)

Two complementary approaches against the cropped binary:

- Template matching (OpenCV `matchTemplate`) for distinctive simple shapes: km-marker triangle, S/B station building box, BSLB box, GL gate-lodge box. Multi-scale at [0.5, 0.75, 1.0, 1.25, 1.5], TM_CCOEFF_NORMED ≥ 0.7, per-class NMS at IoU 0.3.
- DINOv2 nearest-neighbor for chained-ellipse signal posts (Distant, Home, Starter, Shunt, Calling-on). Connected components of the binary, filtered by area (200..50_000 px), aspect ratio (0.2..5.0), and overlap with OCR text bboxes. Each candidate is letterboxed to 224x224 RGB and embedded via DINOv2-base CLS token; cosine similarity ≥ 0.6 against exemplar embeddings assigns the class. Per-class NMS at IoU 0.3.

`Library` in `sip_extractor/symbols.py` keyword-maps the priority class names to the user's slug names from `index.json` (so build_library's exact slugs don't have to match a fixed list).

Output schema (TypedDict in `sip_extractor/schema.py`):

```json
{
  "id": "symbol_001",
  "type": "symbol",
  "class_name": "MainHomeWithoutJR",
  "bbox": [100, 200, 80, 40],
  "confidence": 0.87,
  "method": "dinov2_nn",
  "anchored_text_id": "S15"
}
```

`anchored_text_id` is set later by Stage 10 composition.

Scope for first iteration: 9 most common classes — Distant, Home, Starter, Shunt, Calling-on, KM marker, S/B box, BSLB, GL. Add more later by extending `PRIORITY_CLASSES` in `symbols.py`.

Dependencies: `pip install transformers torch` (faiss not used — brute-force NN over 60 exemplars is microseconds).

Tuning notes (todo on next pass):

- The 0.7 template threshold may be too strict for cross-modal matching (reference-PDF exemplars vs scan-quality SIP binary). Lower to ~0.5 if recall is poor.
- Multi-scale `[0.5..1.5]` may need to widen to `[0.25..2.0]` if reference DPI differs significantly from SIP DPI.
- Aspect ratio gate `[0.2..5.0]` may filter out very wide or very tall posts. Check first-pass overlay before adjusting.

### Stage 9.5: VLM Fallback for OCR (NOT BUILT)

For text entities where PaddleOCR confidence < 0.6 OR `text_normalized` fails all regex patterns, re-recognize the crop using Qwen2.5-VL-3B on Colab GPU.

Add `engine` field to text entries: `"paddleocr"` or `"qwen-vl"`.

Prompt: "What text is shown in this image? Output only the text, nothing else."

Runtime: ~0.3s/crop on T4. Expect ~20-50 crops needing fallback per SIP, so ~10-15s per SIP overhead.

### Stage 10: Composition (NOT BUILT, depends on Stage 9)

After symbols are detected, group atom symbols into typed signals. E.g., a `MainHomeWithJR` is composed of: signal post + 4 ellipses + JR marker box. Need a rule engine that says "post + N ellipses within X pixels = a Home signal."

### Stage 11: Final JSON (NOT BUILT)

Merge tracks, symbols, text, and turnouts into one canonical document.

## Schema Decisions Already Made

- Tracks use polylines (`[[x,y], [x,y], ...]`) not bboxes. Polylines preserve curve geometry at turnouts.
- Track names come from OCR (`UP MAIN`, `DN LOOP`) not generated names. Track-circuit IDs (`213T`, `241T`) are metadata, not alternative names.
- Turnouts get their own JSON entries: `{id, type: "turnout", vertex: [vx, vy], connects: [track_id, track_id]}`.
- Track polylines should be Douglas-Peucker simplified (epsilon 2.0 by default, customizable).

## Key Implementation Details to Preserve

- File creation: outputs go to `outputs/<sip_name>/` relative to `PROJECT_ROOT`.
- Crop coordinate alignment: Stage 4 produces `x_min, x_max`. Apply the same crop to `gray` (Stage 6 OCR input) so coordinates align between binary and OCR outputs.
- `Image.MAX_IMAGE_PIXELS = None` at top of imports — engineering scans exceed Pillow's default decompression bomb guard.
- Memory-efficient PaddleOCR: tile-based processing prevents OOM on full-resolution SIPs.
- Models cache to `.model_cache/` on Drive when running on Colab (set via `HF_HOME`, `TORCH_HOME`, `PADDLE_HOME` env vars in setup cell).

## Tech Stack

- Python 3.10+
- PyMuPDF (`fitz`) for PDF rendering and embedded image extraction
- OpenCV for morphology, Hough lines, template matching
- scikit-image for `threshold_sauvola`, `skeletonize`
- numpy, Pillow
- PaddleOCR 3.5.0 + paddlepaddle for OCR
- transformers, torch, faiss-cpu for DINOv2 (Stage 9, to add)

## Testing Sample

The PDF `MUNDEWADI-Signaling_Interlocking_Plan__SIP__from_Railways.pdf` is the primary test SIP. At 150 DPI it's 2945x17515; at 300 DPI it's ~5890x35030. After cropping it becomes ~2945x14476 at 150 DPI. Has 4 main tracks: DN LOOP, DN MAIN, UP MAIN, UP LOOP. OCR produces ~617 text detections on this SIP.

## Where Things Run

| Stage | Description | Local CPU | Colab CPU | Colab GPU |
|-------|-------------|-----------|-----------|-----------|
| 1-5   | Preprocessing | yes (preferred) | yes | overkill |
| 6     | PaddleOCR | yes (~30-60s) | yes (~30s) | yes (~5-10s) |
| 7     | Text classify | yes (instant) | yes | yes |
| 8     | Tracks (broken) | n/a | n/a | n/a |
| 9     | Symbol detection | possible (~10 min) | possible | yes (preferred, ~30s) |
| 9.5   | VLM fallback | painful (~30s/crop) | slow | yes (preferred, ~0.3s/crop) |

Develop Stages 1-7 locally. Run Stages 9+ on Colab.

## Immediate Action Items for Claude Code

In priority order:

1. Set up the package skeleton. Create `pyproject.toml`, `requirements.txt`, `.gitignore`, the `sip_extractor/` package with empty modules, and the `notebooks/` directory.

2. Refactor the working preprocessing + OCR + text-classification pipeline (Stages 1-7) out of `sip_preprocessing.ipynb` into `sip_extractor/preprocessing.py`, `sip_extractor/ocr.py`, `sip_extractor/classify.py`. Each module should expose a clean function that takes input paths and writes outputs.

3. Create `notebooks/01_preprocess.ipynb` as a thin entry point that imports from the package and runs Stages 1-7 on a configurable input PDF. Include the Colab+VS Code setup cell from above.

4. Move `build_library.ipynb` to `notebooks/02_build_library.ipynb` and make sure it works with the new directory layout.

5. Build symbol detection (Stage 9) in `sip_extractor/symbols.py`. Library is already extracted. Start with template matching (CPU, fast). Then add DINOv2 NN for signal posts. Create `notebooks/03_detect_symbols.ipynb` as the entry point — this is the first notebook that should be developed primarily on Colab.

6. Add VLM fallback for low-confidence OCR (Stage 9.5). Add `engine` field to text entries. Update `notebooks/01_preprocess.ipynb` or split into a separate notebook depending on what feels cleaner.

7. Re-attempt track detection (Stage 8) AFTER symbol detection works. Use signal positions as Y-anchors. Do not repeat the failed approaches listed under "Stage 8".

## Files to Read First

- This file (`HANDOFF.md`)
- `sip_preprocessing.ipynb` — current pipeline (Stages 1-8 in one notebook, will be split during refactor)
- `build_library.ipynb` — symbol library builder (will move to `notebooks/02_build_library.ipynb`)
- `library/index.json` — class catalog (after running build_library)

## What to Avoid

- Don't try to detect tracks via Hough lines + clustering (failed)
- Don't fuse skeleton fragments with `MORPH_CLOSE` (fuses parallel tracks)
- Don't classify station names by "all caps" heuristic (false positives everywhere)
- Don't use matplotlib for displaying images in notebooks (use `IPython.display.Image`)
- Don't use bbox for tracks (use polylines)
- Don't run heavy ML inference in the same cell as preprocessing — it makes the notebook slow to iterate on

## Acknowledgments

Reference PDFs are courtesy of Indian Railways signaling documentation (NCR zone) plus internal references. The S/B station-building box symbol is documented in `Types_of_signals_ncr.pdf` page 1.
