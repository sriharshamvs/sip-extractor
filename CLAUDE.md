# sip-extractor

Reusable Python pipeline that converts Indian Railways Signaling Interlocking Plan (SIP) PDFs into structured JSON describing tracks, text labels, and signal symbols. Output is intended to generalize across SIPs from many stations.

For full background, schema decisions, and rationale, see [HANDOFF.md](HANDOFF.md). This file is the operational quick reference.

## Current State

- **Stages 1-7 (working):** PDF render, color-markup removal, Sauvola binarize, edge crop, OCR (PaddleOCR 3.5.0), regex text classification. Live in [sip_preprocessing.ipynb](sip_preprocessing.ipynb).
- **Stage 8 (broken):** Track detection. Multiple iterations failed. Deferred until after Stage 9. Do not retry without reading the failure modes in HANDOFF.md.
- **Symbol library (working):** Exemplar extraction from reference PDFs in [notebooks/02_build_library.ipynb](notebooks/02_build_library.ipynb). Produces `library/<class_name>/exemplar_NN.png` plus `library/index.json`.
- **Stage 9 (working, untuned):** Symbol detection. Template matching for KM marker / S/B box / BSLB / GL; DINOv2-base nearest-neighbor for Distant / Home / Starter / Shunt / Calling-on. Module: [sip_extractor/symbols.py](sip_extractor/symbols.py). Entry point: [notebooks/03_detect_symbols.ipynb](notebooks/03_detect_symbols.ipynb). Thresholds (`0.7` template correlation, `0.6` cosine) are starting points; tune empirically per SIP.
- **Stages 9.5, 10, 11 (not built):** VLM OCR fallback, signal composition, final merged JSON.

## Repository Layout

### Actual (today)

```
sip-extractor/
├── HANDOFF.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── sip_extractor/
│   ├── preprocessing.py  ocr.py  classify.py  symbols.py  schema.py
│   └── utils/{geometry.py, io.py}
├── notebooks/
│   ├── 01_preprocess.ipynb         # Stages 1-7 (preprocess + OCR + classify)
│   ├── 02_build_library.ipynb      # symbol library extraction (one-time)
│   └── 03_detect_symbols.ipynb     # Stage 9 (template match + DINOv2 NN)
├── data/{refs,sips}/               # gitignored
├── library/                        # gitignored, regenerable
└── sip_preprocessing.ipynb         # legacy Stages 1-8 monolith (kept until Stage 8 is rebuilt)
```

### Target (per HANDOFF.md)

```
sip_extractor/                      # the package
  preprocessing.py  ocr.py  classify.py
  symbols.py  tracks.py  compose.py  schema.py
  utils/{geometry.py, io.py}
notebooks/
  01_preprocess.ipynb               # local CPU
  02_build_library.ipynb            # local CPU, one-time
  03_detect_symbols.ipynb           # Colab GPU
  04_full_pipeline.ipynb
  batch_corpus.ipynb                # Colab GPU
data/{refs,sips}/                   # gitignored
library/                            # gitignored, regenerable
outputs/<sip_name>/                 # gitignored
.model_cache/                       # gitignored, persisted on Drive on Colab
```

## Environment & Execution Model

- Python 3.10+.
- Develop Stages 1-7 on local CPU. Run Stages 9+ on Colab GPU (T4 or A100). User has Colab Pro.
- VS Code + Colab plugin: code lives locally, kernel runs on Colab.
- Project root on Colab: `/content/drive/MyDrive/sip-extractor/`. Local root: working directory.

Every notebook starts with this setup pattern (see HANDOFF.md for the full snippet):

```python
IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
# When IS_COLAB: mount Drive, set HF_HOME / TORCH_HOME / PADDLE_HOME under .model_cache/
# Else: PROJECT_ROOT = cwd
```

Caching models on Drive saves 30-60s per session for PaddleOCR and DINOv2, several minutes for Qwen2.5-VL.

## Tech Stack

- PyMuPDF (`fitz`) — PDF render and embedded image extraction.
- OpenCV — morphology, template matching, connected components.
- scikit-image — `threshold_sauvola`, `skeletonize`.
- PaddleOCR 3.5.0 + paddlepaddle.
- numpy, Pillow.
- transformers, torch, faiss-cpu — DINOv2 NN for Stage 9 (to add). Use `faiss-gpu` on Colab if needed.

## Pipeline Stages

| Stage | Name | Status | Where it runs |
|---|---|---|---|
| 1-5 | Preprocessing (render, color drop, Sauvola, crop, save) | working | local CPU preferred |
| 6 | PaddleOCR (tile-based) | working | local 30-60s, Colab T4 5-10s |
| 7 | Text classification (regex) | working | anywhere, instant |
| 8 | Track detection | **broken, deferred** | n/a |
| 9 | Symbol detection (template match + DINOv2 NN) | working, untuned | Colab GPU preferred |
| 9.5 | VLM fallback for low-confidence OCR (Qwen2.5-VL-3B) | not built | Colab GPU |
| 10 | Signal composition (atom symbols → typed signals) | not built | CPU |
| 11 | Final merged JSON | not built | CPU |

## Critical Conventions

- **Writing tone:** senior engineer, not chatbot. No em dashes. No "I'd love" / "let's dive in" phrasing. Direct, simple language in all READMEs, comments, and commit messages.
- `Image.MAX_IMAGE_PIXELS = None` at the top of any module that opens scans. Engineering scans exceed Pillow's decompression bomb guard.
- **Inline image display:** use `IPython.display.Image`. Do NOT use matplotlib in notebooks. matplotlib OOMs on Colab and holds figure state across cells.
- **Outputs:** write to `outputs/<sip_name>/` relative to `PROJECT_ROOT`.
- **Crop alignment:** Stage 4 produces `x_min, x_max`. Apply the same crop to the grayscale image used as Stage 6 OCR input so coordinates align between binary and OCR outputs.
- **Tracks are polylines** (`[[x,y], ...]`), not bboxes. Polylines preserve turnout curve geometry. Simplify with Douglas-Peucker, default epsilon 2.0.
- **Track names from OCR** (`UP MAIN`, `DN LOOP`), not generated. Track-circuit IDs (`213T`, `241T`) are metadata, not alternative names.
- **Heavy ML inference goes in its own cell**, separate from preprocessing. Keeps the notebook fast to iterate on.

## Tuned Parameters (do not casually change)

- `TARGET_DPI = 300`.
- Sauvola: `window = (DPI // 6) | 1`, `k = 0.4`.
- Edge crop: `high_density_factor = 2.0`, `bridge_gap = 2000`. The convolve-threshold-close-walk-inward method is well-tuned. Simpler approaches (morphological opening, Hough-based crop) have been tried and failed.
- PaddleOCR: `text_rec_score_thresh = 0.5` (drops `WoooL` style misreads), `use_textline_orientation = True`, tile `2000` with `200` overlap.
- OCR input is **cleaned grayscale, not binary** (Sauvola fragments thin character strokes).
- Dedup OCR by `text_normalized` (whitespace stripped from short alphanumerics) so `S 19` and `S19` collapse.
- Track polyline simplification: `epsilon = 2.0`.

## Anti-patterns (do not repeat)

- Do not detect tracks via Hough lines + Y-clustering. Catches the page frame and table borders, misses turnout curves.
- Do not fuse skeleton fragments with horizontal `MORPH_CLOSE`. Fuses DN MAIN and UP MAIN into one component.
- Do not anchor text to tracks by Y-distance alone. Labels above-or-below ambiguity gives wrong assignments.
- Do not classify station names by an "all caps + 4+ chars" heuristic. False-positives on `STARTER`, `BSLB`, etc. Use spatial position (leftmost/rightmost text in the diagram band).
- Do not use bboxes for tracks. Use polylines.
- Do not use `point_id` regex broader than 100-119. `200` and `850` are dimensions.

## Text Classification Buckets (Stage 7)

`signal_id`, `track_circuit`, `km_marker`, `track_label`, `point_id`, `dimension`, `note`. See HANDOFF.md "Stage 7" for exact regex shapes.

## Symbol Library (Stage 9 input)

Built by `notebooks/02_build_library.ipynb` from reference PDFs in `data/refs/`:
- 31 distinct classes, 60 exemplars total.
- Sources: `Types_of_signals.pdf` (33 exemplars), `Types_of_signals_ncr.pdf` (27 exemplars, includes the S/B station-building box).
- First-iteration scope for detection: 9 most common classes (Distant, Home, Starter, Shunt, Calling-on, KM marker, S/B box, BSLB, GL). The `Library` loader in `symbols.py` keyword-matches these onto the user's slug names.

## Test Sample

`MUNDEWADI-...SIP....pdf` is the primary test SIP. At 300 DPI it is roughly 5890x35030; cropped roughly 2945x14476 at 150 DPI. Has 4 main tracks: DN LOOP, DN MAIN, UP MAIN, UP LOOP. OCR produces ~617 text detections. OCR failure rate sits around 5% on this SIP and is the motivation for the VLM fallback.

## Roadmap

In priority order:

1. ~~Create package skeleton.~~ Done.
2. ~~Refactor Stages 1-7 into the package.~~ Done.
3. ~~Create `notebooks/01_preprocess.ipynb`.~~ Done.
4. ~~Move `build_library.ipynb` to `notebooks/02_build_library.ipynb`.~~ Done.
5. ~~Build Stage 9 (template matching + DINOv2 NN). Entry point at `notebooks/03_detect_symbols.ipynb`.~~ Done. Outputs untuned; first task on the next pass is to inspect the overlay against MUNDEWADI and adjust thresholds.
6. Add VLM fallback (Stage 9.5) for crops with PaddleOCR confidence < 0.6 or that fail all regex patterns. Add `engine` field (`paddleocr` | `qwen-vl`) to text entries.
7. Build Stage 10 (signal composition: atom symbols → typed signals via rules like "post + 4 ellipses + JR marker = MainHomeWithJR").
8. Re-attempt Stage 8 only after Stage 9 is tuned. Use signal-post Y positions as anchors for finding horizontal lines. Do not repeat the failed approaches above.
9. Build Stage 11 (final merged JSON: tracks + symbols + text into one canonical document).

## Files to Read for Deeper Context

- [HANDOFF.md](HANDOFF.md) — authoritative source for goals, schema decisions, failure-mode post-mortems, and detailed stage descriptions.
- [sip_preprocessing.ipynb](sip_preprocessing.ipynb) — legacy Stages 1-8 monolith; superseded by the `notebooks/` entry points but retained until Stage 8 is rebuilt.
- [notebooks/01_preprocess.ipynb](notebooks/01_preprocess.ipynb), [notebooks/02_build_library.ipynb](notebooks/02_build_library.ipynb), [notebooks/03_detect_symbols.ipynb](notebooks/03_detect_symbols.ipynb) — current per-stage entry points.
- [sip_extractor/symbols.py](sip_extractor/symbols.py) — Stage 9 detector (template match + DINOv2 NN).
- `library/index.json` — class catalog (after running 02_build_library).
