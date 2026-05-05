# sip-extractor

Pipeline for converting Indian Railways Signaling Interlocking Plan (SIP) PDFs
into structured JSON describing tracks, text labels, and signal symbols.

See [HANDOFF.md](HANDOFF.md) for full background, [CLAUDE.md](CLAUDE.md) for
the operational quick reference.

## Install

```bash
pip install -e .
```

For symbol detection (Stage 9+, requires Colab GPU in practice):

```bash
pip install -e ".[symbols]"
```

## Quick start

Drop a SIP PDF in `data/sips/` and run:

```python
from pathlib import Path
from sip_extractor import preprocessing, ocr, classify

pdf = Path("data/sips/MUNDEWADI-SIP.pdf")
out = Path("outputs/MUNDEWADI-SIP")
out.mkdir(parents=True, exist_ok=True)

prep = preprocessing.run(pdf, out, target_dpi=300)
text = ocr.run(prep.gray_cropped, out)
classify.run(text, out)
```

Or use [notebooks/01_preprocess.ipynb](notebooks/01_preprocess.ipynb) as the
entry point.

## Status

Stages 1-7 (preprocessing, OCR, text classification) work. Stages 8 onward
(track detection, symbol detection, VLM fallback, composition) are not yet
in this package. See HANDOFF.md "Pipeline Stages" for the full status table.
