# Dataset — Macedonian ALPR (classical pipeline)

This dataset supports a **classical-CV** license-plate pipeline (no deep learning, no Tesseract).
It is **not** used to train a neural net — only to (a) train/validate the `cv2.ml.KNearest` character
classifier and (b) evaluate detection + OCR accuracy.

> **GLPD note:** the Global License Plate Dataset (`siddagra/Global-Licenseplate-Dataset`) was the
> originally-planned real source, but its Hugging Face repo is **gated/withdrawn** (401 anonymously,
> 404 when logged in) and there is no accessible mirror. It is therefore **not used**. The character
> templates are instead **synthetic-primary** (rendered from an OFL DIN-1451 font), with real glyphs
> mixed in; real images come from the non-gated sources below.

## Sources

| Source | What | Count | Use | License / attribution |
|---|---|---|---|---|
| **Synthetic (DIN-1451)** | Rendered + augmented glyphs, 32 classes | **4,800** | KNN training backbone + templates | Font: *Alte DIN 1451 Mittelschrift*, Peter Wiegel, **OFL-1.1** |
| **olavsplates.com** | Labeled MK plate photos (full + tight close-ups); filename = plate | 24 | Real glyph crops (templates); reference | Enthusiast site — **attribute**, not redistributed (gitignored) |
| **Platesmania.com** (user-uploaded) | Real in-traffic photos, filename = plate; 7 regions, 19×4-digit + 4×3-digit | 23 | **Held-out eval** + detector tuning | platesmania.com — **attribution link required** |
| **Wikimedia Commons** | Cat. "License plates of North Macedonia" (filtered) | 22 | Reference / extra eval (per-file license) | Per-file CC-BY-SA / PD — record per file |
| **Repo `Images/` + `Videos/`** | 6 photos (`registracii*`, `IMG_*`) + 3 videos | 6 + 3 | Primary graded eval + detect-only/hard + negatives | Project-owned |

**Real glyph extraction:** `extract_real_glyphs.py` rectifies/segments the olavsplates close-ups and
auto-labels glyphs from the filename plate string, keeping only plates whose segment count matches the
label length (so bad segmentations never pollute the set). Current yield: **21 real glyphs** (the quick
segmenter is intentionally conservative; the production `alpr/segmentation.py` will raise this in M1/M2).

## Layout

```
data/
  raw/                     # gitignored — third-party / bulk originals
    fonts/                 # din1451alt.ttf (OFL) + variant
    olavsplates/           # + manifest.csv (labels)
    platesmania_mk/        # + manifest.csv (labels)
    wikimedia_mk/          # + manifest.csv (license/author per file)
  templates/               # gitignored — KNN glyph crops, 32 class folders (0-9, A-Z minus Q/W/X/Y)
    <CLASS>/synth_*.png    # synthetic
    <CLASS>/real_*.png     # extracted real
    _meta.csv              # file,class,source(synth|real)   [tracked]
  demo/                    # gitignored — curated live-demo images + manifest.csv [tracked]
  eval/
    ground_truth.csv       # file,plate_string,region,source,difficulty,split   [tracked]
```

Bulk/binary/third-party images are **gitignored**; provenance + labels live in the tracked
`manifest.csv` / `_meta.csv` / `ground_truth.csv`, and everything is **reproducible** from `tools/dataset/`.

## Current totals (real, verified)

- **Templates:** 4,821 glyphs (4,800 synthetic + 21 real) across all **32 classes**.
- **Held-out eval:** 23 distinct labeled plates (20 Platesmania + 3 in `registracii1`), 7 regions (BT, GE, GV, OH, PP, SK, SU); **5 detect-only** hard repo images; 3 videos.
- **Demo set:** 13 curated images (`data/demo/`).

## Reproduce

```
.venv/Scripts/python tools/dataset/scrape_olavsplates.py      # olavsplates (+labels)
.venv/Scripts/python tools/dataset/synth_glyphs.py --per-class 150   # 4,800 synthetic glyphs
.venv/Scripts/python tools/dataset/extract_real_glyphs.py     # real glyphs from close-ups
.venv/Scripts/python tools/dataset/build_eval.py              # ground_truth.csv
# Wikimedia + Platesmania: see manifests; Platesmania is anti-bot-walled (manual download).
```

## Attribution (for the seminar report)

- Synthetic glyphs rendered from **Alte DIN 1451 Mittelschrift** (Peter Wiegel, OFL-1.1).
- Real plate images: **olavsplates.com** and **platesmania.com** (attribution links required),
  **Wikimedia Commons** (per-file CC-BY-SA / public domain — cite author + license per image used).
- No GLPD data is used (repo inaccessible).
