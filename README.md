---
title: Macedonian ALPR (classical CV)
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Macedonian License-Plate Recognition — classical computer vision

A seminar project for *Дигитално процесирање на слика* (FINKI). Detects Macedonian license
plates and recognizes their characters using **classical computer vision only** — OpenCV +
numpy, **no deep learning, no Tesseract**. The graded model lives in `alpr/`; a thin FastAPI
layer (`web/`) serves it for the demo.

## Pipeline (`alpr/`, cv2 + numpy only)
`detection` (morphology + blue-strip colour + contour cascade) → `rectify` (projection-profile
deskew + plate-body crop) → `segmentation` (chromatic badge/strip removal → connected components)
→ `ocr` (HOG features + `cv2.ml.KNearest`) → `validation` (MK `LL DDD(D) LL` format + 34-code
region whitelist + confusable resolution). *Detect-always, read-when-confident*: every plate gets
a box; a string is emitted only above the confidence threshold.

**Held-out accuracy (real Platesmania/olavsplates plates):** plate-exact 15.4%, char-level 19.6%.
Clean/frontal plates read perfectly; small/low-res/degraded plates abstain (the classical ceiling).
See `docs/superpowers/specs/` for the design and `data/README.md` for the dataset.

## Run locally
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # dev (incl. tests)
.venv/Scripts/python tools/dataset/synth_glyphs.py        # generate data/templates/ (first run)
.venv/Scripts/python tools/dataset/export_templates.py    # -> data/templates.npz
.venv/Scripts/uvicorn web.app:app --reload                # http://127.0.0.1:8000
```
Endpoints: `POST /api/analyze` (image or video → boxes + recognized text), `GET /api/results.csv`,
`GET /api/health`.

## Deploy (Hugging Face Spaces — permanent demo host)
This repo is a Docker Space: create a **Docker** Space and push these files (the `Dockerfile`,
`requirements.txt`, `alpr/`, `web/`, `data/templates.npz`). Spaces builds the image and serves on
port 7860 over HTTPS — a laptop-independent URL openable from the iPhone (wake it before presenting
to clear the cold-start). For local iPhone testing, expose the local server with a cloudflared
tunnel instead. The same `web/app.py` runs unchanged in both.

The React frontend (upload UI) is built separately and deployed to GitHub Pages, calling this
backend's `/api/analyze`.
