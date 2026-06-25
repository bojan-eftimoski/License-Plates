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
layer (`web/`) serves it, and a React app (`frontend/`) is the demo UI.

## Live demo (nothing to install)
- **Web app (GitHub Pages):** https://bojan-eftimoski.github.io/License-Plates/
- **Backend / API (Hugging Face Space):** https://bojaN244-LicensePlates.hf.space — `/api/health`, `/api/analyze`, `/api/results.csv`

The Space sleeps when idle; the first request after a while takes ~30 s to wake (the web app shows a "waking backend" status and retries).

## Run locally

The trained model ships in the repo (`data/templates.npz`), so **no dataset or template
generation is needed** — just install and run.

### Backend (the graded model + API)
```bash
pip install -r requirements.txt
uvicorn web.app:app                 # serves http://127.0.0.1:8000
```
`POST /api/analyze` (image or video → boxes + recognized text), `GET /api/results.csv`, `GET /api/health`.

### Frontend (demo UI)
```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173
```
The dev UI calls `http://127.0.0.1:8000` by default (matching the backend above). You can change
the **Backend URL** field in the UI to point at the live Space instead of running the backend.

### Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Pipeline (`alpr/`, cv2 + numpy only)
`detection` (morphology + blue-strip colour + contour cascade, plus a whole-frame fallback for
cropped/plate-only images) → `rectify` (projection-profile deskew + plate-body crop) →
`segmentation` (chromatic badge/strip removal → connected components) → `ocr` (HOG features +
`cv2.ml.KNearest`) → `validation` (MK `LL DDD(D) LL` format + 34-code region whitelist +
confusable resolution). *Detect-always, read-when-confident*: every plate gets a box; a string is
emitted only above the confidence threshold (otherwise it abstains rather than guess wrong).

Clean, frontal, well-sized plates read reliably; small / low-resolution / angled / degraded plates
abstain — the inherent ceiling of classical CV without learning. Per-character accuracy is ≈ 78 %
on plates that segment correctly. See `docs/superpowers/specs/` for the design and `data/README.md`
for the dataset.

## Project layout
- `alpr/` — the classical pipeline (detection, rectify, segmentation, ocr, validation, types, features)
- `web/app.py` — FastAPI wrapper (`/api/...`)
- `frontend/` — React + TypeScript UI (upload, video, live-camera AR, EN/MK)
- `data/templates.npz` — trained glyph dataset (synthetic DIN-1451 + real glyphs); the model
- `tools/` — dataset/build utilities (not needed to run the app)
- `Dockerfile`, `requirements.txt` — backend container for Hugging Face Spaces (port 7860)

## Deploy
Backend is a **Docker Space** on Hugging Face (build from `Dockerfile`, `requirements.txt`,
`alpr/`, `web/`, `data/templates.npz`; serves on port 7860 over HTTPS). The frontend builds to
static files and is published to GitHub Pages, calling the Space's `/api/analyze`.
