"""FastAPI backend wrapping the graded classical ALPR model (web/ — not graded).

Serves the React frontend (deployed separately to GitHub Pages) via a public endpoint:
  POST /api/analyze   image OR video upload -> detected boxes + recognized plate text
  GET  /api/results.csv   the session's recognized plates as the deliverable CSV
  GET  /api/health        liveness

The graded alpr/ core is imported unchanged (exact 15.4% pipeline); this layer only handles
HTTP, decoding, video frame-sampling, and annotation. CORS is open so the static Pages app
can call it (reached in dev via a cloudflared tunnel; in prod via a free host).
"""
import base64
import csv
import io
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from alpr.pipeline import ALPR

app = FastAPI(title="Macedonian ALPR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_alpr = ALPR(abstain_threshold=0.5)
_history: list[dict] = []          # recognized plates this session (for the CSV deliverable)

VIDEO_EXT = (".mp4", ".mov", ".avi", ".webm", ".m4v", ".mkv")
MAX_VIDEO_SAMPLES = 60             # cap frames analyzed so uploads stay responsive


def _plate_dict(r, image_name):
    return {"image": image_name, "bbox": list(r.bbox), "plate_text": r.plate_text,
            "region": r.region, "confidence": round(r.confidence, 3)}


def _annotate(img, results) -> str:
    """Draw boxes (green=read, red=detected-only) + text; return base64 JPEG."""
    vis = img.copy()
    for r in results:
        x, y, w, h = r.bbox
        col = (0, 200, 0) if r.plate_text else (0, 0, 230)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 3)
        if r.plate_text:
            cv2.putText(vis, f"{r.plate_text} {r.confidence:.2f}", (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return base64.b64encode(cv2.imencode(".jpg", vis)[1]).decode()


def _record(plates):
    for p in plates:
        if p["plate_text"]:
            _history.append(p)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "classical-mk-alpr"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "upload"
    is_video = (file.content_type or "").startswith("video") or name.lower().endswith(VIDEO_EXT)
    return _analyze_video(data, name) if is_video else _analyze_image(data, name)


def _analyze_image(data, name):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)
    results = _alpr.read_plate(img)
    plates = [_plate_dict(r, name) for r in results]
    _record(plates)
    return {"type": "image", "plates": plates, "annotated": _annotate(img, results)}


def _analyze_video(data, name):
    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1] or ".mp4", delete=False)
    tmp.write(data)
    tmp.close()
    try:
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
        stride = max(1, total // MAX_VIDEO_SAMPLES)
        best: dict[str, dict] = {}      # plate_text -> best read so far (temporal best-frame)
        annotated = None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                results = _alpr.read_plate(frame)
                reads = [r for r in results if r.plate_text]
                for r in reads:
                    if r.plate_text not in best or r.confidence > best[r.plate_text]["confidence"]:
                        d = _plate_dict(r, name)
                        d["frame"] = i
                        best[r.plate_text] = d
                if annotated is None and reads:
                    annotated = _annotate(frame, results)
            i += 1
        cap.release()
    finally:
        os.unlink(tmp.name)
    plates = list(best.values())
    _record(plates)
    return {"type": "video", "frames": i, "plates": plates, "annotated": annotated}


@app.get("/api/results.csv")
def results_csv():
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["image", "plate_text", "region", "confidence"])
    for h in _history:
        w.writerow([h["image"], h["plate_text"], h.get("region", ""), h["confidence"]])
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=results.csv"})
