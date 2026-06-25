# Design: Macedonian License-Plate Detection & Recognition (ALPR)

**Course:** Дигитално процесирање на слика (Digital Image Processing), FINKI
**Type:** Seminar project (individual)
**Date:** 2026-06-25
**Status:** Approved design (revised after adversarial spec review) — pending implementation plans

---

## 0. Locked decisions (authoritative checklist)

Persisted here so implementers/reviewers check against the real agreed list, not a reconstruction.

1. **Methodology:** classical detection **and** classical OCR only — **no deep learning, no Tesseract / pre-trained OCR.**
2. **Input scope:** handle the full difficulty range, but via *detect-always / read-when-confident* + video best-frame (see §4).
3. **Grading:** algorithms **and** accuracy rewarded equally; **only the Python core is graded**, not UI/framework.
4. **Output file:** CSV.
5. **Behavior:** always draw a detection box (red); emit a string only when confident (green); otherwise abstain. Video tracks plates and reads the best/sharpest frame with per-character majority voting.
6. **Toolbox:** full classical toolbox allowed (multi-scale, MSER, color cues, perspective rectification, morphology, adaptive/Otsu threshold).
7. **OCR templates:** render from a plate font **plus** augment with real crops.
8. **Eval data:** label the provided 6 images + 3 videos; student supplies ~30 stratified images + 3–6 approach videos.
9. **Backend:** one FastAPI service serving the model + frontend.
10. **Persistence:** no database; CSV file + in-memory session history.
11. **Real-time:** full WebSocket streaming; `getUserMedia` frames downscaled; server keeps only the latest frame.
12. **Detection strategy:** hybrid cascade "C" (morphology + blue-strip color + MSER → score/merge → geometric + format verify → perspective rectify).
13. **Classifier:** KNN via `cv2.ml.KNearest`; **no scikit-learn.**
14. **Build order:** M0 git init → M1 OCR → M2 detector stage-A + rectify + segment → M3 add color + MSER cues + threshold tuning → M4 video → M5 web → M6 PPT.

---

## 1. Purpose & context

Implement a demo application that takes a photo (or video) of a vehicle, **detects the
Macedonian license plate, recognizes its characters via classical image processing, and
writes the results to a file**. The work must be visibly **based on the course materials**
in `Materials/` (color models, histograms, filtering, edges, morphology, thresholding,
geometric transforms, keypoints/regions) — see decision 1.

Deliverables:
1. The demo application (graded core + thin web wrapper).
2. A PowerPoint presentation, **≥20 slides, in Macedonian**, describing the algorithms.

### 1.1 Grading constraints

Per decision 3: only the Python image-processing / OCR core is graded; framework and UI are
not. The core is built **test-first (TDD)** as a clean, independently testable library; the
web app is a thin caller. The graded `alpr/` package imports **only `cv2` + `numpy` at
inference time** (see §9 for the toolset boundary).

### 1.2 Success criteria

- **Gating criterion (hard, testable):** *precision-first* — **zero wrong reads above the
  confidence threshold** on the eval set (§10). This is THE bar a milestone must clear.
- **Reported-only metrics (tracked, not gating):** detection recall, character accuracy,
  plate-exact accuracy, and abstain rate, stratified by difficulty. Concrete numeric targets
  (e.g. plate-exact ≥ X% on easy) are **placeholders to be set once the eval set exists**
  (§8) — they are tracked over milestones, not pass/fail gates.
- Every algorithmic stage maps to a concrete `Materials/` topic (for the presentation).

---

## 2. Domain facts: Macedonian plates

Post-2012 format, left→right: **`[blue NMK strip] [LL region] [chromatic badge] [DDD(D)] [LL]`**,
e.g. `SK ⬛ 9507 BT`. Black characters on white. Dimensions 520 × 110 mm
(**aspect ratio ≈ 4.73:1**).

- **Readable glyph count: 7 or 8** = 2 region letters + **3–4 digits** (4 modern, 3 legacy)
  + 2 suffix letters. The validator accepts both lengths.
- **Non-character regions to exclude during segmentation:** the **blue NMK strip** (left
  edge) and the **central red/orange chromatic badge** (Cyrillic coat-of-arms emblem
  between the region code and the digits). The badge grayscales to a mid-tone (~128) and
  **survives Otsu/adaptive thresholding as a glyph-height blob**; if not removed it is
  mis-segmented as a phantom extra character and breaks slot mapping. Both are removed via a
  **chromatic (high-saturation) mask** before binarization (§5.2 step 4). The badge gap is
  also a useful anchor that splits the line into the left letter group and the right
  digits+letters group.
- **Character set:** Latin only; **Q, W, X, Y never used** ⇒ alphabet = 22 letters + 10
  digits = 32 glyphs.
- **Confusables — what position actually buys us:** slot position resolves **digit↔letter
  cross-confusions only** (0↔O, 1↔I, 2↔Z, 5↔S, 8↔B), because letter-slots and digit-slots
  are known. It does **not** resolve intra-class confusions (letter↔letter like O↔D, C↔G,
  E↔B; digit↔digit like 0↔8) — those are handled by the classifier and, for the **first two
  letters**, by the **region-code whitelist** below.
- **Region codes (authoritative, 34):** `BE BT DB DE DH DK GE GV KA KI KO KR KP KS KU MB MK
  NE OH PE PP PS RA RE SK SN SU SR ST TE VA VE VI VV`. The first two letters MUST be one of
  these; the suffix two letters may be any of the 22 allowed letters. (All 34 codes lie
  within the allowed alphabet — no Q/W/X/Y.) The validator must **not overfit to `SK`** even
  if only `SK` samples are supplied.

---

## 3. Scope & non-goals

**In scope:** detection + recognition on still images and video (file upload) and a live
camera feed; detect-always/read-when-confident behavior; CSV output; ≥20-slide Macedonian PPT.

**Non-goals:** no deep learning / Tesseract; no database; no production deployment/auth; no UI
polish this phase (native styling only). The **live camera feed is a non-graded demo
convenience** (decision 11) and intentionally produces no CSV row — it is not part of the
graded "write results to a file" core.

---

## 4. Behavioral model: detect-always, read-when-confident

Resolves "classical-only" vs. "handle all images":

- **Detection always shows** every plate-like region as a box (red) — the system visibly
  "sees" plates even when it cannot read them.
- **A string is emitted only when confidence ≥ threshold** (box turns green). On
  tiny/blurred/extreme-angle plates the system **abstains** (`plate_text = None`, box
  retained) rather than guessing wrong. A ~10 px plate has too little signal to read; abstain
  keeps precision high and is honest about classical limits.
- **Video turns "unreadable" into "readable":** plates are tracked across frames; OCR runs on
  the **best frame(s)** (largest, sharpest, most frontal as the car approaches) and the
  characters are **majority-voted** across reads.

---

## 5. Architecture

Two decoupled layers. The graded core knows nothing about the web app; both the CSV path and
the web app call the same `read_plate()`.

```
repo root
├── alpr/            # GRADED CORE — pure classical-CV library + CLI (cv2 + numpy only at inference)
├── web/             # thin FastAPI service + minimal responsive frontend (not graded)
├── data/
│   ├── templates/   # rendered plate-font glyphs (+ augmentation) and real crops; trained KNN model
│   └── eval/        # labeled test images + ground_truth.csv
├── tests/           # pytest: unit per module + end-to-end accuracy harness
├── docs/            # this spec, slide source notes
├── Materials/ Images/ Videos/   # existing, untouched
└── requirements.txt
```

### 5.1 Graded core modules (`alpr/`)

| Module | Responsibility | Materials topic |
|---|---|---|
| `pipeline.py` | `read_plate(image) -> list[PlateResult]`; orchestrates all stages | (integration) |
| `preprocess.py` | working-scale resize, grayscale, illumination normalize (CLAHE) | Histograms |
| `detection.py` | hybrid cascade: multi-cue candidates → score → NMS merge | Edges, Morphology, Color Models, Regions |
| `rectify.py` | locate plate quad → perspective-warp to canonical rectangle (with fallback ladder) | Geometric Transformations |
| `segmentation.py` | chromatic badge/strip removal → binarize → connected components → ordered glyph crops | Color Models, Thresholding, Morphology, Region labeling |
| `ocr.py` | normalize glyph → HOG/zoning/moment features → `cv2.ml.KNearest` → char + confidence | Template matching, keypoints/CBIR descriptors |
| `validation.py` | MK format (7–8 glyphs, 3–4 digits) + region-code whitelist; confusable resolution; confidence | (domain rules) |
| `video.py` | track plates (IoU + fallback association) → best-frame selection → temporal char voting | Edge detection (sharpness), Image subtraction |
| `results.py` | dataclasses + CSV writer | — |
| `cli.py` | `python -m alpr <folder|file> -o results.csv` batch runner | — |

### 5.1.1 Interfaces & data contracts (pin down for TDD)

Public entry point is **name-agnostic** — it takes a bare image and returns results; the
caller (CLI / web) supplies the filename for the CSV.

```python
read_plate(image_bgr: np.ndarray) -> list[PlateResult]          # multiple plates per image

# intermediate types passed between modules:
Candidate     { bbox:(x,y,w,h), quad:Optional[4pts], score:float, cue:str }   # detection.py
RectifiedPlate{ warp:np.ndarray, quad:4pts, ok:bool }                          # rectify.py
GlyphCrop     { norm:np.ndarray(32x32 uint8 binary), bbox:(x,y,w,h), index:int}# segmentation.py
CharResult    { value:str, confidence:float, slot:'L'|'D' }                    # ocr.py + validation
PlateResult   { bbox:(x,y,w,h), quad:4pts, plate_text:Optional[str],
                region:Optional[str], confidence:float, chars:list[CharResult] }

# module signatures:
preprocess(image_bgr)            -> Preprocessed{ bgr, gray, scale }
detect_candidates(pre)           -> list[Candidate]
rectify(image_bgr, candidate)    -> RectifiedPlate
segment(rectified)               -> list[GlyphCrop]      # badge/strip excluded; len 7 or 8
classify(glyph)                  -> CharResult           # value+confidence, slot filled by validation
validate(chars, region_codes)    -> (plate_text|None, region|None, confidence, slotted chars)
```

**Glyph-normalization contract (shared by `segmentation.py` and `ocr.py` — must match
exactly or accuracy collapses at integration):** per-glyph deskew (image moments) → Otsu
binarize → tight bounding-box crop → aspect-preserving resize into **32×32** with padding →
stroke-width normalization. Both the OCR training templates and the runtime segmenter emit
glyphs in this exact form. An M1 fixture uses a **real segmented crop** (not only rendered
glyphs) to test this seam early.

### 5.2 Single-image pipeline (cascade "C")

1. **Preprocess** — resize to a working scale; grayscale; optional CLAHE for shade/dusk.
2. **Detect (three independent cues — none is mandatory, so one failing cue never loses the
   plate):**
   - **(a) Morphology/edge:** **blackhat** (deliberately, since plates are *dark glyphs on a
     light background* — the opposite polarity to the barcode example, so blackhat not tophat)
     + Sobel-x gradient + morphological close + Otsu → candidate boxes.
   - **(b) Color:** HSV detection of the **blue NMK strip** (broad hue band for white-balance
     robustness) as one anchor; do not *require* it and do not hard-code "plate is to its
     right" — feed it into scoring alongside (a)/(c).
   - **(c) MSER:** stable regions grouped into a collinear, similar-height character line.
   - **Merge** overlapping candidates via non-max suppression; **score** by aspect-ratio fit
     (≈4.73), fill ratio, and edge density.
3. **Rectify (fallback ladder, never abort):** (1) `minAreaRect`/`approxPolyDP` of the
   largest contour → 4-corner perspective warp; (2) if not 4 vertices, fall back to the
   **rotated bounding box** (deskew by angle only); (3) if angle unstable, segment on the
   **axis-aligned crop**. Validate the warp by output **aspect ratio ≈ 4.73** and presence of
   the **blue strip on the left edge**; reject degenerate warps and drop to the next rung.
4. **Segment:** on the rectified RGB crop, compute per-pixel **saturation** and **zero out
   high-saturation (chromatic) regions** → removes the blue strip and the central badge.
   Then Otsu/adaptive binarize → connected components (vertical-projection fallback) → filter
   glyph components by size/aspect → order left-to-right using the **badge gap** as the split
   anchor. Expect **7 or 8** glyph crops.
5. **OCR:** normalize each glyph per the §5.1.1 contract; extract **HOG / zoning / Hu-moment
   features** (not raw pixels — far more robust to dirt/shadow/embossing) → `cv2.ml.KNearest`
   against rendered+augmented templates → character + per-glyph confidence (§5.4).
6. **Validate:** the digit count is unknown a priori, so **try both the 7-glyph (3-digit) and
   8-glyph (4-digit) `LL DDD(D) LL` slot templates, score format-match, keep the best.** Map
   each glyph to its slot, apply digit↔letter confusable resolution by slot, enforce
   **region ∈ the 34-code whitelist** (hard gate) and no Q/W/X/Y. Compute confidence (§5.4).
7. **Abstain gate:** if confidence < threshold (or format/region gate fails), keep `bbox`,
   set `plate_text = None`.

### 5.3 Video path (`video.py`)

- Sample frames at a stride; detect per sampled frame.
- **Track** plates into tracklets. Primary association = IoU; **fallback when IoU is low**
  (fast/approaching car — boxes may not overlap between sampled frames): match by
  box-center proximity + similar aspect + **monotonic size growth** (a plate grows as it
  approaches) and/or normalized cross-correlation of the previous plate crop; predict the
  next box (constant-velocity/growth extrapolation) and gate association on the *predicted*
  box. **Reduce the stride while a tracklet is active.**
- Per tracklet: score frames by **size × sharpness** (variance-of-Laplacian), OCR the top-K
  sharpest, **majority-vote each character position** → one stable string per car; emit when
  the vote stabilizes / confidence clears the threshold.

### 5.4 Confidence & abstain threshold (concrete, monotonic — not "calibrated probability")

- **Per glyph:** use the KNN **distance margin / ratio test** — `r = d1/d2` (nearest vs.
  second-nearest class distance); confidence `c = clamp(1 − r, 0, 1)`. A unique match
  (`r→0`) ⇒ `c≈1`; an ambiguous one (`r→1`) ⇒ `c≈0`. This is comparable across glyphs in a
  way a raw L2 distance is not, and is course-adjacent (keypoint matching ratio test).
- **Per plate:** `confidence = min_i c_i × format_ok`, where `min` is the conservative
  weakest-link aggregation (precision-first) and `format_ok ∈ {0,1}` is a **hard gate**
  (valid 7–8 slot pattern AND region ∈ whitelist). Per-slot digit/letter agreement is part
  of the hard gate, not a soft multiplier.
- **Threshold τ:** chosen as the operating point on the eval set where **errors above τ = 0**,
  plus a safety margin; document the **small-sample caveat** (~30 images is thin). Until M3
  calibration data exists, M2 uses a **conservative provisional default** (favoring abstain)
  so the precision-first invariant is never violated before calibration.

### 5.5 Output file (CSV)

Columns: `image_name, plate_text, region, confidence, bbox_x, bbox_y, bbox_w, bbox_h`. One row
per detected plate. **`plate_text` is the empty string (not the literal "None")** when
abstaining. **`image_name` is injected by the caller** (e.g. `append_row(image_name, r)` /
`write_csv(list[tuple[str, PlateResult]])`). **Header written once on file create**, rows
appended thereafter (single-process assumption — no DB). Box-drawing / JSON serialization for
the web layer lives in `web/`, **not** in `results.py` (which stays CSV-only).

---

## 6. Web wrapper (`web/` — thin, not graded)

One **FastAPI** service that imports `alpr` and serves a minimal responsive frontend. None of
`fastapi`/`uvicorn`/the annotate helper is ever imported by `alpr/`.

- `POST /api/analyze` — multipart upload (image *or* video) → runs `read_plate()` or the
  video path → returns JSON (boxes, `plate_text`, region, confidence, annotated preview);
  appends rows to the CSV + in-memory session history. The annotate/`to_json` helper lives
  here, outside the graded core.
- `WS /ws/realtime` — live camera (non-graded demo). Browser captures rear-camera frames
  (`getUserMedia`, `facingMode:"environment"`), downscales to ~640 px on an offscreen canvas,
  sends JPEG frames. Server keeps **only the latest frame** (drops backlog → no lag), runs
  detection per frame and throttled OCR / on tracklet stabilization, streams back
  `{boxes, reads}`. Browser overlays on a `<canvas>` (red = detected, green = read).
- `GET /api/results.csv` — download the accumulated CSV deliverable.

**Frontend:** one responsive page, three modes (Upload image / Upload video / Live camera),
drawn boxes, results table, "Download CSV". Native styling only.

**Known gotcha (designed around):** `getUserMedia` requires `localhost` or **HTTPS**. To use
the live camera on a **phone** against the laptop server, the service ships a one-command
**HTTPS dev mode** with a self-signed cert (or tunnel).

---

## 7. Deliverable 2 — presentation (≥20 slides, Macedonian)

~22 slides via `document-skills:pptx`, each algorithm slide mapped to a `Materials/` topic:

1. Наслов · 2. Проблем и цел · 3. Формат на МК таблички (NMK, регион, 3–4 цифри, 2 букви;
без Q/W/X/Y; 34 региони; централен амблем) · 4. Преглед на системот (дијаграм) ·
5. Претпроцесирање (resize, grayscale, CLAHE) · 6. Детекција — каскада преглед ·
7. Cue 1: морфологија + рабови (blackhat) · 8. Cue 2: боја — сина NMK лента (HSV) ·
9. Cue 3: MSER региони · 10. Спојување + скорирање (NMS, aspect ratio) · 11. Перспективна
корекција (+ fallback) · 12. Сегментација + отстранување на амблемот (хроматска маска) ·
13. OCR — темплејти од фонт + HOG/моменти + KNN · 14. Валидација (7–8 слотови, 3–4 цифри) +
регион whitelist + разрешување 0/O, 1/I · 15. Доверба и праг (detect-always /
read-when-confident) · 16. Видео — следење, најдобар кадар, гласање · 17. Резултати (точност
по тежина) · 18. Успешни примери · 19. Ограничувања · 20. Демо апликација · 21. Заклучок /
идни подобрувања · 22. Референци + мапирање кон материјалите.

---

## 8. Dataset (built — see `data/README.md`)

**GLPD was dropped** (the planned real source is gated/withdrawn: 401 anonymously, 404 logged in,
no mirror). The dataset is therefore **synthetic-primary** for OCR, with real images from non-gated
sources for evaluation. Bulk/third-party images are gitignored; provenance + labels live in tracked
`manifest.csv`/`_meta.csv`/`ground_truth.csv`, and everything is reproducible from `tools/dataset/`.

- **KNN templates — 4,821 glyphs across all 32 classes** (0–9, A–Z minus Q/W/X/Y):
  **4,800 synthetic** rendered+augmented from the OFL **DIN-1451 Mittelschrift** font (`synth_glyphs.py`)
  + **21 real** glyphs auto-extracted from labeled olavsplates close-ups (`extract_real_glyphs.py`,
  length-checked so bad segmentations never pollute). Real-glyph yield rises once the production
  `alpr/segmentation.py` replaces the quick extractor (M1/M2).
- **Held-out eval — `data/eval/ground_truth.csv`:** 23 distinct labeled plates (20 Platesmania,
  user-supplied; + 3 in `registracii1`), 7 regions (BT/GE/GV/OH/PP/SK/SU), 19×4-digit + 4×3-digit;
  plus **5 detect-only** hard repo images and **3 videos**. olavsplates is excluded from eval (its
  close-ups feed templates → avoids leakage).
- **Demo set — `data/demo/` (13 images):** curated clean/frontal/sharp cases for the live demo
  (to be re-ranked by real confidence once the pipeline runs, M2).
- **Font:** Macedonian plates use a DIN-1451/EU-style face (confirmed **not** FE-Schrift); templates
  use *Alte DIN 1451 Mittelschrift* (Peter Wiegel, **OFL-1.1**).

**Gaps / follow-ups:** real-image pool is modest (~45 labeled plates total incl. olavsplates), so
synthetics carry the KNN load; Wikimedia (22, mostly unlabeled) and more Platesmania uploads can grow
the eval set; report accuracy on the **real held-out split only** so synthetics never inflate it
(§10). No-`SK`-overfit guardrail retained.

---

## 9. Dependencies & toolset boundary

`opencv-python`, `numpy`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `Pillow`,
`pytest`; optional `matplotlib`. **Toolset rule (decision 1/13):** the graded `alpr/` core
imports **only `cv2` + `numpy`** on the inference/recognition path. **Pillow is used only
offline** to render template glyphs (a data-prep step, never imported at inference).
`fastapi`/`uvicorn`/`matplotlib` are **web/reporting only** and never imported by `alpr/`.
KNN is OpenCV's own `cv2.ml.KNearest` — **no scikit-learn**, no deep-learning libs, no Tesseract.

**Plate font:** Macedonian post-2012 plates use a **DIN-1451 / EU-style** font (NOT German
FE-Schrift — its anti-counterfeit shapes differ and would systematically mismatch). Source the
closest available EU-plate font for rendering; lean on real-crop templates (§8.1) +
augmentation (blur/rotate/noise) to close the residual gap; document the chosen font.

---

## 10. Testing strategy (TDD — graded core)

- **Unit tests per module** against fixtures:
  - `detection`: crops with known plate bboxes → assert per-cue recall at IoU ≥ threshold
    (gives the three-cue stage a red/green target independent of the full pipeline).
  - `rectify`: synthetic warp → recovered rectangle; plus a broken-contour fallback case.
  - `segmentation`: known rectified plate → **asserts the badge/strip are excluded and exactly
    7–8 glyph slots remain** (the blocker case).
  - `ocr`: rendered glyphs (deterministic) **and** a real segmented crop fixture (tests the
    shared normalization contract).
  - `validation`: format strings incl. 3- and 4-digit plates, region whitelist, confusables.
  - `video`: synthetic tracklet (incl. low-IoU growth case) → correct association + voting.
- **End-to-end accuracy harness** over `data/eval/`: reports detection recall, character
  accuracy, plate-exact accuracy, abstain rate, stratified easy/moderate/hard. These metrics
  are the results slides.
- **Precision-first invariant test:** zero wrong reads above τ on the eval set. **Gated to M3**
  (needs calibration data); M2 runs the conservative provisional threshold instead.

---

## 11. Build sequencing — one implementation plan per milestone

This spec is too large for a single plan; **each milestone is its own spec→plan→implement
cycle**, and **M2 is the minimum shippable graded core**.

- **M0** — `git init` (done), scaffold package + tests + `requirements.txt`.
- **M1** — OCR: render font glyphs + augmentation → features → `ocr.py` + KNN, unit-tested on
  rendered glyphs **and a real segmented-crop fixture** (locks the normalization contract).
  Region-code whitelist (the 34 in §2) is finalized here, not deferred.
- **M2** — Detector **cue (a)** + `rectify` (with fallback) + `segmentation` (with badge
  removal) → end-to-end on *easy* images; CSV + CLI. Acceptance = **reads easy plates** using
  a conservative provisional threshold (NOT the precision-first gate yet). Minimum shippable core.
- **M3** — add cues **(b)+(c)** + scoring/merge → moderate/hard images; **calibrate τ** on the
  eval set; the precision-first invariant test now gates.
- **M4** — video path (tracking + fallback association → best-frame → voting).
- **M5** — FastAPI upload + frontend, then WebSocket live camera (+ HTTPS dev mode).
- **M6** — generate the PPT; final accuracy harness; harden/optimize/audit pass on the core.

The cascade is built **incrementally on top of cue (a)**, so M2 is a working fallback if later
milestones run short on time.

---

## 12. Risks & open items

- **Hard images:** distant/blurred plates may detect but not read — accepted by design
  (abstain). Set expectations in the PPT "Ограничувања" slide.
- **Font fidelity:** no exact MK font may be freely available; mitigated by an EU-style font +
  real crops + augmentation (§9). Document the choice.
- **Confidence on a thin eval set:** τ tuned on ~30 images is statistically thin; use a safety
  margin and document the caveat (§5.4).
- **`getUserMedia` HTTPS on phone:** mitigated by HTTPS dev mode.
- **Eval data dependency:** M3 calibration and the accuracy numbers need the student's labeled
  images/videos; M0–M2 proceed without them.
- **Region-code list:** RESOLVED — the 34 codes are inlined in §2 and finalized at M1.
