# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the workspace for a **seminar project** in the FINKI course *Дигитално процесирање на слика* (Digital Image Processing / DPNS). It is **not an application codebase** — it is two things side by side:

1. **The assignment** (`Tema_Na_Seminarska.txt`, `Informacii_Za_Seminarski.txt`, both in Macedonian): implement a demo app that takes a photo of a vehicle, **detects the Macedonian license plate, recognizes its letters/digits (OCR), and writes the result to a file**. Deliverables are the demo app plus a ≥20-slide PowerPoint (in Macedonian) describing the implemented algorithms.
2. **Course reference material** (`Materials/`): ~100 standalone OpenCV example scripts and lecture PDFs provided by the instructor. These are building blocks to draw from — **not** the project itself.

**The project app does not exist yet.** As of now the repo contains only the reference scripts, the input data, and the brief. New project code should be written at the repository root (or a new top-level folder), kept separate from `Materials/`.

## Layout

- `Materials/` — read-only reference examples, grouped by topic (color models, filters, histograms, thresholding, morphology, edge/keypoint detection & SIFT, CBIR image search, geometric transforms, barcode detection, …). Each subfolder is self-contained with its own sample images and one or more `.py` scripts.
- `Images/` — **project input photos**: `registracii*.jpg` and `IMG_*.png` are Macedonian vehicle/plate images to test the detector against.
- `Videos/` — video assets (`video1.mp4`, `video2.mp4`, `video3.mov`).
- Root `.txt` files — the assignment brief.

## Dependencies

There is no `requirements.txt`, virtualenv, or any package metadata. Scripts import `cv2`, `numpy`, and `matplotlib`. Install globally before running:

```
pip install opencv-python numpy matplotlib
```

The plate-recognition project will additionally need an OCR engine (e.g. Tesseract via `pytesseract`, or a trained classifier) — none is present yet; pick and document one.

## Running scripts

There is **no build, no lint, and no test suite**. Each `.py` file is run directly with `python`. Two important gotchas:

- **Most scripts hardcode relative image filenames** (e.g. `cv2.imread('messi5.jpg')`). You must `cd` into the script's own directory first, or the read silently returns `None` and the script crashes later:
  ```
  cd "Materials/Image-manipulations" && python example1.py
  ```
- **~10 scripts use `argparse`** instead. They carry a `# USAGE` comment block at the top documenting the exact invocation, e.g.:
  ```
  cd Materials/image_search_engine
  python index.py  --dataset images --index index.pickle
  python search.py --dataset images --index index.pickle
  ```
- Scripts open **blocking GUI windows** via `cv2.imshow(...)` + `cv2.waitKey(0)` — they hang until a key (often `q`) is pressed in the window. Some use matplotlib and force the Tk backend with `mpl_use('TkAgg')`. Neither works headlessly; expect a display.

## Reference examples most relevant to the plate-recognition project

When implementing the detector/OCR, these existing scripts are the closest starting points:

- `Materials/BarCodeDetection/detect_barcode.py` — gradient + morphology + contour region localization; the same pipeline shape applies to finding a plate's bounding box.
- `Materials/Examples (2)/Examples/otsu_thresholding.py` & `adaptive_thresholding.py` — binarization for isolating characters.
- `Materials/Examples (3)/Examples/` — binary erosion/dilation and edge detection for cleaning up character masks.
- `Materials/Examples (4)/Examples/` — contour/shape detection for segmenting individual characters.
- `Materials/AutomaticQuizGrading/quiz.py` — affine template alignment + thresholded `absdiff`, a template-matching approach to character recognition.
- `Materials/image_search_engine/` — the only **multi-file** module here: a CBIR pipeline (`pyimagesearch/` package with an `RGBHistogram` descriptor and a `Searcher` using chi-squared histogram distance, plus a pickled index). Useful as a model for descriptor + matching code structure.

## Code conventions in the existing scripts

The reference scripts follow a "PyImageSearch" tutorial style — match it only when extending `Materials/`; for the project itself, prefer clean modern Python 3:

- A `# USAGE` header comment, then `# import the necessary packages`, then heavy line-by-line inline comments.
- argparse parsed into a dict: `args = vars(ap.parse_args())`.
- Mixed indentation across files (some use tabs, e.g. `searcher.py`; others spaces) and mixed Python 2/3 idioms.
- Committed build artifacts exist under `image_search_engine/` (`*.pyc`, `__pycache__/`, `index.pickle`) — don't treat them as source.
