"""Glyph normalization + feature extraction (shared contract for templates and runtime).

A glyph is normalized to a NORM x NORM binary (white glyph on black) via tight-crop +
aspect-preserving pad, then described with HOG. The SAME normalization is used by the
synthetic template generator, the real-glyph extractor, and the runtime segmenter, so
training and inference features live in the same space.
"""
import cv2
import numpy as np

NORM = 32
MARGIN = 3
# HOG over the 32x32 cell: 9 blocks x (2x2 cells) x 9 bins = 324-d descriptor.
_HOG = cv2.HOGDescriptor((NORM, NORM), (16, 16), (8, 8), (8, 8), 9)


def normalize(mask: np.ndarray) -> np.ndarray:
    """White-glyph-on-black `mask` (any size) -> NORM x NORM uint8, centered, aspect-preserved."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((NORM, NORM), np.uint8)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    inner = NORM - 2 * MARGIN
    s = inner / max(ch, cw)
    rh, rw = max(1, round(ch * s)), max(1, round(cw * s))
    crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
    out = np.zeros((NORM, NORM), np.uint8)
    y0, x0 = (NORM - rh) // 2, (NORM - rw) // 2
    out[y0:y0 + rh, x0:x0 + rw] = crop
    return out


def extract(img32: np.ndarray) -> np.ndarray:
    """NORM x NORM glyph -> float32 HOG feature vector."""
    img = img32 if img32.dtype == np.uint8 else img32.astype(np.uint8)
    if img.shape != (NORM, NORM):
        img = cv2.resize(img, (NORM, NORM))
    return _HOG.compute(img).flatten().astype(np.float32)
