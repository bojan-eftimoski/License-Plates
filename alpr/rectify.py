"""Rectify a detected plate candidate to a canonical fronto-parallel rectangle (cv2 + numpy).

Crop the candidate (+margin), then:
  1. DESKEW by the projection-profile method -- rotate to the angle that makes the dark-ink
     row-projection sharpest (text rows crisp => deskewed). Robust to the mild rotations that
     defeat 4-corner contour finding;
  2. crop to the PLATE BODY (largest bright region) to drop the dark border frame;
  3. resize to canonical CANON_W x CANON_H.
`ok` is False only for degenerate/empty crops. cv2 + numpy only.
"""
import cv2
import numpy as np

from .types import Candidate, RectifiedPlate

CANON_H = 96
ASPECT = 4.727
CANON_W = round(CANON_H * ASPECT)        # 454
MARGIN = 0.06
MAX_SKEW = 14.0                          # deg searched each way


def _expand_bbox(bbox, W, H):
    x, y, w, h = bbox
    mx, my = int(round(w * MARGIN)), int(round(h * MARGIN))
    return (max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my))


def _ink(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def _deskew_angle(gray) -> float:
    """Angle (deg) maximizing the variance of the ink row-projection (sharpest text rows)."""
    th = _ink(gray)
    h, w = th.shape
    best_a, best_score = 0.0, -1.0
    for a in np.arange(-MAX_SKEW, MAX_SKEW + 0.1, 2.0):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(a), 1.0)
        rot = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_NEAREST)
        rows = rot.sum(axis=1).astype(np.float64)
        score = float(rows.var())
        if score > best_score:
            best_score, best_a = score, float(a)
    return best_a


def _plate_body_bbox(gray):
    """Bbox of the largest bright region (the white plate body), holes filled -> drops border."""
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    H, W = gray.shape
    # Must look like a plate body, not a car-body / background blob: spans most of the crop
    # width and is plate-shaped. Otherwise skip the crop and let segmentation handle it.
    if w < 0.55 * W or not (2.2 <= w / max(1, h) <= 7.0):
        return None
    return x, y, w, h


def rectify(image_bgr: np.ndarray, candidate: Candidate) -> RectifiedPlate:
    H, W = image_bgr.shape[:2]
    x0, y0, x1, y1 = _expand_bbox(candidate.bbox, W, H)
    if x1 - x0 < 16 or y1 - y0 < 10:
        warp = np.zeros((CANON_H, CANON_W, 3), np.uint8)
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        return RectifiedPlate(warp=warp, quad=quad, ok=False)

    crop = image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 1. deskew
    angle = _deskew_angle(gray)
    ch, cw = gray.shape
    M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), angle, 1.0)
    crop = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 2. crop to plate body (drop dark border)
    bb = _plate_body_bbox(gray)
    if bb is not None:
        bx, by, bw_, bh_ = bb
        pad = 2
        crop = crop[max(0, by - pad):by + bh_ + pad, max(0, bx - pad):bx + bw_ + pad]

    if crop.size == 0:
        crop = image_bgr[y0:y1, x0:x1]

    # 3. canonical resize
    warp = cv2.resize(crop, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    crop_aspect = (x1 - x0) / float(max(1, y1 - y0))
    ok = (float(cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY).std()) >= 5.0
          and (bb is not None or crop_aspect >= 2.0))   # reject non-plate-shaped inputs
    return RectifiedPlate(warp=warp, quad=quad, ok=ok)
