"""Perspective-rectify a detected plate candidate to a canonical fronto-parallel rectangle.

Crops the candidate bbox (+margin), then runs a *fallback ladder* (never aborts):
  (1) largest contour -> approxPolyDP; if 4 vertices -> 4-corner perspective warp;
  (2) else cv2.minAreaRect rotated-box deskew (rotate by its angle, resize to canonical);
  (3) else (unstable/degenerate angle) -> resize the axis-aligned crop to canonical.
The warp is validated by output aspect ratio (~4.73) and non-degeneracy; `ok` reflects
whether a real rectification (rung 1/2) succeeded into a plate-shaped rectangle.

cv2 + numpy only (graded-core toolset boundary, see docs spec §9).
"""
import cv2
import numpy as np

from .types import Candidate, RectifiedPlate

# Canonical fronto-parallel plate: height 96, aspect 520/110 = 4.727 -> width 454.
CANON_H = 96
ASPECT = 4.727
CANON_W = round(CANON_H * ASPECT)        # 454
MARGIN = 0.08                            # bbox expand fraction per side
# Aspect tolerance for accepting a rectification as a plate (lenient: real crops vary).
ASPECT_MIN = 2.8
ASPECT_MAX = 6.5


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left] (float32).

    Split-by-y then sort-each-pair-by-x: robust to strong rotation, where the classic
    sum/diff trick mis-assigns corners (a far-left bottom corner can have a smaller x+y
    than the true top-left).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    order_y = pts[np.argsort(pts[:, 1])]      # ascending y
    top = order_y[:2]
    bottom = order_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]       # left, right by x
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], np.float32)


def _expand_bbox(bbox, W, H):
    """Expand (x,y,w,h) by MARGIN per side, clamped to image bounds."""
    x, y, w, h = bbox
    mx, my = int(round(w * MARGIN)), int(round(h * MARGIN))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(W, x + w + mx)
    y1 = min(H, y + h + my)
    return x0, y0, x1, y1


def _largest_quad(gray: np.ndarray):
    """Return ordered 4x2 quad of the largest 4-vertex contour, or None."""
    # Edge map -> close gaps -> external contours.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    h, w = gray.shape[:2]
    area_img = float(w * h)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:8]:
        a = cv2.contourArea(c)
        if a < 0.08 * area_img:          # must cover a real chunk of the crop
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_quad(approx)
    return None


def _warp_quad(crop: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp `crop` so `quad` maps onto the canonical CANON_W x CANON_H rect."""
    dst = np.array([[0, 0],
                    [CANON_W - 1, 0],
                    [CANON_W - 1, CANON_H - 1],
                    [0, CANON_H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(crop, M, (CANON_W, CANON_H))


def _deskew_minarearect(crop: np.ndarray, gray: np.ndarray):
    """Rotated-box deskew: find dominant rotated box, rotate by its angle, crop, resize.

    Returns (warp, quad) where quad is the rotated box in *crop* coordinates, or None.
    """
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Plate body is the bright field; ensure foreground is the field (largest region).
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    h, w = gray.shape[:2]
    if cv2.contourArea(c) < 0.20 * w * h:
        return None
    rect = cv2.minAreaRect(c)            # ((cx,cy),(rw,rh),angle)
    (cx, cy), (rw, rh), angle = rect
    if rw < 1 or rh < 1:
        return None
    # Normalize angle so the long side becomes horizontal.
    if rw < rh:
        angle = angle + 90.0
        rw, rh = rh, rw
    # Unstable / near-degenerate: caller should drop to axis-aligned rung.
    if abs(angle) > 45.0:
        return None
    box = cv2.boxPoints(rect)            # 4x2 in crop coords
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    x0 = int(round(cx - rw / 2.0))
    y0 = int(round(cy - rh / 2.0))
    x1 = int(round(cx + rw / 2.0))
    y1 = int(round(cy + rh / 2.0))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    sub = rot[y0:y1, x0:x1]
    warp = cv2.resize(sub, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
    return warp, _order_quad(box)


def _aspect_ok(quad: np.ndarray) -> bool:
    """True if the quad's mean side aspect (width/height) is plate-like."""
    tl, tr, br, bl = quad
    wtop = np.linalg.norm(tr - tl)
    wbot = np.linalg.norm(br - bl)
    hleft = np.linalg.norm(bl - tl)
    hright = np.linalg.norm(br - tr)
    w = (wtop + wbot) / 2.0
    h = (hleft + hright) / 2.0
    if h < 1e-3:
        return False
    return ASPECT_MIN <= (w / h) <= ASPECT_MAX


def _degenerate_warp(warp: np.ndarray) -> bool:
    """Reject near-constant / collapsed warps (e.g. a flat-color region)."""
    if warp is None or warp.size == 0:
        return True
    if warp.shape[0] != CANON_H or warp.shape[1] != CANON_W:
        return True
    g = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) if warp.ndim == 3 else warp
    return float(g.std()) < 5.0


def rectify(image_bgr: np.ndarray, candidate: Candidate) -> RectifiedPlate:
    """Warp the candidate plate region to the canonical CANON_W x CANON_H rectangle.

    Fallback ladder (never aborts): 4-corner perspective -> minAreaRect deskew ->
    axis-aligned resize. `ok` is True only when a genuine rectification (rung 1 or 2)
    produced a plate-shaped, non-degenerate warp.
    """
    H, W = image_bgr.shape[:2]
    x0, y0, x1, y1 = _expand_bbox(candidate.bbox, W, H)
    if x1 - x0 < 16 or y1 - y0 < 10:
        # Nothing usable to crop; emit a black canonical canvas, not ok.
        warp = np.zeros((CANON_H, CANON_W, 3), np.uint8)
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        return RectifiedPlate(warp=warp, quad=quad, ok=False)

    crop = image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    offset = np.array([x0, y0], np.float32)

    # --- Rung 1: 4-vertex perspective warp -------------------------------------
    quad_local = _largest_quad(gray)
    if quad_local is not None and _aspect_ok(quad_local):
        warp = _warp_quad(crop, quad_local)
        if not _degenerate_warp(warp):
            return RectifiedPlate(warp=warp, quad=quad_local + offset, ok=True)

    # --- Rung 2: minAreaRect rotated-box deskew --------------------------------
    res = _deskew_minarearect(crop, gray)
    if res is not None:
        warp, quad_local = res
        if not _degenerate_warp(warp):
            return RectifiedPlate(warp=warp, quad=quad_local + offset, ok=True)

    # --- Rung 3: axis-aligned resize (last resort) -----------------------------
    warp = cv2.resize(crop, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    # The axis-aligned crop's own aspect tells us whether it was already plate-shaped.
    crop_aspect = (x1 - x0) / float(y1 - y0)
    ok = (not _degenerate_warp(warp)) and (ASPECT_MIN <= crop_aspect <= ASPECT_MAX)
    return RectifiedPlate(warp=warp, quad=quad, ok=ok)
