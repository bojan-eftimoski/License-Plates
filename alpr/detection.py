"""Classical license-plate detection: hybrid candidate cascade (cv2 + numpy only).

Two independent cues propose plate-like boxes, which are then merged (NMS) and scored:

  (a) MORPH/EDGE -- plates are a dense horizontal band of dark glyphs on a light
      background. We blackhat (dark-on-light glyphs), take the Sobel-x gradient,
      morphologically close the strokes into a solid bar, Otsu-threshold and keep
      contour bounding boxes whose aspect/area look like a plate.

  (b) COLOR -- the blue NMK strip on the left edge is a strong, saturated blue. We
      build an HSV mask for it, take each blue blob as an anchor and expand to the
      right (the plate body) to propose a box. This recovers plates the morph cue
      smears into the bumper.

Boxes are merged with non-max suppression and scored by aspect closeness to the
canonical 4.73, fill ratio (foreground coverage of a plate-tuned mask) and edge
density. detect_candidates() returns Candidates sorted best-first.

The MK plate aspect is ~4.73 INCLUDING the blue strip. The morph cue tends to fire
on the glyph block only (no strip), whose aspect is closer to ~3.5-4; scoring tolerates
the whole 3.0-6.5 band and we widen morph boxes slightly to the left to recover the strip.
"""
import cv2
import numpy as np

from .types import Candidate

# --- plate geometry ---------------------------------------------------------
PLATE_ASPECT = 4.73          # canonical w/h including the blue NMK strip
ASPECT_MIN = 2.2             # accept glyph-block-only boxes (no strip) up to full plate
ASPECT_MAX = 7.5
MIN_PLATE_W = 40             # px, at working scale -- below this OCR has no signal
MAX_PLATE_W_FRAC = 0.97      # a plate never spans (almost) the whole frame width
MIN_PLATE_H = 11

# working scale: detection runs at a fixed width so morphology kernels are stable
# across the 460px platesmania crops and larger uploads alike.
WORK_W = 720
# a real plate occupies a non-trivial slice of the frame; this is the width fraction
# at which the size prior saturates to ~1 (plates smaller than this are penalized).
SIZE_PRIOR_W_FRAC = 0.16


def _to_work_scale(image_bgr):
    """Resize so width == WORK_W (only downscale-or-up to a sane band). Returns (img, scale)."""
    h, w = image_bgr.shape[:2]
    scale = WORK_W / float(w)
    # don't blow tiny images up enormously or shrink huge ones to mush
    scale = float(np.clip(scale, 0.25, 3.0))
    if abs(scale - 1.0) < 1e-3:
        return image_bgr, 1.0
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (nw, nh), interpolation=interp), scale


def _aspect_score(w, h):
    """1.0 at exactly 4.73, decaying smoothly. Generous so glyph-only boxes survive."""
    if h <= 0:
        return 0.0
    ar = w / float(h)
    # use log-ratio so 2x and 0.5x off are penalized symmetrically
    r = np.log(ar / PLATE_ASPECT)
    return float(np.exp(-(r * r) / (2 * 0.35 * 0.35)))


def _morph_cue(gray):
    """Blackhat + Sobel-x + horizontal close + Otsu -> plate-like bounding boxes.

    Closing is deliberately HORIZONTAL-dominant (merge glyphs of one line into a bar)
    with minimal vertical reach, then a vertical open snaps the bar away from the dark
    car body / shadow below so the box height stays glyph-tight. Multi-kernel covers a
    range of plate scales.
    """
    boxes = []
    H, W = gray.shape
    gray_eq = cv2.equalizeHist(gray)
    # blackhat: dark glyphs on light plate -> bright glyph response
    for kw, kh in ((11, 5), (17, 5), (25, 7)):
        bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, bh_kernel)

        # horizontal gradient picks up the vertical strokes of glyphs
        gx = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.absolute(gx)
        mn, mx = gx.min(), gx.max()
        if mx - mn < 1e-6:
            continue
        gx = ((gx - mn) / (mx - mn) * 255).astype(np.uint8)

        gx = cv2.GaussianBlur(gx, (5, 5), 0)
        _, th = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # close horizontally only -> one bar per plate line, no vertical bleed
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw + 6, 1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k, iterations=1)
        # short vertical close to fuse the two stroke-edges of each glyph row
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        # vertical open snaps the bar off the car-body/shadow it may touch below
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if not _plausible(w, h, W):
                continue
            # widen slightly to the left to recover the blue strip the gradient missed
            pad = int(w * 0.18)
            nx = max(0, x - pad)
            nw = min(W - nx, w + pad)
            boxes.append((nx, y, nw, h, "morph"))
    return boxes


def _color_cue(image_bgr, gray):
    """HSV mask for the saturated blue NMK strip; expand right to propose a plate.

    The strip alone is ambiguous (logos, signs, parking lines are also blue), so each
    blue blob is only accepted as a plate anchor when the region immediately to its
    right is a BRIGHT (white plate body) textured band -- this rejects sky/sign/line blue.
    """
    boxes = []
    H, W = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # broad blue band for white-balance robustness; require real saturation+value so
    # pale sky / shadow blue doesn't fire.
    lower = np.array([100, 80, 60], np.uint8)
    upper = np.array([134, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # the strip is a tall-ish thin vertical band: taller than wide, not huge
        if h < MIN_PLATE_H or h > 0.55 * H:
            continue
        ar = w / float(h)
        if ar > 1.4:            # too wide to be the strip itself
            continue
        if w < 2:
            continue
        # the full plate is ~ aspect*h wide; strip occupies the leftmost ~h-wide slot.
        plate_w = int(PLATE_ASPECT * h)
        px = max(0, x - int(0.1 * h))
        pw = min(W - px, plate_w)
        py = max(0, y - int(0.15 * h))
        ph = min(H - py, int(h * 1.30))
        if not _plausible(pw, ph, W):
            continue
        # verify a bright plate body just right of the strip
        bx0 = min(W - 1, x + w + 1)
        bx1 = min(W, x + w + int(2.2 * h))
        body = gray[py:py + ph, bx0:bx1]
        if body.size == 0 or float(body.mean()) < 95:
            continue
        boxes.append((px, py, pw, ph, "color"))
    return boxes


def _plausible(w, h, frame_w):
    """Cheap geometric gate shared by both cues."""
    if w < MIN_PLATE_W or h < MIN_PLATE_H:
        return False
    if w > MAX_PLATE_W_FRAC * frame_w:
        return False
    ar = w / float(h)
    return ASPECT_MIN <= ar <= ASPECT_MAX


def _edge_density(gray, box):
    """Fraction of strong-gradient pixels inside the box (plates are texture-dense)."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    edges = np.abs(gx) > 40
    return float(edges.mean())


def _fill_ratio(gray, box):
    """Coverage of dark-glyph-like pixels after a plate-local Otsu (texture sanity)."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    frac = float((th > 0).mean())
    # a real plate's glyph ink covers ~15-45%; map that band to ~1, penalize extremes
    return float(np.clip(1.0 - abs(frac - 0.30) / 0.30, 0.0, 1.0))


def _size_prior(w, frame_w):
    """Smoothly favor boxes occupying a real-plate slice of the frame; small => penalized."""
    frac = w / float(frame_w)
    return float(np.clip(frac / SIZE_PRIOR_W_FRAC, 0.0, 1.0))


def _brightness(gray, box):
    """Mean intensity inside the box (white plates are bright); mapped to [0,1]."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(np.clip((roi.mean() - 60) / 120.0, 0.0, 1.0))


def _glyph_likeness(gray, box):
    """(glyph_score, fg_bg_contrast) for the box.

    glyph_score counts character-shaped dark components inside the box; ~5-9 => plate-like.
    A MK plate is a row of regularly-sized dark glyphs ~50-90% of the plate height on a
    bright ground. This is the most plate-SPECIFIC verifier and the main thing separating
    a real plate from a window/grille/trim box that merely has plate-ish aspect+texture.

    contrast is mean(background) - mean(foreground); real black-on-white plate ink is
    high-contrast, low-contrast watermark text / gray trim scores lower (soft signal).
    """
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0 or h < 6:
        return 0.0, 0.0
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    fg = roi[th > 0]
    bg = roi[th == 0]
    contrast = (float(bg.mean()) - float(fg.mean())) if fg.size and bg.size else 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    good = 0
    for i in range(1, n):
        cw, ch, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if ch < 0.35 * h or ch > 1.05 * h:        # glyph height ~ plate height
            continue
        ar = cw / float(ch) if ch else 0
        if ar > 1.3 or ar < 0.08:                 # glyphs are taller than wide-ish
            continue
        if area < 0.06 * cw * ch:                 # too sparse -> noise
            continue
        good += 1
    # plates carry 7-8 glyphs; tolerate partial detection. Peak at >=5, decay past 12.
    if good <= 1:
        gscore = 0.0
    elif good >= 5:
        gscore = float(np.clip(1.0 - max(0, good - 9) * 0.08, 0.4, 1.0))
    else:
        gscore = good / 5.0 * 0.85
    return gscore, contrast


def _score(gray, box):
    x, y, w, h = box
    frame_w = gray.shape[1]
    a = _aspect_score(w, h)
    e = _edge_density(gray, box)
    f = _fill_ratio(gray, box)
    sz = _size_prior(w, frame_w)
    br = _brightness(gray, box)
    gl, contrast = _glyph_likeness(gray, box)
    # high-contrast black-on-white ink saturates ~120; low-contrast watermark/trim scores less
    ct = float(np.clip(contrast / 120.0, 0.0, 1.0))
    # edge density saturates fast; clamp so a busy background box can't dominate aspect
    e_s = float(np.clip(e / 0.35, 0.0, 1.0))
    base = 0.20 * a + 0.10 * e_s + 0.07 * f + 0.10 * br + 0.40 * gl + 0.13 * ct
    # size prior is multiplicative-ish: a perfect-aspect speck still loses to a real plate
    return base * (0.45 + 0.55 * sz)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def _nms(scored, iou_thresh=0.3):
    """Greedy NMS over (box, score, cue), best-first."""
    scored = sorted(scored, key=lambda s: s[1], reverse=True)
    kept = []
    for box, sc, cue in scored:
        if all(_iou(box, k[0]) < iou_thresh for k in kept):
            kept.append((box, sc, cue))
    return kept


def detect_candidates(image_bgr: np.ndarray) -> list[Candidate]:
    """Detect plate-like regions in a BGR image, returned as Candidates best-first.

    Runs the morph/edge and blue-strip color cues at a fixed working scale, merges
    overlaps with NMS, scores by aspect/edge/fill and maps boxes back to source pixels.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    work, scale = _to_work_scale(image_bgr)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    raw = []
    raw += _morph_cue(gray)
    raw += _color_cue(work, gray)

    if not raw:
        return []

    scored = []
    for x, y, w, h, cue in raw:
        sc = _score(gray, (x, y, w, h))
        # color-cue boxes get a small prior bump: a saturated blue strip is a strong,
        # specific MK-plate signal that pure texture boxes lack.
        if cue == "color":
            sc += 0.08
        scored.append(((x, y, w, h), sc, cue))

    kept = _nms(scored, iou_thresh=0.3)

    out = []
    inv = 1.0 / scale
    for (x, y, w, h), sc, cue in kept:
        bx = (int(round(x * inv)), int(round(y * inv)),
              int(round(w * inv)), int(round(h * inv)))
        out.append(Candidate(bbox=bx, score=float(sc), cue=cue, quad=None))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
