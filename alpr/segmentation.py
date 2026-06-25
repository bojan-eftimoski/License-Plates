"""Character segmentation on a rectified plate (cv2 + numpy only).

Pipeline: drop CHROMATIC pixels (blue NMK strip + red/orange badge) to background ->
CLAHE + adaptive threshold (robust to shadows) -> restrict to the character row band ->
connected components shaped like glyphs -> post-process toward the known 7-8 count
(drop noise, split merged). Ordered left-to-right. Expect 7-8 glyphs.
"""
import cv2
import numpy as np

from .features import normalize
from .types import GlyphCrop

PLATE_H = 96
SAT_CHROMA = 60
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _erase_chroma(gray, hsv):
    """Blank the WHOLE bounding box of each blue-strip / red-yellow-badge blob.

    Erasing only chromatic pixels leaves the badge's dark Cyrillic letters behind (they
    get mis-segmented as glyphs); filling the blob's bbox removes the badge entirely.
    Hue-specific (not just high-saturation) so a blue color-cast plate isn't wiped out.
    """
    h, w = gray.shape
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    blue = (hue >= 100) & (hue <= 135) & (sat > 90)
    red = ((hue <= 12) | (hue >= 168)) & (sat > 90)
    yellow = (hue >= 18) & (hue <= 40) & (sat > 110)
    mask = ((blue | red | yellow).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                              stats[i, cv2.CC_STAT_AREA])
        if area < 0.0025 * h * w:                    # ignore tiny colour speckle
            continue
        if bw > 0.35 * w:                            # never erase a plate-wide region
            continue
        p = 2
        gray[max(0, y - p):y + bh + p, max(0, x - p):x + bw + p] = 255
    return gray


def _binarize(warp_bgr):
    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
    gray = _CLAHE.apply(cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY))
    gray = _erase_chroma(gray, hsv)                  # remove strip + badge (whole box)
    block = max(15, (warp_bgr.shape[0] // 2) | 1)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 12)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return th


def _text_band(th):
    """Vertical [y0,y1) of the dominant character row (drops top/bottom border + dealer frame)."""
    rowsum = (th > 0).sum(axis=1).astype(np.float32)
    rowsum = cv2.GaussianBlur(rowsum.reshape(-1, 1), (1, 7), 0).flatten()
    if rowsum.max() < 1:
        return 0, th.shape[0]
    rows = np.where(rowsum > rowsum.max() * 0.30)[0]
    return int(rows.min()), int(rows.max()) + 1


def _components(th, h):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, bh, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                             stats[i, cv2.CC_STAT_AREA])
        if bh < 0.45 * h or bh > 0.98 * h:          # glyph height ~ band height
            continue
        if w > 0.8 * h or area < 0.10 * w * bh:      # reject merged-wide / sparse frame
            continue
        out.append([x, y, w, bh, i])
    return out, lab


def _split_wide(boxes, lab, band_h):
    """Split components much wider than the median into char-count pieces via projection valleys."""
    if not boxes:
        return boxes
    widths = sorted(b[2] for b in boxes)
    med = widths[len(widths) // 2]
    result = []
    for x, y, w, bh, i in boxes:
        k = int(round(w / med)) if med > 0 else 1
        if k <= 1 or w < 1.6 * med:
            result.append((x, y, w, bh, i))
            continue
        sub = (lab[:, x:x + w] == i).astype(np.uint8)
        col = sub.sum(axis=0).astype(np.float32)
        col = cv2.GaussianBlur(col.reshape(1, -1), (5, 1), 0).flatten()
        # cut at the k-1 deepest interior valleys, keep cuts apart
        order = np.argsort(col)
        cuts, mind = [], int(0.5 * med)
        for c in order:
            if 0.15 * w < c < 0.85 * w and all(abs(c - cc) > mind for cc in cuts):
                cuts.append(int(c))
            if len(cuts) >= k - 1:
                break
        edges = [0] + sorted(cuts) + [w]
        for a, b in zip(edges, edges[1:]):
            if b - a > 0.25 * med:
                result.append((x + a, y, b - a, bh, i))
    return result


def segment(warp_bgr: np.ndarray) -> list[GlyphCrop]:
    if warp_bgr is None or warp_bgr.size == 0:
        return []
    h, w = warp_bgr.shape[:2]
    if h != PLATE_H:
        warp_bgr = cv2.resize(warp_bgr, (max(1, int(round(w * PLATE_H / h))), PLATE_H))

    th = _binarize(warp_bgr)
    y0, y1 = _text_band(th)
    band = th[y0:y1]
    band_h = max(1, y1 - y0)

    boxes, lab = _components(band, band_h)
    boxes = _split_wide(boxes, lab, band_h)
    # drop slivers (noise / leftover frame bits) relative to the median glyph width
    if boxes:
        widths = sorted(b[2] for b in boxes)
        med = widths[len(widths) // 2]
        boxes = [b for b in boxes if b[2] > 0.28 * med]
    boxes.sort(key=lambda b: b[0])

    glyphs = []
    for idx, (x, y, bw, bh, i) in enumerate(boxes):
        sub = band[y:y + bh, x:x + bw]
        glyphs.append(GlyphCrop(norm=normalize(sub), bbox=(x, y0 + y, bw, bh), index=idx))
    return glyphs
