"""Semi-automatically extract labeled real character glyphs from olavsplates close-up crops.

These crops are tight, near-frontal plate images whose filename encodes the plate string
(the label). For each: drop chromatic regions (blue NMK strip + red/orange badge) via an
HSV saturation mask, Otsu-binarize, connected-component segment, order left-to-right, and
ONLY if the glyph count equals the label length do we zip glyphs->characters and save them
to data/templates/<CLASS>/ (source=real). The length check rejects bad segmentations so we
never pollute the KNN set. This is a dataset tool, separate from the production alpr core.
"""
import csv
import re
from pathlib import Path

import cv2
import numpy as np

SRC = Path("data/raw/olavsplates")
OUT = Path("data/templates")
NORM, MARGIN = 32, 3


def parse_label(name: str):
    m = re.search(r"(n?mk)_([a-z0-9-]+?)(_close)?\.jpg$", name, re.I)
    if not m:
        return None
    return re.sub(r"[^a-z0-9]", "", m.group(2), flags=re.I).upper()


def normalize(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    inner = NORM - 2 * MARGIN
    s = inner / max(ch, cw)
    rh, rw = max(1, int(round(ch * s))), max(1, int(round(cw * s)))
    crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
    out = np.zeros((NORM, NORM), np.uint8)
    y0, x0 = (NORM - rh) // 2, (NORM - rw) // 2
    out[y0:y0 + rh, x0:x0 + rw] = crop
    return out


def segment(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    H = 128
    img = cv2.resize(img_bgr, (int(w * H / h), H))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[hsv[:, :, 1] > 70] = 255          # drop chromatic strip + badge -> background
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if not (0.35 * H <= hh <= 0.95 * H):
            continue
        if ww < 0.03 * H or ww > 0.9 * H or area < 30 or ww / hh > 1.3:
            continue
        boxes.append((x, y, ww, hh, i))
    boxes.sort(key=lambda b: b[0])
    glyphs = []
    for x, y, ww, hh, i in boxes:
        mask = ((lab[y:y + hh, x:x + ww] == i).astype(np.uint8)) * 255
        g = normalize(mask)
        if g is not None:
            glyphs.append(g)
    return glyphs


def main():
    rows = []
    kept_plates = skipped = 0
    for f in sorted(SRC.glob("*_close.jpg")):
        label = parse_label(f.name)
        img = cv2.imread(str(f))
        if not label or img is None:
            continue
        glyphs = segment(img)
        if len(glyphs) != len(label):
            skipped += 1
            print(f"  [skip] {f.name}: segmented {len(glyphs)} != label {len(label)} ({label})")
            continue
        kept_plates += 1
        for ch, g in zip(label, glyphs):
            cdir = OUT / ch
            cdir.mkdir(parents=True, exist_ok=True)
            name = f"real_{label}_{ch}.png"
            cv2.imwrite(str(cdir / name), g)
            rows.append((f"{ch}/{name}", ch, "real"))

    meta = OUT / "_meta.csv"
    write_header = not meta.exists()
    with meta.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["file", "class", "source"])
        w.writerows(rows)

    classes = sorted({r[1] for r in rows})
    print(f"\nExtracted {len(rows)} real glyphs from {kept_plates} plates "
          f"({skipped} skipped on length-check) covering {len(classes)} classes: {classes}")


if __name__ == "__main__":
    main()
