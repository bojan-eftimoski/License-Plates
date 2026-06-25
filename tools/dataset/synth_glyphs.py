"""Generate synthetic character templates for the classical KNN OCR backbone.

Renders the 32 Macedonian plate classes (0-9, A-Z minus Q/W/X/Y) from the OFL
DIN 1451 Mittelschrift font, applies plate-like augmentation (rotation, perspective,
stroke thickness, blur, noise), and writes normalized 32x32 binary glyph crops to
data/templates/<CLASS>/. Foreground = white (255) on black (0). This mirrors the
glyph-normalization contract used by the alpr core (deskew/tight-crop/aspect-pad).
"""
import argparse
import csv
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = [c for c in string.ascii_uppercase if c not in "QWXY"]   # 22 letters
CLASSES = list("0123456789") + LETTERS                              # 32 classes
NORM = 32          # output cell size
MARGIN = 3         # padding inside the cell


def render_glyph(ch: str, font: ImageFont.FreeTypeFont, canvas: int = 256) -> np.ndarray:
    """Black glyph on white, roughly centered. Returns grayscale uint8."""
    img = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox((0, 0), ch, font=font)
    x = (canvas - (r - l)) // 2 - l
    y = (canvas - (b - t)) // 2 - t
    d.text((x, y), ch, fill=0, font=font)
    return np.array(img)


def augment(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    """Plate-like degradations on a black-on-white glyph (white border preserved)."""
    h, w = gray.shape
    # rotation + scale
    ang = rng.uniform(-9, 9)
    scale = rng.uniform(0.85, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    gray = cv2.warpAffine(gray, M, (w, h), borderValue=255)
    # mild perspective jitter
    if rng.random() < 0.6:
        d = w * 0.08
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
        gray = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(src, dst), (w, h),
                                   borderValue=255)
    # stroke thickness (erode grows the dark glyph, dilate thins it)
    k = np.ones((3, 3), np.uint8)
    r = rng.random()
    if r < 0.33:
        gray = cv2.erode(gray, k, iterations=1)
    elif r > 0.66:
        gray = cv2.dilate(gray, k, iterations=1)
    # blur
    if rng.random() < 0.7:
        ks = rng.choice([3, 3, 5])
        gray = cv2.GaussianBlur(gray, (ks, ks), 0)
    # gaussian noise
    if rng.random() < 0.6:
        noise = rng.uniform(4, 18)
        gray = np.clip(gray.astype(np.float32) + np.random.normal(0, noise, gray.shape), 0, 255).astype(np.uint8)
    return gray


def normalize(gray: np.ndarray) -> np.ndarray | None:
    """Otsu-binarize -> tight crop -> aspect-preserving pad into NORM x NORM. White glyph on black."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ys, xs = np.where(th > 0)
    if len(xs) == 0:
        return None
    crop = th[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    inner = NORM - 2 * MARGIN
    s = inner / max(ch, cw)
    rh, rw = max(1, int(round(ch * s))), max(1, int(round(cw * s)))
    crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
    out = np.zeros((NORM, NORM), np.uint8)
    y0, x0 = (NORM - rh) // 2, (NORM - rw) // 2
    out[y0:y0 + rh, x0:x0 + rw] = crop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="data/raw/fonts/din1451alt.ttf")
    ap.add_argument("--out", default="data/templates")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    font = ImageFont.truetype(args.font, 180)
    out = Path(args.out)

    rows = []
    for cls in CLASSES:
        cdir = out / cls
        cdir.mkdir(parents=True, exist_ok=True)
        base = render_glyph(cls, font)
        # one clean canonical template + augmented variants
        for i in range(args.per_class):
            g = base if i == 0 else augment(base, rng)
            norm = normalize(g)
            if norm is None:
                continue
            name = f"synth_{i:03d}.png"
            cv2.imwrite(str(cdir / name), norm)
            rows.append((f"{cls}/{name}", cls, "synth"))

    meta = out / "_meta.csv"
    write_header = not meta.exists()
    with meta.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["file", "class", "source"])
        w.writerows(rows)

    print(f"Generated {len(rows)} synthetic glyphs across {len(CLASSES)} classes "
          f"({args.per_class}/class) -> {out}")


if __name__ == "__main__":
    main()
