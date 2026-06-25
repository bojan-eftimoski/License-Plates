"""Extract labeled real character glyphs from olavsplates close-ups via the PRODUCTION pipeline.

For each close-up (filename = ground-truth plate string) run the real alpr rectify + segment;
where the glyph count matches the label length, zip glyphs->characters and save each (plus a
few augmented variants, to give the scarce real samples weight) into data/templates/<CLASS>/.
This mixes real-glyph appearance into the synthetic-trained KNN to close the synth->real gap.

TRAINING-SAFE: olavsplates plates are NOT in the held-out eval (Platesmania + registracii1),
so this introduces no leakage.
"""
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np

from alpr.rectify import rectify
from alpr.segmentation import segment
from alpr.types import Candidate

SRC = Path("data/raw/olavsplates")
OUT = Path("data/templates")
AUG = 4                      # augmented variants per real glyph


def _label(name: str):
    m = re.search(r"(n?mk)_([a-z0-9-]+?)(_close)?\.jpg$", name, re.I)
    return re.sub(r"[^a-z0-9]", "", m.group(2), flags=re.I).upper() if m else None


def _augment(g: np.ndarray, rng: random.Random) -> np.ndarray:
    out = g.copy()
    M = cv2.getRotationMatrix2D((16, 16), rng.uniform(-7, 7), 1.0)
    out = cv2.warpAffine(out, M, (32, 32))
    r = rng.random()
    if r < 0.30:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    elif r > 0.70:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def main():
    rng = random.Random(7)
    for old in OUT.glob("*/real_*.png"):
        old.unlink()

    kept = 0
    real_rows = []
    for f in sorted(SRC.glob("*_close.jpg")):
        lab = _label(f.name)
        im = cv2.imread(str(f))
        if not lab or im is None:
            continue
        h, w = im.shape[:2]
        rp = rectify(im, Candidate(bbox=(0, 0, w, h), score=1.0, cue="extract"))
        glyphs = segment(rp.warp)
        if len(glyphs) != len(lab):
            continue
        kept += 1
        for i, (ch, gl) in enumerate(zip(lab, glyphs)):
            cdir = OUT / ch
            cdir.mkdir(parents=True, exist_ok=True)
            for n, v in enumerate([gl.norm] + [_augment(gl.norm, rng) for _ in range(AUG)]):
                name = f"real_{lab}_{i}_{n}.png"
                cv2.imwrite(str(cdir / name), v)
                real_rows.append((f"{ch}/{name}", ch, "real"))

    # regenerate _meta.csv (synthetic + real) for provenance
    meta_rows = []
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for png in sorted(d.glob("*.png")):
            src = "real" if png.name.startswith("real_") else "synth"
            meta_rows.append((f"{d.name}/{png.name}", d.name, src))
    with (OUT / "_meta.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "source"])
        w.writerows(meta_rows)

    classes = sorted({r[1] for r in real_rows})
    print(f"extracted {kept} plates -> {len(real_rows)} real glyph images "
          f"(base + {AUG}x aug) covering {len(classes)} classes: {classes}")


if __name__ == "__main__":
    main()
