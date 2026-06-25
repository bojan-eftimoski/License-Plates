"""Export the trained glyph templates (synthetic + real) to data/templates.npz.

This is the committed deployment artifact the FastAPI backend loads on Hugging Face Spaces,
where the per-class PNG directory is absent (and the real glyphs can't be regenerated without
the gitignored raw data). It IS the "exported KNN templates" — one source of truth for the
graded model and any future client-side port. Regenerate after changing data/templates/.
"""
import glob
import os

import cv2
import numpy as np

SRC = "data/templates"
OUT = "data/templates.npz"


def main():
    imgs, labels = [], []
    for cls in sorted(os.listdir(SRC)):
        d = os.path.join(SRC, cls)
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*.png")):
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(im)
                labels.append(cls)
    np.savez_compressed(OUT, images=np.array(imgs, np.uint8), labels=np.array(labels))
    size_mb = os.path.getsize(OUT) / 1e6
    synth = sum(1 for f in glob.glob(f"{SRC}/*/synth_*.png"))
    real = sum(1 for f in glob.glob(f"{SRC}/*/real_*.png"))
    print(f"exported {len(imgs)} glyphs ({synth} synth + {real} real) -> {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
