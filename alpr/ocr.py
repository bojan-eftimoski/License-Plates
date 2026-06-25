"""Classical character OCR: HOG features + cv2.ml.KNearest with ratio-test confidence.

Trains on the glyph templates in data/templates/ (synthetic + real). The trained samples
can be exported (features + labels) for the client-side OpenCV.js demo so it shares one
source of truth with the graded Python model. No deep learning, no Tesseract, no sklearn.
"""
import glob
import os

import cv2
import numpy as np

from .features import extract

LETTERS = list("ABCDEFGHIJKLMNOPRSTUVZ")        # A-Z minus Q, W, X, Y (22)
CLASSES = list("0123456789") + LETTERS          # 32 Macedonian plate glyph classes


def load_glyph_dataset(templates_dir: str = "data/templates"):
    """Return (imgs, labels). Prefers the <CLASS>/*.png dir (dev); falls back to the committed
    `<templates_dir>.npz` artifact (deployment, e.g. HF Spaces, where the PNG dir is absent)."""
    imgs, labels = [], []
    if os.path.isdir(templates_dir):
        for cls in sorted(os.listdir(templates_dir)):
            d = os.path.join(templates_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "*.png")):
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    imgs.append(im)
                    labels.append(cls)
    if imgs:
        return imgs, labels
    npz = templates_dir + ".npz"
    if os.path.isfile(npz):
        d = np.load(npz)
        return [im for im in d["images"]], [str(s) for s in d["labels"]]
    return imgs, labels


class GlyphClassifier:
    """KNN over HOG glyph features. classify() returns (char, confidence in [0,1])."""

    def __init__(self, k: int = 5):
        self.k = k
        self.knn = None
        self.classes: list[str] = []

    def fit(self, imgs, labels):
        self.classes = sorted(set(labels))
        idx = {c: i for i, c in enumerate(self.classes)}
        feats = np.array([extract(im) for im in imgs], np.float32)
        resp = np.array([idx[l] for l in labels], np.int32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(feats, cv2.ml.ROW_SAMPLE, resp)
        return self

    def classify(self, img32):
        f = extract(img32).reshape(1, -1)
        _, results, neighbours, dist = self.knn.findNearest(f, self.k)
        pred = int(results[0, 0])
        neigh = neighbours.flatten().astype(int)
        d = dist.flatten()
        # ratio-test confidence: nearest same-class vs nearest other-class distance
        same = d[neigh == pred]
        other = d[neigh != pred]
        d1 = float(same.min()) if same.size else 0.0
        d2 = float(other.min()) if other.size else (d1 * 1e3 + 1.0)
        conf = float(np.clip(1.0 - d1 / (d2 + 1e-9), 0.0, 1.0))
        return self.classes[pred], conf
