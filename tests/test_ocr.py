import os
import random

import numpy as np
import pytest

from alpr.features import NORM, normalize
from alpr.ocr import CLASSES, GlyphClassifier, load_glyph_dataset

TEMPLATES = "data/templates"


@pytest.fixture(scope="module")
def dataset():
    if not os.path.isdir(TEMPLATES) or not os.listdir(TEMPLATES):
        pytest.skip("templates not generated (run tools/dataset/synth_glyphs.py)")
    return load_glyph_dataset(TEMPLATES)


def test_classes_are_the_32_mk_glyphs():
    assert len(CLASSES) == 32
    for banned in "QWXY":
        assert banned not in CLASSES


def test_all_classes_have_templates(dataset):
    _, labels = dataset
    assert sorted(set(labels)) == sorted(CLASSES)


def test_normalize_centers_and_pads_to_NORM():
    m = np.zeros((50, 40), np.uint8)
    m[10:45, 12:28] = 255
    out = normalize(m)
    assert out.shape == (NORM, NORM)
    assert out.max() == 255 and out.min() == 0


def test_knn_holdout_accuracy(dataset):
    imgs, labels = dataset
    rng = random.Random(0)
    by_cls: dict[str, list] = {}
    for im, l in zip(imgs, labels):
        by_cls.setdefault(l, []).append(im)

    tr_i, tr_l, te_i, te_l = [], [], [], []
    for l, ims in by_cls.items():
        ims = ims[:]
        rng.shuffle(ims)
        n = max(1, int(len(ims) * 0.2))
        te_i += ims[:n]; te_l += [l] * n
        tr_i += ims[n:]; tr_l += [l] * (len(ims) - n)

    clf = GlyphClassifier().fit(tr_i, tr_l)
    correct = sum(clf.classify(im)[0] == l for im, l in zip(te_i, te_l))
    acc = correct / len(te_i)
    assert acc >= 0.95, f"holdout glyph accuracy {acc:.3f} below 0.95"
