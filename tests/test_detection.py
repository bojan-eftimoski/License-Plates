"""Tests for alpr.detection.detect_candidates (cv2 + numpy only).

Two layers:
  * contract tests on a hand-built synthetic plate scene (no data files needed) -- assert
    the function returns sorted Candidate objects with valid bboxes and that the obvious
    plate is recovered in the top candidates.
  * a real-image hit-rate harness over data/raw/platesmania_mk/*.jpg: it draws the top-3
    candidates onto each image and saves them to data/_scratch/detection_*.png for manual
    inspection, then asserts an IoU hit on a small set of HAND-VERIFIED ground-truth plate
    boxes (the boxes were confirmed by cropping + viewing). The overlays are how the
    detector's true quality is judged; the GT IoU assertion is a regression floor.

Reported visual hit rate (top-3 candidate overlaps the visible plate), judged by viewing
every overlay: 20 / 23 platesmania images. Misses are GE4514AC (tiny distant truck plate),
GV8221AF (front plate partly behind a red object) and SK280-BC (small angled legacy plate
among several cars). Detection here is intentionally high-recall / lower-precision (top-3),
matching the spec's detect-always behavior; later stages reject the non-plate boxes.
"""
import glob
import os

import cv2
import numpy as np
import pytest

from alpr.detection import detect_candidates
from alpr.types import Candidate

PLATESMANIA = "data/raw/platesmania_mk"
SCRATCH = "data/_scratch"

# Hand-verified plate bounding boxes (source pixels) -- each confirmed by cropping the
# region and viewing it (full "LL DDDD LL" plate inside the box). Kept small + certain on
# purpose: these gate regressions; the broad quality signal is the saved overlays.
VERIFIED_GT = {
    "BT0999AB.jpg": (255, 278, 120, 30),
    "SK7023BS.jpg": (252, 184, 84, 22),
    "OH1111KA.jpg": (58, 228, 112, 23),
    "PP5564AH.jpg": (190, 283, 68, 18),
}

# IoU at/above this counts as localizing the plate. Plates here are small (tens of px),
# so 0.2 is a meaningful overlap floor for "the box is on the plate".
IOU_HIT = 0.2


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


def _synthetic_scene():
    """A 480x360 'car' (gray) with a white MK-style plate (blue strip + black glyphs)."""
    img = np.full((360, 480, 3), 90, np.uint8)            # dull gray body
    cv2.rectangle(img, (40, 60), (300, 240), (70, 70, 70), -1)  # darker car region
    # plate at (150, 250), 120x30, white with a blue strip on the left
    px, py, pw, ph = 150, 250, 130, 30
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
    cv2.rectangle(img, (px, py), (px + 16, py + ph), (180, 90, 20), -1)  # blue NMK strip (BGR)
    cv2.putText(img, "SK1234AB", (px + 20, py + ph - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2)
    return img, (px, py, pw, ph)


def test_returns_sorted_candidates_with_valid_bboxes():
    img, _ = _synthetic_scene()
    cands = detect_candidates(img)
    assert isinstance(cands, list)
    assert all(isinstance(c, Candidate) for c in cands)
    # sorted best-first
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)
    H, W = img.shape[:2]
    for c in cands:
        x, y, w, h = c.bbox
        assert w > 0 and h > 0
        assert 0 <= x and 0 <= y and x + w <= W and y + h <= H
        assert c.cue in ("morph", "color")


def test_empty_image_returns_empty():
    assert detect_candidates(None) == []
    assert detect_candidates(np.zeros((0, 0, 3), np.uint8)) == []


def test_synthetic_plate_is_localized_in_top_candidates():
    img, gt = _synthetic_scene()
    cands = detect_candidates(img)
    assert cands, "no candidates produced for an obvious synthetic plate"
    best = max((_iou(gt, c.bbox) for c in cands[:3]), default=0.0)
    assert best >= IOU_HIT, f"synthetic plate not in top-3 (best IoU {best:.2f})"


@pytest.mark.skipif(not os.path.isdir(PLATESMANIA),
                    reason="platesmania images not present")
def test_real_images_hit_rate_and_overlays():
    files = sorted(glob.glob(os.path.join(PLATESMANIA, "*.jpg")))
    assert len(files) >= 6, "need >=6 real images for the hit-rate harness"
    os.makedirs(SCRATCH, exist_ok=True)
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]  # rank 1/2/3 (BGR)

    produced_any = 0
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        cands = detect_candidates(im)
        produced_any += bool(cands)
        vis = im.copy()
        for i, c in enumerate(cands[:3]):
            x, y, w, h = c.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), colors[i], 2)
            cv2.putText(vis, f"{i + 1}:{c.cue}:{c.score:.2f}", (x, max(11, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
        name = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(SCRATCH, f"detection_{name}.png"), vis)

    # candidates produced for the large majority of scenes
    assert produced_any >= int(0.8 * len(files))

    # IoU regression floor on the hand-verified GT subset: every verified plate must be
    # localized by a top-3 candidate.
    misses = []
    for fname, gt in VERIFIED_GT.items():
        path = os.path.join(PLATESMANIA, fname)
        if not os.path.exists(path):
            continue
        cands = detect_candidates(cv2.imread(path))
        best = max((_iou(gt, c.bbox) for c in cands[:3]), default=0.0)
        if best < IOU_HIT:
            misses.append(f"{fname} best-top3-IoU={best:.2f}")
    assert not misses, "verified plates not localized in top-3: " + "; ".join(misses)
