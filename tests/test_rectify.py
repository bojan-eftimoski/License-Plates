"""Unit tests for alpr.rectify: synthetic perspective warp -> recovered canonical plate,
plus fallback-ladder / rejection cases. Visualizations are saved to data/_scratch/ for
human confirmation that the module actually deskews real plate imagery.
"""
import os

import cv2
import numpy as np
import pytest

from alpr.rectify import CANON_H, CANON_W, rectify
from alpr.types import Candidate, RectifiedPlate

CLEAN_PLATE = "data/raw/olavsplates/nmk_st0915ad_close.jpg"
SCRATCH = "data/_scratch"


def _on_gray_canvas(plate, pad_frac=0.30, fill=120):
    """Place a frontal plate crop centered on a uniform gray canvas (room to skew into)."""
    h, w = plate.shape[:2]
    pad = int(pad_frac * w)
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), fill, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = plate
    return canvas, pad, w, h


def _apply_perspective(plate, strength=0.18):
    """Apply a KNOWN perspective warp to a frontal plate, returning the warped scene."""
    canvas, pad, w, h = _on_gray_canvas(plate)
    H, W = canvas.shape[:2]
    src = np.float32([[pad, pad], [pad + w, pad],
                      [pad + w, pad + h], [pad, pad + h]])
    dx, dy = strength * w, strength * h
    dst = np.float32([
        [pad + dx, pad + dy],
        [pad + w - dx * 0.4, pad],
        [pad + w, pad + h - dy * 0.5],
        [pad - dx * 0.2, pad + h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(canvas, M, (W, H), borderValue=(120, 120, 120))


@pytest.fixture(scope="module")
def plate():
    if not os.path.isfile(CLEAN_PLATE):
        pytest.skip(f"missing fixture image {CLEAN_PLATE}")
    img = cv2.imread(CLEAN_PLATE)
    if img is None:
        pytest.skip(f"could not read {CLEAN_PLATE}")
    return img


def test_canonical_dimensions_are_locked():
    # Height 96, aspect 4.727 -> width 454 (spec §rectify).
    assert CANON_H == 96
    assert CANON_W == 454


def test_rectify_deskews_synthetic_perspective_warp(plate):
    """Take a clean plate, perspective-warp it, then rectify -> canonical & ok."""
    warped = _apply_perspective(plate, strength=0.18)
    H, W = warped.shape[:2]
    cand = Candidate(bbox=(0, 0, W, H), score=1.0, cue="test")

    res = rectify(warped, cand)

    assert isinstance(res, RectifiedPlate)
    assert res.warp.shape == (CANON_H, CANON_W, 3)
    assert res.ok is True
    # quad must be 4x2 source corners, all inside the warped scene.
    assert res.quad.shape == (4, 2)
    assert (res.quad[:, 0] >= -1).all() and (res.quad[:, 0] <= W + 1).all()
    assert (res.quad[:, 1] >= -1).all() and (res.quad[:, 1] <= H + 1).all()
    # recovered output must be plate-shaped (~4.73) by construction (96 x 454).
    aspect = res.warp.shape[1] / res.warp.shape[0]
    assert 4.0 <= aspect <= 5.5
    # real content, not a collapsed/blank warp.
    assert cv2.cvtColor(res.warp, cv2.COLOR_BGR2GRAY).std() > 20.0

    os.makedirs(SCRATCH, exist_ok=True)
    cv2.imwrite(f"{SCRATCH}/rectify_test_before.png", warped)
    cv2.imwrite(f"{SCRATCH}/rectify_test_after.png", res.warp)


def test_rectify_frontal_plate_passes(plate):
    """An already-frontal close-up rectifies to canonical and is ok."""
    h, w = plate.shape[:2]
    res = rectify(plate, Candidate(bbox=(0, 0, w, h), score=1.0, cue="test"))
    assert res.warp.shape == (CANON_H, CANON_W, 3)
    assert res.ok is True


def test_rectify_rotated_plate_deskews(plate):
    """Pure in-plane rotation exercises the minAreaRect deskew rung; still canonical+ok."""
    canvas, pad, w, h = _on_gray_canvas(plate)
    cH, cW = canvas.shape[:2]
    M = cv2.getRotationMatrix2D((cW / 2, cH / 2), 12.0, 1.0)
    rot = cv2.warpAffine(canvas, M, (cW, cH), borderValue=(120, 120, 120))
    res = rectify(rot, Candidate(bbox=(0, 0, cW, cH), score=1.0, cue="test"))
    assert res.warp.shape == (CANON_H, CANON_W, 3)
    assert res.ok is True


def test_rectify_rejects_flat_region():
    """A constant-color (non-plate) region must yield ok=False, not a confident warp."""
    flat = np.full((120, 480, 3), 90, np.uint8)
    res = rectify(flat, Candidate(bbox=(0, 0, 480, 120), score=1.0, cue="test"))
    assert res.warp.shape == (CANON_H, CANON_W, 3)
    assert res.ok is False


def test_rectify_rejects_wrong_aspect_crop(plate):
    """A square (wrong-aspect) crop falling through to the axis-aligned rung is not ok."""
    sq = cv2.resize(plate, (200, 200))
    res = rectify(sq, Candidate(bbox=(0, 0, 200, 200), score=1.0, cue="test"))
    assert res.warp.shape == (CANON_H, CANON_W, 3)
    assert res.ok is False


def test_rectify_tiny_bbox_is_safe(plate):
    """A degenerate tiny bbox never crashes; returns a canonical-shaped, not-ok result."""
    res = rectify(plate, Candidate(bbox=(0, 0, 3, 2), score=1.0, cue="test"))
    assert res.warp.shape == (CANON_H, CANON_W, 3)
    assert res.ok is False
