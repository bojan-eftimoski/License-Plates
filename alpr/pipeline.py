"""End-to-end classical ALPR pipeline: detect -> rectify -> segment -> OCR -> validate.

read_plate(image) -> list[PlateResult]. Detect-always / read-when-confident: every plausible
plate gets a box; plate_text is set only when segmentation + validation succeed and confidence
clears the abstain threshold. cv2 + numpy only.
"""
import cv2
import numpy as np

from .detection import detect_candidates
from .ocr import GlyphClassifier, load_glyph_dataset
from .rectify import rectify
from .segmentation import segment
from .types import CharResult, PlateResult
from .validation import validate


class ALPR:
    def __init__(self, templates_dir: str = "data/templates", abstain_threshold: float = 0.5):
        imgs, labels = load_glyph_dataset(templates_dir)
        self.clf = GlyphClassifier().fit(imgs, labels)
        self.abstain_threshold = abstain_threshold

    def read_plate(self, image_bgr: np.ndarray, max_candidates: int = 14) -> list[PlateResult]:
        results: list[PlateResult] = []
        for cand in detect_candidates(image_bgr)[:max_candidates]:
            try:
                results.append(self._read_candidate(image_bgr, cand))
            except cv2.error:
                # one malformed candidate must not abort the whole image
                results.append(PlateResult(cand.bbox, cand.quad, None, None, 0.0, []))
        return _dedup(results)

    def _read_candidate(self, image_bgr, cand) -> PlateResult:
        rp = rectify(image_bgr, cand)
        if not rp.ok:
            return PlateResult(cand.bbox, cand.quad, None, None, 0.0, [])
        glyphs = segment(rp.warp)
        if not (7 <= len(glyphs) <= 8):
            return PlateResult(cand.bbox, rp.quad, None, None, 0.0, [])
        chars = [CharResult(*self.clf.classify(g.norm)) for g in glyphs]
        text, region, conf, slotted = validate(chars)
        if text is None or conf < self.abstain_threshold:
            text = None  # detect-always, read-when-confident
        return PlateResult(cand.bbox, rp.quad, text, region, conf, slotted)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _dedup(results: list[PlateResult], iou_thresh: float = 0.5) -> list[PlateResult]:
    """Keep the highest-confidence result among overlapping boxes."""
    kept: list[PlateResult] = []
    for r in sorted(results, key=lambda r: r.confidence, reverse=True):
        if all(_iou(r.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(r)
    return kept
