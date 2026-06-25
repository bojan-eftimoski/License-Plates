"""Held-out accuracy gate for the graded ALPR pipeline.

Runs alpr.read_plate over data/eval/ground_truth.csv (labeled rows) and reports REAL
plate-exact and character-level accuracy. Usage:
    .venv/Scripts/python tools/eval.py [abstain_threshold]
"""
import csv
import sys

import cv2

from alpr.pipeline import ALPR

GT = "data/eval/ground_truth.csv"


def load_gt(path: str = GT):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["plate_string"]:
                rows.append((r["file"], r["plate_string"].split("|")))
    return rows


def char_matches(gt: str, preds: set[str]) -> int:
    """Max positional character overlap between gt and any predicted string."""
    return max((sum(a == b for a, b in zip(gt, p)) for p in preds), default=0)


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    alpr = ALPR(abstain_threshold=thr)
    gt = load_gt()

    images = plate_total = plate_hit = char_total = char_hit = detected_imgs = 0
    for file, plates in gt:
        img = cv2.imread(file)
        if img is None:
            print(f"  [skip missing] {file}")
            continue
        images += 1
        results = alpr.read_plate(img)
        preds = {r.plate_text for r in results if r.plate_text}
        if any(r.bbox for r in results):
            detected_imgs += 1
        for gtp in plates:
            plate_total += 1
            plate_hit += gtp in preds
            char_hit += char_matches(gtp, preds)
            char_total += len(gtp)

    print(f"\n=== Held-out eval (abstain_threshold={thr}) ===")
    print(f"images evaluated: {images}   plates: {plate_total}   images with a detection: {detected_imgs}")
    if plate_total:
        print(f"PLATE-EXACT accuracy: {plate_hit}/{plate_total} = {plate_hit / plate_total:.1%}")
    if char_total:
        print(f"CHAR-LEVEL  accuracy: {char_hit}/{char_total} = {char_hit / char_total:.1%}")


if __name__ == "__main__":
    main()
