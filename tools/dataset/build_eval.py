"""Assemble data/eval/ground_truth.csv from the labeled real sources.

Held-out eval = Platesmania (distinct plates, never used for templates) + repo registracii1
(known labels). Repo hard images (registracii2/3, IMG_*) are listed detect-only (no OCR label).
olavsplates is deliberately excluded from eval — its close-ups feed the templates, so scoring
on it would leak. Re-run after adding/labeling more real images.
"""
import csv
from pathlib import Path

EVAL = Path("data/eval")
PM_MANIFEST = Path("data/raw/platesmania_mk/manifest.csv")

# repo images with confidently hand-read labels / difficulty
REPO_LABELED = [("Images/registracii1.jpg", "SK9507BT|SK9481BT|SK7168BG", "SK", "easy")]
REPO_DETECT_ONLY = [("Images/registracii2.jpg", "hard"), ("Images/registracii3.jpg", "hard"),
                    ("Images/IMG_6449.png", "hard"), ("Images/IMG_6450.png", "hard"),
                    ("Images/IMG_6451.png", "hard")]


def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    rows = []
    if PM_MANIFEST.exists():
        with PM_MANIFEST.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(file=f"data/raw/platesmania_mk/{r['local_file']}",
                                 plate_string=r["plate_string"], region=r["region"],
                                 source="platesmania", difficulty="real-traffic", split="test"))
    for fn, plate, region, diff in REPO_LABELED:
        rows.append(dict(file=fn, plate_string=plate, region=region,
                         source="repo", difficulty=diff, split="test"))
    for fn, diff in REPO_DETECT_ONLY:
        rows.append(dict(file=fn, plate_string="", region="",
                         source="repo", difficulty=diff, split="detect-only"))

    with (EVAL / "ground_truth.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "plate_string", "region", "source", "difficulty", "split"])
        w.writeheader()
        w.writerows(rows)

    labeled = [r for r in rows if r["plate_string"]]
    plates = {p for r in labeled for p in r["plate_string"].split("|")}
    print(f"ground_truth.csv: {len(rows)} rows, {len(labeled)} labeled, "
          f"{len(plates)} distinct plates, regions {sorted({p[:2] for p in plates})}")


if __name__ == "__main__":
    main()
