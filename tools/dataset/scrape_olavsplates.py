"""Scrape North Macedonian plate images from olavsplates.com.

Each image filename encodes the plate string (e.g. foto_n/nmk_st0915ad.jpg -> ST 0915 AD),
so this yields auto-labeled real plates. `_close.jpg` variants are tight plate crops ideal
for glyph extraction. Reference-only enthusiast source: attribute olavsplates.com in the
report; images land under data/raw/ (gitignored), not redistributed.
"""
import csv
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://olavsplates.com/"
PAGES = ["macedonia.html", "macedonia_duplicates.html", "macedonia_abroad.html"]
OUT = Path("data/raw/olavsplates")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()


def parse_label(fname: str):
    """foto_m/mk_bt256-bv.jpg -> (era, plate_string, region). Returns (None,None,None) if no match."""
    m = re.search(r"/(n?mk)_([a-z0-9-]+?)(_close)?\.jpg$", fname, re.I)
    if not m:
        return None, None, None
    era = "NMK" if m.group(1).lower() == "nmk" else "MK"
    plate = re.sub(r"[^a-z0-9]", "", m.group(2), flags=re.I).upper()
    region = plate[:2] if len(plate) >= 2 and plate[:2].isalpha() else None
    return era, plate, region


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    for page in PAGES:
        try:
            html = fetch(BASE + page).decode("utf-8", "replace")
        except Exception as e:
            print(f"[skip] {page}: {e}")
            continue
        srcs = re.findall(r'<img[^>]+src="([^"]+)"', html, re.I)
        plate_srcs = [s for s in srcs if re.search(r"foto_[mn]/", s)]
        print(f"[page] {page}: {len(plate_srcs)} plate images")
        for src in plate_srcs:
            url = BASE + src.lstrip("/")
            if url in seen:
                continue
            seen.add(url)
            era, plate, region = parse_label(src)
            if not plate:
                continue
            is_close = "_close" in src
            local = OUT / Path(src).name
            if not local.exists():
                try:
                    local.write_bytes(fetch(url))
                    time.sleep(0.3)
                except Exception as e:
                    print(f"  [fail] {url}: {e}")
                    continue
            rows.append(dict(local_file=local.name, source_url=url, era=era,
                             plate_string=plate, region=region, is_closeup=int(is_close)))

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["local_file", "source_url", "era",
                                          "plate_string", "region", "is_closeup"])
        w.writeheader()
        w.writerows(rows)

    closeups = sum(r["is_closeup"] for r in rows)
    regions = sorted({r["region"] for r in rows if r["region"]})
    print(f"\nDownloaded {len(rows)} images ({closeups} close-ups, {len(rows)-closeups} full).")
    print(f"Distinct region codes: {len(regions)} -> {regions}")
    print(f"Manifest: {OUT/'manifest.csv'}")


if __name__ == "__main__":
    main()
