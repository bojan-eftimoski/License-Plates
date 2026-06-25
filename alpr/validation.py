"""MK plate-format validation + confusable resolution (domain rules, cv2/numpy-free).

Takes the ordered per-glyph OCR results and enforces the Macedonian post-2012 plate
format `LL DDD(D) LL` (2 region letters + 3-4 digits + 2 suffix letters). Each glyph is
mapped to its slot class and digit<->letter cross-confusions are coerced to the slot's
class (a digit-slot 'O' becomes '0', a letter-slot '0' becomes 'O'). A plate is valid only
when every slot coerces, the region code is in the 34-code whitelist, and no banned glyph
(Q/W/X/Y) appears. Confidence is the weakest-link min over glyphs, gated to 0 if invalid
(see spec 5.4). No deep learning, no Tesseract, no sklearn.
"""
from typing import Optional

from .types import CharResult

# The 34 authoritative MK region codes (spec section 2). First two letters MUST be one of
# these; do NOT overfit to SK even though most supplied samples are SK.
REGION_CODES = {
    "BE", "BT", "DB", "DE", "DH", "DK", "GE", "GV", "KA", "KI", "KO", "KR",
    "KP", "KS", "KU", "MB", "MK", "NE", "OH", "PE", "PP", "PS", "RA", "RE",
    "SK", "SN", "SU", "SR", "ST", "TE", "VA", "VE", "VI", "VV",
}

# Latin alphabet minus Q/W/X/Y never appears on MK plates.
BANNED = set("QWXY")

# digit<->letter cross-confusions that slot position resolves (spec section 2).
DIGIT_FROM_LETTER = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
LETTER_FROM_DIGIT = {v: k for k, v in DIGIT_FROM_LETTER.items()}


def _coerce(ch: str, slot: str) -> Optional[str]:
    """Coerce `ch` into the class required by `slot` ('D' digit / 'L' letter).

    Returns the coerced character, or None if it cannot legally fill the slot.
    """
    ch = ch.upper()
    if ch in BANNED:
        return None
    if slot == "D":
        if ch.isdigit():
            return ch
        return DIGIT_FROM_LETTER.get(ch)          # letter mistaken for a digit
    # slot == 'L'
    if ch.isalpha():
        return ch
    return LETTER_FROM_DIGIT.get(ch)              # digit mistaken for a letter


def _slot_template(n: int) -> Optional[list[str]]:
    """Slot classes for an n-glyph plate: LL DDD(D) LL. n must be 7 or 8."""
    if n == 8:
        return ["L", "L", "D", "D", "D", "D", "L", "L"]   # 4-digit (modern)
    if n == 7:
        return ["L", "L", "D", "D", "D", "L", "L"]        # 3-digit (legacy)
    return None


def validate(chars: list[CharResult]):
    """Validate ordered OCR chars against the MK format and resolve confusables.

    Returns (plate_text, region, confidence, slotted_chars):
      - plate_text: the joined coerced glyphs when valid, else None.
      - region:     the (coerced) first two letters when valid, else None.
      - confidence: min per-glyph confidence * (1.0 if valid else 0.0).
      - slotted_chars: the same CharResult objects with `.slot` filled ('L'/'D').
    """
    template = _slot_template(len(chars))

    # Always fill slots when the length is templatable, so callers/debuggers see them.
    if template is not None:
        for c, slot in zip(chars, template):
            c.slot = slot

    base_conf = min((c.confidence for c in chars), default=0.0)

    if template is None:
        return None, None, 0.0, chars

    coerced = [_coerce(c.value, slot) for c, slot in zip(chars, template)]
    valid = all(cc is not None for cc in coerced)

    region = None
    plate_text = None
    if valid:
        region = "".join(coerced[:2])
        if region not in REGION_CODES:
            valid = False

    if valid:
        plate_text = "".join(coerced)
        # Write the resolved character back so downstream uses the coerced glyph.
        for c, cc in zip(chars, coerced):
            c.value = cc
    else:
        region = None

    confidence = base_conf * (1.0 if valid else 0.0)
    return plate_text, region, confidence, chars
