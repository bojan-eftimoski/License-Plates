import pytest

from alpr.types import CharResult
from alpr.validation import (
    DIGIT_FROM_LETTER,
    LETTER_FROM_DIGIT,
    REGION_CODES,
    validate,
)


def chars(values, conf=0.9):
    """Build ordered CharResult list from a sequence of glyph strings."""
    return [CharResult(value=v, confidence=conf) for v in values]


def test_region_codes_are_the_34_authoritative_codes():
    assert len(REGION_CODES) == 34
    for banned in ("QQ", "WW", "XX", "YY"):
        assert banned not in REGION_CODES
    # spot-check a few from the spec
    for code in ("SK", "BT", "VV", "MK", "OH"):
        assert code in REGION_CODES


def test_valid_8char_with_digit_slot_letter_O_coerced_to_zero():
    # SK 9507 BT, but OCR emitted 'O' in a digit slot -> must become '0'.
    cs = chars(["S", "K", "9", "5", "O", "7", "B", "T"])
    text, region, conf, slotted = validate(cs)
    assert text == "SK9507BT"
    assert region == "SK"
    assert conf == pytest.approx(0.9)
    assert [c.slot for c in slotted] == ["L", "L", "D", "D", "D", "D", "L", "L"]


def test_valid_3digit_legacy_plate():
    cs = chars(["B", "T", "2", "5", "6", "B", "V"])
    text, region, conf, slotted = validate(cs)
    assert text == "BT256BV"
    assert region == "BT"
    assert conf == pytest.approx(0.9)
    assert [c.slot for c in slotted] == ["L", "L", "D", "D", "D", "L", "L"]


def test_bad_region_rejected_with_zero_confidence():
    cs = chars(["Q", "Q", "9", "5", "0", "7", "B", "T"])
    text, region, conf, _ = validate(cs)
    assert text is None
    assert region is None
    assert conf == 0.0


def test_region_not_in_whitelist_rejected():
    # ZZ is two valid letters but not a real region code -> reject (no SK-overfit).
    cs = chars(["Z", "Z", "9", "5", "0", "7", "B", "T"])
    text, region, conf, _ = validate(cs)
    assert text is None and region is None and conf == 0.0


def test_letter_slot_digit_coerced_to_letter():
    # GE region, suffix letters mis-read as digits 8->B, 0->O.
    cs = chars(["G", "E", "4", "5", "1", "4", "8", "0"])
    text, region, conf, _ = validate(cs)
    assert text == "GE4514BO"
    assert region == "GE"
    assert conf == pytest.approx(0.9)


def test_banned_glyph_in_letter_slot_rejected():
    cs = chars(["S", "K", "1", "2", "9", "7", "A", "W"])  # W banned
    text, region, conf, _ = validate(cs)
    assert text is None and region is None and conf == 0.0


def test_wrong_length_rejected():
    cs = chars(["S", "K", "9", "5", "0"])  # length 5, not templatable
    text, region, conf, slotted = validate(cs)
    assert text is None and region is None and conf == 0.0
    # untemplatable length leaves slots unassigned
    assert all(c.slot == "" for c in slotted)


def test_confidence_is_weakest_link_min():
    cs = [
        CharResult("S", 0.95), CharResult("K", 0.80), CharResult("1", 0.99),
        CharResult("2", 0.70), CharResult("9", 0.99), CharResult("7", 0.99),
        CharResult("A", 0.99), CharResult("S", 0.99),
    ]
    text, _, conf, _ = validate(cs)
    assert text == "SK1297AS"
    assert conf == pytest.approx(0.70)  # min over glyphs, valid -> *1.0


def test_confusable_maps_are_inverse():
    for letter, digit in DIGIT_FROM_LETTER.items():
        assert LETTER_FROM_DIGIT[digit] == letter


def test_slots_filled_even_when_invalid_but_templatable():
    cs = chars(["Q", "Q", "9", "5", "0", "7", "B", "T"])
    _, _, _, slotted = validate(cs)
    assert [c.slot for c in slotted] == ["L", "L", "D", "D", "D", "D", "L", "L"]


@pytest.mark.parametrize("plate,region,n_digits", [
    ("BT0999AB", "BT", 4),
    ("GE4514AC", "GE", 4),
    ("GV3410AC", "GV", 4),
    ("OH1111KA", "OH", 4),
    ("SK1297AS", "SK", 4),
    ("SU1681AE", "SU", 4),
    ("SK9507BT", "SK", 4),
    ("BT256BV", "BT", 3),
    ("SK280BC", "SK", 3),
    ("SK915UU", "SK", 3),
])
def test_real_groundtruth_plates_validate_perfectly(plate, region, n_digits):
    """Every real eval-set label must pass the validator unchanged at full confidence."""
    cs = chars(list(plate))
    text, reg, conf, slotted = validate(cs)
    assert text == plate
    assert reg == region
    assert conf == pytest.approx(0.9)
    assert sum(c.slot == "D" for c in slotted) == n_digits
