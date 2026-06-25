"""Shared data contracts passed between alpr/ pipeline stages (cv2 + numpy only)."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Candidate:
    """A detected plate-like region from detection.py."""
    bbox: tuple            # (x, y, w, h) in source-image pixels
    score: float           # higher = more plate-like
    cue: str               # which detector cue produced it ('morph'|'color'|'mser'|...)
    quad: Optional[np.ndarray] = None   # 4x2 float32 corners, if available


@dataclass
class RectifiedPlate:
    """A candidate warped to a canonical fronto-parallel plate by rectify.py."""
    warp: np.ndarray       # BGR canonical plate image (fixed height, aspect ~4.73)
    quad: np.ndarray       # 4x2 source corners used for the warp
    ok: bool               # False if rectification was degenerate (caller may skip)


@dataclass
class GlyphCrop:
    """One segmented character from segmentation.py."""
    norm: np.ndarray       # 32x32 uint8, white glyph on black (alpr.features.normalize form)
    bbox: tuple            # (x, y, w, h) within the rectified plate
    index: int             # left-to-right order


@dataclass
class CharResult:
    """One recognized character."""
    value: str
    confidence: float      # [0,1]
    slot: str = ""         # 'L' (letter) | 'D' (digit) | '' (unassigned)


@dataclass
class PlateResult:
    """Final per-plate result returned by pipeline.read_plate()."""
    bbox: tuple                         # (x, y, w, h) in source-image pixels
    quad: Optional[np.ndarray]
    plate_text: Optional[str]           # None when abstaining (low confidence / invalid)
    region: Optional[str]
    confidence: float                   # [0,1]
    chars: list = field(default_factory=list)   # list[CharResult]
