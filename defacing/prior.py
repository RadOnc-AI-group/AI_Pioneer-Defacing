"""
Signs that a CT was de-faced before. They only add a note to the report: a scan is always
processed as usual, because a poorly de-faced scan must not pass through unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .masks import Axes

_DEFACED_TEXT = re.compile(r"de-?fac")
BLOCK_MIN_ML = 20.0             # one connected block of a single air value of at least this size ...
OTHER_AIR_MAX_FRACTION = 0.5    # ... while less than this share of the air behind the head has that value
AIR_RANGE = (-1100.0, -500.0)   # below: padding outside the field of view; above: tissue
MAX_VALUES = 5                  # the most frequent exact air values that are looked at


def _text(ds, keyword: str) -> str:
    value = ds.get(keyword)
    if value is None or isinstance(value, str):
        return value or ""
    return " ".join(str(v) for v in value)


def header_evidence(ds) -> str | None:
    """The DICOM header says the image was de-faced."""
    for item in ds.get("DeidentificationMethodCodeSequence", []):
        if str(item.get("CodeValue", "")) == "113102":
            return "DeidentificationMethodCodeSequence has code 113102 (Clean Recognizable Visual Features)"
    for keyword in ("DeidentificationMethod", "ImageComments"):
        if _DEFACED_TEXT.search(_text(ds, keyword).lower()):
            return f"{keyword} mentions de-facing"
    return None


def constant_block(hu: np.ndarray, axes: Axes) -> tuple[float, float] | None:
    """(value, ml) of a large block of one exact air value in the anterior half of the image,
    if that value is not simply the scan's air (e.g. clipping at -1024 HU or -1000 HU)."""
    from scipy.ndimage import label
    axis, increasing = axes.anterior
    n = hu.shape[axis]
    front = [slice(None)] * hu.ndim
    back = [slice(None)] * hu.ndim
    front[axis], back[axis] = (slice(n // 2, n), slice(0, n // 2)) if increasing else (slice(0, n // 2), slice(n // 2, n))
    anterior, posterior = hu[tuple(front)], hu[tuple(back)]
    in_air = (anterior > AIR_RANGE[0]) & (anterior <= AIR_RANGE[1])
    if not in_air.any():
        return None
    values, counts = np.unique(anterior[in_air], return_counts=True)
    voxel_ml = float(np.prod(axes.spacing)) / 1000.0
    air_behind = posterior[(posterior > AIR_RANGE[0]) & (posterior <= AIR_RANGE[1])]
    # several candidates: in air clipped at -1024 HU, that value can be more frequent than the block
    for i in np.argsort(counts)[::-1][:MAX_VALUES]:
        value, count = float(values[i]), int(counts[i])
        if count * voxel_ml < BLOCK_MIN_ML:
            break
        if air_behind.size and (air_behind == value).mean() >= OTHER_AIR_MAX_FRACTION:
            continue
        labels, found = label(anterior == value)
        if found:
            largest = int(np.bincount(labels.ravel())[1:].max())
            if largest * voxel_ml >= BLOCK_MIN_ML:
                return value, largest * voxel_ml
    return None


def evidence(hu: np.ndarray, axes: Axes, ds=None, path: Path | None = None) -> str | None:
    """Why the image looks already de-faced, or None."""
    if ds is not None:
        found = header_evidence(ds)
        if found:
            return found
    if path is not None and _DEFACED_TEXT.search(path.name.lower()):
        return f"the file name contains {path.name!r}"
    block = constant_block(hu, axes)
    if block:
        return f"a block of {block[1]:.0f} ml in front holds exactly {block[0]:g} HU, unlike the air elsewhere"
    return None
