"""Shared constants, options and small helpers."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("deface")

FACE_VALUE_HU = -1000.0  # value of the "constant" fill (and nominal air)
FILL_MODES = ("air", "soft-tissue", "constant")

# Recognized image file extensions, longest-first so ".nii.gz" wins over ".nii"
KNOWN_SUFFIXES = (".nii.gz", ".nii", ".nrrd")


class PatientError(Exception):
    """A patient could not be processed; in batch mode the run continues with the next one."""


@dataclass
class Options:
    device: str = "cpu"
    dilate_mm: float = 3.0          # margin around the face mask
    dilate: int = 2                 # the margin in voxels, if the voxel size is unknown
    extend_anterior: bool = True
    fill: str = "air"
    seed: int | None = None
    keep_body: bool = False
    keep_skin: bool = False
    keep_rtplan: bool = False
    keep_rtdose: bool = False
    remove_roi: list[re.Pattern] = field(default_factory=list)
    pii_check: bool = True
    allow_empty_mask: bool = False

    def settings(self) -> dict:
        """Settings that change the output; stored in the manifest of every patient."""
        return {"dilate_mm": self.dilate_mm, "dilate": self.dilate, "extend_anterior": self.extend_anterior,
                "fill": self.fill, "keep_body": self.keep_body, "keep_skin": self.keep_skin,
                "keep_rtplan": self.keep_rtplan, "keep_rtdose": self.keep_rtdose,
                "remove_roi": [p.pattern for p in self.remove_roi]}


def classify_suffix(path: Path) -> str:
    """
    Return the recognized extension of path ('.nii.gz', '.nii', '.nrrd') or ''.

    Unlike Path.suffixes, this only looks at the end of the filename, so
    filenames with extra dots (DICOM-UID-style names, versioned filenames,
    etc.) are still classified correctly.
    """
    name = path.name.lower()
    for suffix in KNOWN_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def strip_known_suffix(path: Path) -> str:
    """Return the filename with its recognized image extension removed."""
    suffix = classify_suffix(path)
    if suffix:
        return path.name[: -len(suffix)]
    return path.stem


def safe_cast(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Round/clip data into dtype's representable range before casting, to avoid wraparound."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        data = np.clip(np.round(data), info.min, info.max)
    return data.astype(dtype)


def is_hidden(rel: Path) -> bool:
    """macOS AppleDouble files ("._x.dcm"), .DS_Store and anything in hidden folders."""
    return any(part.startswith(".") for part in rel.parts)
