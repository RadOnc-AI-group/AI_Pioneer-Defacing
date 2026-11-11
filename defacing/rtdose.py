"""
Masked RTDOSE (--mask-rtdose): the dose is set to zero in the de-faced region, so that the
distribution no longer outlines the face.

A dose voxel is set to zero if its cell touches a de-faced CT voxel (dose grids are coarser
than the CT), unless its centre lies in a kept target volume, which keeps its dose. The
positions are compared in patient coordinates, so grid direction and orientation do not matter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from .common import PatientError
from .masks import dilate_mm

DOSE_MARK = "De-faced: RTDOSE set to zero in the de-faced region"


def dose_positions(ds: Dataset) -> tuple[list[np.ndarray], float]:
    """LPS positions (rows, cols, 3) of the voxels of every frame, and the frame distance in mm."""
    iop = np.array(ds.ImageOrientationPatient, float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    origin = np.array(ds.ImagePositionPatient, float)
    dr, dc = (float(v) for v in ds.PixelSpacing)
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    if frames > 1:
        offsets = np.array(ds.GridFrameOffsetVector, float)
        if offsets.size != frames:
            raise ValueError("GridFrameOffsetVector does not match NumberOfFrames")
        offsets = offsets - offsets[0]         # relative (first = 0) or absolute (first = z of the origin)
    else:
        offsets = np.zeros(1)
    r, c = np.meshgrid(np.arange(int(ds.Rows)), np.arange(int(ds.Columns)), indexing="ij")
    plane = origin + r[..., None] * dr * col + c[..., None] * dc * row
    step = float(np.median(np.abs(np.diff(offsets)))) if frames > 1 else float(ds.get("SliceThickness", 0) or 0)
    return [plane + f * normal for f in offsets], step


def voxels_to_zero(ds: Dataset, series: list) -> np.ndarray:
    """Boolean (frames, rows, cols): dose voxels to set to zero, over all CT series in `series`."""
    planes, step = dose_positions(ds)
    dr, dc = (float(v) for v in ds.PixelSpacing)
    zero = np.zeros((len(planes), *planes[0].shape[:2]), bool)
    dose_cell = 0.5 * float(np.linalg.norm([dr, dc, step]))
    for s in series:
        if not s.region.any():
            continue
        spacing_zyx = tuple(s.grid.spacing[::-1])
        radius = dose_cell + 0.5 * float(np.linalg.norm(s.grid.spacing))
        near = dilate_mm(s.region, spacing_zyx, radius)
        size = np.array(s.grid.size)
        for k, plane in enumerate(planes):
            idx = np.rint(s.grid.to_index(plane.reshape(-1, 3))).astype(int)
            # outside the CT: the nearest CT voxel decides (keeps the image edge zero in front of the face)
            idx = np.clip(idx, 0, size - 1)
            hit = near[idx[:, 2], idx[:, 1], idx[:, 0]]
            if s.keep is not None:
                hit &= ~s.keep[idx[:, 2], idx[:, 1], idx[:, 0]]
            zero[k] |= hit.reshape(plane.shape[:2])
    return zero


def mask_dose(ds: Dataset, series: list) -> tuple[np.ndarray, np.ndarray]:
    """Set the dose to zero in the de-faced region of `series`, in place. Returns (the voxels
    set to zero, the original dose), both (frames, rows, cols), for verify_dose."""
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    dose = ds.pixel_array.copy()
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    shape = (frames, int(ds.Rows), int(ds.Columns))
    original = dose.reshape(shape)
    dose = original.copy()
    zero = voxels_to_zero(ds, series)
    dose[zero] = 0
    ds.PixelData = dose.tobytes()
    previous = ds.get("DeidentificationMethod")
    previous = [] if not previous else ([str(previous)] if isinstance(previous, str) else [str(v) for v in previous])
    ds.DeidentificationMethod = previous + [DOSE_MARK]
    return zero, original


def verify_dose(path: Path, original: np.ndarray, zero: np.ndarray) -> None:
    """Read the written file back: zero inside, unchanged outside."""
    written = pydicom.dcmread(path).pixel_array.reshape(zero.shape)
    if written[zero].any() or not np.array_equal(written[~zero], original[~zero]):
        raise PatientError(f"verification: the masked RTDOSE {path.name} does not hold the expected values")
