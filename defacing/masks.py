"""Face mask: alignment to the image grid, dilation and extension to the anterior image edge."""
from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .common import PatientError

BODY_HU = -500.0            # tissue, couch and devices are above this


@dataclass
class Axes:
    """Orientation of a numpy volume: voxel size per numpy axis (mm), the numpy axis that
    points most nearly to the patient's anterior (and whether the index increases that
    way), the numpy axis closest to the head-feet direction, and whether the index
    increases towards the head along it."""
    spacing: tuple[float, float, float]
    anterior: tuple[int, bool]
    si_axis: int
    superior: bool = True


def _axis_towards(index_to_world: np.ndarray, direction: np.ndarray) -> tuple[int, bool]:
    """Index axis that points most nearly along `direction` (world coordinates), and
    whether the index increases in that direction. index_to_world's columns are the
    index axes; they are normalised, so voxel sizes do not bias the choice."""
    cols = index_to_world / np.linalg.norm(index_to_world, axis=0, keepdims=True)
    along = direction @ cols
    axis = int(np.argmax(np.abs(along)))
    return axis, bool(along[axis] > 0)


def sitk_axes(img: sitk.Image) -> Axes:
    """Axes of sitk.GetArrayFromImage(img): numpy axes are (z, y, x), world space is LPS."""
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    a_axis, a_inc = _axis_towards(direction, np.array([0.0, -1.0, 0.0]))   # anterior = -y
    s_axis, s_inc = _axis_towards(direction, np.array([0.0, 0.0, 1.0]))    # superior = +z
    return Axes(spacing=tuple(float(s) for s in img.GetSpacing()[::-1]),
                anterior=(2 - a_axis, a_inc), si_axis=2 - s_axis, superior=s_inc)


def nifti_axes(img: nib.Nifti1Image) -> Axes:
    """Axes of a nibabel image's data array: numpy axes are (i, j, k), world space is RAS."""
    affine = img.affine[:3, :3]
    a_axis, a_inc = _axis_towards(affine, np.array([0.0, 1.0, 0.0]))       # anterior = +y
    s_axis, s_inc = _axis_towards(affine, np.array([0.0, 0.0, 1.0]))       # superior = +z
    return Axes(spacing=tuple(float(z) for z in img.header.get_zooms()[:3]),
                anterior=(a_axis, a_inc), si_axis=s_axis, superior=s_inc)


def extend_mask_anteriorly(mask: np.ndarray, axis: int, increasing: bool) -> np.ndarray:
    """
    Add every voxel between the mask and the anterior image edge, line by line along `axis`.

    Lines that do not touch the mask stay unchanged, and nothing posterior of the most
    anterior mask voxel of a line is added, so tissue behind the face is not affected.
    This removes anything lying in front of the face, such as a thermoplastic RT
    immobilisation mask moulded to the face or a bolus.
    """
    m = mask if increasing else np.flip(mask, axis)
    # True at and posterior of the most anterior mask voxel of each line
    behind_front = np.flip(np.logical_or.accumulate(np.flip(m, axis), axis=axis), axis)
    out = m | (m.any(axis=axis, keepdims=True) & ~behind_front)
    return out if increasing else np.flip(out, axis)


def tissue_around_mask(hu: np.ndarray, mask: np.ndarray, axis: int, increasing: bool) -> tuple[int, int]:
    """
    Voxels above BODY_HU (outside the mask) in front of the most anterior and behind the
    most posterior mask voxel, summed over the lines along `axis` that touch the mask.
    """
    if not increasing:                                   # index increasing towards anterior
        hu, mask = np.flip(hu, axis), np.flip(mask, axis)
    body = (hu > BODY_HU) & ~mask
    lines = mask.any(axis=axis, keepdims=True)
    # True at and in front of the most posterior / at and behind the most anterior mask voxel
    from_back = np.logical_or.accumulate(mask, axis=axis)
    from_front = np.flip(np.logical_or.accumulate(np.flip(mask, axis), axis=axis), axis)
    front = int((body & lines & ~from_front).sum())
    back = int((body & lines & ~from_back).sum())
    return front, back


def check_face_anterior(hu: np.ndarray, mask: np.ndarray, axis: int, increasing: bool) -> None:
    """
    The face lies at the front of the head: on the lines through the face mask there is
    much more tissue behind the mask than in front of it (where only air, an immobilisation
    mask or, for prone scans, the couch can be). If not, the orientation in the header is
    wrong (wrong PatientPosition, flipped NIfTI/NRRD orientation) and extending the mask
    "anteriorly" would erase the head instead of what lies in front of the face.
    """
    front, back = tissue_around_mask(hu, mask, axis, increasing)
    if front > back and front > 100:
        raise PatientError(f"the face mask does not lie at the anterior side of the head ({front} tissue voxels "
                           f"in front of it, {back} behind it): the orientation stored in the image header is "
                           "probably wrong (PatientPosition, NIfTI/NRRD orientation). Extending the mask to the "
                           "'anterior' image edge would erase the head; nothing was written for this patient")


def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    """True if two images occupy exactly the same voxel grid (size + physical space)."""
    return (a.GetSize() == b.GetSize()
            and a.GetSpacing() == b.GetSpacing()
            and a.GetOrigin() == b.GetOrigin()
            and a.GetDirection() == b.GetDirection())


def _dilate(mask: np.ndarray, dilate_vox: int) -> np.ndarray:
    if dilate_vox > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilate_vox)
    return mask


def dilate_mm(mask: np.ndarray, spacing: tuple, mm: float) -> np.ndarray:
    """Every voxel whose centre lies within `mm` of a mask voxel centre (Euclidean, with the
    voxel size `spacing` per numpy axis), so the margin is the same in every direction."""
    if mm <= 0 or not mask.any():
        return mask
    from scipy.ndimage import distance_transform_edt
    pad = [int(np.ceil(mm / s)) for s in spacing]
    idx = np.nonzero(mask)
    crop = tuple(slice(max(0, i.min() - p), i.max() + p + 1) for i, p in zip(idx, pad))
    out = mask.copy()
    out[crop] = distance_transform_edt(~mask[crop], sampling=spacing) <= mm + 1e-6
    return out


def _grow(mask: np.ndarray, spacing: tuple, dilate_vox: int, mm: float | None) -> np.ndarray:
    return _dilate(mask, dilate_vox) if mm is None else dilate_mm(mask, spacing, mm)


def applied_mask_sitk(mask_sitk: sitk.Image, reference: sitk.Image, dilate_vox: int,
                      extend_anterior: bool, hu: np.ndarray | None = None,
                      dilate_by_mm: float | None = None) -> np.ndarray:
    """
    The region to de-face as a boolean array of `reference` (numpy axes z, y, x).

    The mask is resampled (nearest neighbour, in physical space) onto the reference grid
    when the grids differ; this is a no-op when they already match exactly. Then it is
    dilated (by `dilate_by_mm` millimetres, or by `dilate_vox` voxels if that is None)
    and, optionally, extended to the anterior image edge. With `hu` (the image as a
    numpy array), the extension is refused if the mask is not at the front of the head
    (check_face_anterior).
    """
    if not _same_grid(reference, mask_sitk):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_sitk = resampler.Execute(sitk.Cast(mask_sitk, sitk.sitkUInt8))
    mask = _grow(sitk.GetArrayFromImage(mask_sitk) > 0, reference.GetSpacing()[::-1], dilate_vox, dilate_by_mm)
    if extend_anterior:
        axis, increasing = sitk_axes(reference).anterior
        if hu is not None:
            check_face_anterior(hu, mask, axis, increasing)
        mask = extend_mask_anteriorly(mask, axis, increasing)
    return mask


def applied_mask_nifti(mask_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, dilate_vox: int,
                       extend_anterior: bool, hu: np.ndarray | None = None,
                       dilate_by_mm: float | None = None) -> np.ndarray:
    """applied_mask_sitk for nibabel images (numpy axes i, j, k of ref_img)."""
    if mask_img.shape[:3] == ref_img.shape[:3] and np.allclose(mask_img.affine, ref_img.affine):
        mask = np.asanyarray(mask_img.dataobj) > 0
    else:
        from scipy.ndimage import affine_transform
        # Maps a ref-image voxel index to the corresponding mask-image voxel index.
        transform = np.linalg.inv(mask_img.affine) @ ref_img.affine
        resampled = affine_transform(
            np.asanyarray(mask_img.dataobj).astype(np.float32), transform[:3, :3],
            offset=transform[:3, 3], output_shape=ref_img.shape[:3], order=0, mode="constant", cval=0,
        )
        mask = resampled > 0.5
    mask = _grow(mask, ref_img.header.get_zooms()[:3], dilate_vox, dilate_by_mm)
    if extend_anterior:
        axis, increasing = nifti_axes(ref_img).anterior
        if hu is not None:
            check_face_anterior(hu, mask, axis, increasing)
        mask = extend_mask_anteriorly(mask, axis, increasing)
    return mask


def mask_parts(path) -> list[tuple[float, float]]:
    """(volume in ml, gap in mm to the largest part) of the separate parts (26-connected) of a
    mask, largest first. The gap is measured between the bounding boxes of the parts."""
    from scipy.ndimage import find_objects, generate_binary_structure, label
    img = nib.load(str(path))
    zooms = np.array(img.header.get_zooms()[:3], float)
    lab, n = label(np.asanyarray(img.dataobj) > 0, structure=generate_binary_structure(3, 3))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())[1:]
    boxes = find_objects(lab)
    order = np.argsort(-sizes)
    main = boxes[order[0]]
    parts = []
    for k in order:
        gap = [max(0, b.start - m.stop + 1, m.start - b.stop + 1) * z for b, m, z in zip(boxes[k], main, zooms)]
        parts.append((float(sizes[k]) * float(np.prod(zooms)) / 1000.0, float(np.sqrt(np.sum(np.square(gap))))))
    return parts


def mask_volume_ml(path) -> float:
    """Volume of the (raw) TotalSegmentator face mask in millilitres."""
    img = nib.load(str(path))
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float((np.asanyarray(img.dataobj) > 0).sum()) * voxel_ml
