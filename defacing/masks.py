"""Face mask: alignment to the image grid, dilation and extension to the anterior image edge."""
from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import SimpleITK as sitk


@dataclass
class Axes:
    """Orientation of a numpy volume: voxel size per numpy axis (mm), the numpy axis that
    points most nearly to the patient's anterior (and whether the index increases that
    way), and the numpy axis closest to the head-feet direction."""
    spacing: tuple[float, float, float]
    anterior: tuple[int, bool]
    si_axis: int


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
    s_axis, _ = _axis_towards(direction, np.array([0.0, 0.0, 1.0]))        # superior = +z
    return Axes(spacing=tuple(float(s) for s in img.GetSpacing()[::-1]),
                anterior=(2 - a_axis, a_inc), si_axis=2 - s_axis)


def nifti_axes(img: nib.Nifti1Image) -> Axes:
    """Axes of a nibabel image's data array: numpy axes are (i, j, k), world space is RAS."""
    affine = img.affine[:3, :3]
    a_axis, a_inc = _axis_towards(affine, np.array([0.0, 1.0, 0.0]))       # anterior = +y
    s_axis, _ = _axis_towards(affine, np.array([0.0, 0.0, 1.0]))           # superior = +z
    return Axes(spacing=tuple(float(z) for z in img.header.get_zooms()[:3]),
                anterior=(a_axis, a_inc), si_axis=s_axis)


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


def applied_mask_sitk(mask_sitk: sitk.Image, reference: sitk.Image, dilate_vox: int,
                      extend_anterior: bool) -> np.ndarray:
    """
    The region to de-face as a boolean array of `reference` (numpy axes z, y, x).

    The mask is resampled (nearest neighbour, in physical space) onto the reference grid
    when the grids differ; this is a no-op when they already match exactly. Then it is
    dilated and, optionally, extended to the anterior image edge.
    """
    if not _same_grid(reference, mask_sitk):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_sitk = resampler.Execute(sitk.Cast(mask_sitk, sitk.sitkUInt8))
    mask = _dilate(sitk.GetArrayFromImage(mask_sitk) > 0, dilate_vox)
    if extend_anterior:
        axis, increasing = sitk_axes(reference).anterior
        mask = extend_mask_anteriorly(mask, axis, increasing)
    return mask


def applied_mask_nifti(mask_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, dilate_vox: int,
                       extend_anterior: bool) -> np.ndarray:
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
    mask = _dilate(mask, dilate_vox)
    if extend_anterior:
        axis, increasing = nifti_axes(ref_img).anterior
        mask = extend_mask_anteriorly(mask, axis, increasing)
    return mask


def mask_volume_ml(path) -> float:
    """Volume of the (raw) TotalSegmentator face mask in millilitres."""
    img = nib.load(str(path))
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float((np.asanyarray(img.dataobj) > 0).sum()) * voxel_ml
