"""Face mask geometry: orientation, anterior extension, resampling, dilation, sanity check."""
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

import phantom as ph
from defacing.common import PatientError
from defacing.masks import (applied_mask_nifti, applied_mask_sitk, check_face_anterior, dilate_mm,
                            extend_mask_anteriorly, mask_parts, mask_volume_ml, nifti_axes, sitk_axes,
                            tissue_around_mask)


def sitk_image(orientation: str, shape=(8, 9, 10), spacing=(2.0, 3.0, 4.0)) -> sitk.Image:
    """sitk image (numpy shape z, y, x) with the direction of a DICOM orientation."""
    row, col = (np.array(v, float) for v in ph.ORIENTATIONS[orientation])
    img = sitk.Image(shape[2], shape[1], shape[0], sitk.sitkInt16)
    img.SetDirection(tuple(np.stack([row, col, np.cross(row, col)], axis=1).ravel()))
    img.SetSpacing(spacing)
    return img


def numpy_axis_world_direction(img: sitk.Image, numpy_axis: int) -> np.ndarray:
    d = np.array(img.GetDirection()).reshape(3, 3)
    return d[:, 2 - numpy_axis]           # numpy axis (z, y, x) -> sitk index (x, y, z)


@pytest.mark.parametrize("orientation", list(ph.ORIENTATIONS))
def test_sitk_axes_point_to_the_patient_front_and_head(orientation):
    img = sitk_image(orientation)
    axes = sitk_axes(img)
    axis, increasing = axes.anterior
    step = numpy_axis_world_direction(img, axis) * (1 if increasing else -1)
    assert step[1] < -0.7                 # LPS: anterior is -y
    assert abs(numpy_axis_world_direction(img, axes.si_axis)[2]) > 0.7
    assert axes.spacing == tuple(img.GetSpacing()[::-1])


@pytest.mark.parametrize("orientation", ["ras", "lps", "las"])
def test_nifti_axes_point_to_the_patient_front(orientation):
    affine = ph.nifti_affine(orientation)
    img = nib.Nifti1Image(np.zeros((56, 56, 36), np.int16), affine)
    axis, increasing = nifti_axes(img).anterior
    assert axis == 1
    assert increasing == (affine[1, 1] > 0)   # RAS: anterior is +y
    assert nifti_axes(img).si_axis == 2


@pytest.mark.parametrize("increasing", [True, False])
def test_extend_mask_anteriorly_fills_only_in_front_of_the_mask(increasing):
    mask = np.zeros((3, 10), bool)            # 3 lines along axis 1
    mask[0, 4] = True
    mask[1, [3, 6]] = True                    # line with two mask voxels
    out = extend_mask_anteriorly(mask, axis=1, increasing=increasing)
    if increasing:                            # anterior edge at index 9
        assert out[0].tolist() == [False] * 4 + [True] * 6
        assert out[1].tolist() == [False] * 3 + [True, False, False] + [True] * 4
    else:                                     # anterior edge at index 0
        assert out[0].tolist() == [True] * 5 + [False] * 5
        assert out[1].tolist() == [True] * 4 + [False, False, True] + [False] * 3
    assert not out[2].any()                   # a line without mask stays unchanged
    assert (out | mask).sum() == out.sum()    # the mask itself is kept


def test_extend_mask_anteriorly_3d_other_axis():
    mask = np.zeros((4, 5, 6), bool)
    mask[2, 1, 3] = True
    out = extend_mask_anteriorly(mask, axis=0, increasing=False)
    assert out[:3, 1, 3].all() and not out[3, 1, 3]
    assert out.sum() == 3


def head_volume(prone=False):
    """HU volume (z, y, x) with a head in the middle and its face towards -y (index 0)."""
    hu = np.full((10, 40, 30), -1000.0)
    hu[:, 10:35, 5:25] = 40.0                 # head
    if prone:
        hu[:, 2:4, :] = 300.0                 # couch in front of the face
    face = np.zeros(hu.shape, bool)
    face[3:7, 10:14, 10:20] = True
    return hu, face


def test_tissue_around_mask_counts_front_and_back():
    hu, face = head_volume()
    front, back = tissue_around_mask(hu, face, axis=1, increasing=False)
    assert front == 0
    assert back == 4 * (35 - 14) * 10
    assert tissue_around_mask(hu, face, axis=1, increasing=True) == (back, front)


@pytest.mark.parametrize("prone", [False, True])
def test_check_face_anterior_accepts_correct_orientation(prone):
    hu, face = head_volume(prone)
    check_face_anterior(hu, face, axis=1, increasing=False)


def test_check_face_anterior_rejects_wrong_orientation():
    hu, face = head_volume()
    with pytest.raises(PatientError, match="orientation"):
        check_face_anterior(hu, face, axis=1, increasing=True)


def test_check_face_anterior_ignores_empty_mask():
    hu, face = head_volume()
    check_face_anterior(hu, np.zeros_like(face), axis=1, increasing=True)


def test_applied_mask_sitk_same_grid_dilation_and_extension():
    ref = sitk_image("axial_hfs", shape=(6, 12, 12), spacing=(1.0, 1.0, 1.0))
    arr = np.zeros((6, 12, 12), np.uint8)
    arr[3, 5, 6] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ref)
    plain = applied_mask_sitk(mask, ref, 0, False)
    assert plain.sum() == 1 and plain[3, 5, 6]
    dilated = applied_mask_sitk(mask, ref, 1, False)
    assert dilated.sum() == 7                  # 6-connected dilation by one voxel
    extended = applied_mask_sitk(mask, ref, 0, True)
    assert extended[3, :6, 6].all() and extended.sum() == 6   # anterior = row index 0 (HFS)


def test_applied_mask_sitk_resamples_a_mask_on_another_grid():
    ref = sitk_image("axial_hfs", shape=(8, 16, 16), spacing=(1.0, 1.0, 1.0))
    coarse = np.zeros((4, 8, 8), np.uint8)
    coarse[1, 2, 3] = 1                        # covers ref voxels z 2-3, y 4-5, x 6-7
    mask = sitk.GetImageFromArray(coarse)
    mask.SetSpacing((2.0, 2.0, 2.0))
    mask.SetOrigin(tuple(np.array(ref.GetOrigin()) + 0.5))
    mask.SetDirection(ref.GetDirection())
    out = applied_mask_sitk(mask, ref, 0, False)
    assert out.shape == (8, 16, 16)
    assert set(map(tuple, np.argwhere(out))) == {(z, y, x) for z in (2, 3) for y in (4, 5) for x in (6, 7)}


def test_applied_mask_sitk_refuses_wrong_orientation():
    hu, face = head_volume()
    ref = sitk_image("axial_hfs", shape=hu.shape, spacing=(1.0, 1.0, 1.0))
    mask = sitk.GetImageFromArray(face.astype(np.uint8))
    mask.CopyInformation(ref)
    applied_mask_sitk(mask, ref, 0, True, hu=hu)               # face at the front: fine
    back = sitk.GetImageFromArray(np.ascontiguousarray(face[:, ::-1]).astype(np.uint8))
    back.CopyInformation(ref)
    with pytest.raises(PatientError):
        applied_mask_sitk(back, ref, 0, True, hu=hu)           # "face" at the back of the head


def test_applied_mask_nifti_resamples_by_affine():
    ref = nib.Nifti1Image(np.zeros((10, 10, 10), np.int16), np.eye(4))
    flipped_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    flipped_affine[0, 3] = 9.0                 # same grid, x index reversed
    arr = np.zeros((10, 10, 10), np.uint8)
    arr[2, 3, 4] = 1
    out = applied_mask_nifti(nib.Nifti1Image(arr, flipped_affine), ref, 0, False)
    assert np.argwhere(out).tolist() == [[7, 3, 4]]


def test_dilate_mm_is_the_same_in_every_direction():
    mask = np.zeros((11, 21, 21), bool)
    mask[5, 10, 10] = True
    out = dilate_mm(mask, (3.0, 1.0, 1.0), 3.0)             # 3 mm slices, 1 mm pixels
    assert out[4:7, 10, 10].all() and not out[3, 10, 10]     # one slice up and down
    assert out[5, 10, 7:14].all() and not out[5, 10, 6]      # three pixels sideways
    assert out[5, 12, 12] and not out[5, 13, 12]             # sqrt(8) <= 3 < sqrt(10)
    assert not out[4, 11, 10]                                # sqrt(9 + 1) > 3
    assert np.array_equal(dilate_mm(mask, (3.0, 1.0, 1.0), 0.0), mask)
    assert dilate_mm(mask, (3.0, 1.0, 1.0), 2.9)[5, 10, 10] and not dilate_mm(mask, (3.0, 1.0, 1.0), 2.9)[4, 10, 10]


def test_applied_mask_dilates_by_mm_or_by_voxels():
    ref = sitk_image("axial_hfs", shape=(9, 12, 12), spacing=(1.0, 1.0, 4.0))
    arr = np.zeros((9, 12, 12), np.uint8)
    arr[4, 6, 6] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ref)
    by_mm = applied_mask_sitk(mask, ref, 1, False, dilate_by_mm=2.0)
    assert by_mm[4].sum() == 13 and by_mm[3].sum() == 0      # 4 mm slices are further away than 2 mm
    by_vox = applied_mask_sitk(mask, ref, 1, False)
    assert by_vox.sum() == 7 and by_vox[3, 6, 6]


def test_mask_parts(tmp_path):
    arr = np.zeros((40, 20, 20), np.uint8)
    arr[2:12, 2:12, 2:12] = 1                                 # main part
    arr[13, 5, 5] = 1                                         # fragment right next to it
    arr[30:33, 2:5, 2:5] = 1                                  # part 19 slices (38 mm) away
    nib.save(nib.Nifti1Image(arr, np.diag([2.0, 1.0, 1.0, 1.0])), str(tmp_path / "m.nii.gz"))
    parts = mask_parts(tmp_path / "m.nii.gz")
    assert [round(ml * 1000) for ml, _ in parts] == [2000, 54, 2]
    assert parts[0][1] == 0 and parts[1][1] == pytest.approx(38.0) and parts[2][1] == pytest.approx(4.0)


def test_mask_volume_ml(tmp_path):
    arr = np.zeros((10, 10, 10), np.uint8)
    arr[:5, :4, :2] = 1                        # 40 voxels of 2 x 2 x 2.5 mm = 10 mm^3 each
    nib.save(nib.Nifti1Image(arr, np.diag([2.0, 2.0, 2.5, 1.0])), str(tmp_path / "m.nii.gz"))
    assert mask_volume_ml(tmp_path / "m.nii.gz") == pytest.approx(0.4)
