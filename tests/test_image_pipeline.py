"""End-to-end de-facing of NIfTI and NRRD images (TotalSegmentator replaced)."""
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

import phantom as ph
from conftest import tree_hashes
from defacing.common import Options, PatientError
from defacing.patient import deface_image_file


def check(data, affine, out_data):
    world = ph.nifti_world_lps(affine, data.shape)
    behind = world[..., 1] > -40
    assert np.allclose(out_data[behind], data[behind], atol=0.5 + 1e-6)
    assert (out_data[ph.in_face(world)] < -850).all()
    assert (out_data[ph.in_nose(world)] < -850).all()


@pytest.mark.parametrize("orientation", ["ras", "lps", "las"])
@pytest.mark.parametrize("name", ["p01_CT.nii.gz", "p01_CT.nii"])
def test_nifti_orientations(tmp_path, fake_ts, orientation, name):
    src = tmp_path / name
    data, affine = ph.write_nifti(src, orientation)
    before = tree_hashes(tmp_path)
    res = deface_image_file(src, "p01_CT", tmp_path / "out", tmp_path / "qc", Options())
    assert res.status == "done" and tree_hashes(tmp_path / "out") and tree_hashes(tmp_path).items() >= before.items()
    out = nib.load(str(tmp_path / "out" / "p01_CT_deFaced.nii.gz"))
    assert np.allclose(out.affine, affine) and out.get_data_dtype() == np.int16
    check(data, affine, out.get_fdata())
    assert all((tmp_path / "qc" / n).exists() for n in ("face_mask.nii.gz", "manifest.json", "QC_defacing.png"))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["p01_CT_deFaced.nii.gz"]


@pytest.mark.parametrize("dtype, slope, inter", [
    (np.uint16, 1.0, -1024.0),          # dcm2niix output of unsigned CT data
    (np.int16, 0.5, -1024.0),
    (np.float32, None, None),
    (np.int32, None, None),
])
def test_nifti_data_types_and_scaling_are_kept(tmp_path, fake_ts, dtype, slope, inter):
    src = tmp_path / "ct.nii.gz"
    data, affine = ph.write_nifti(src, dtype=dtype, slope=slope, inter=inter)
    deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options())
    out = nib.load(str(tmp_path / "out" / "ct_deFaced.nii.gz"))
    assert out.get_data_dtype() == dtype
    if slope is not None:
        assert (out.dataobj.slope, out.dataobj.inter) == (slope, inter)
    check(data, affine, out.get_fdata())


def test_nifti_with_trailing_singleton_dimension(tmp_path, fake_ts):
    src = tmp_path / "ct.nii.gz"
    data, affine = ph.write_nifti(src)
    nib.save(nib.Nifti1Image(np.round(data).astype(np.int16)[..., None], affine), str(src))
    deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options())
    out = nib.load(str(tmp_path / "out" / "ct_deFaced.nii.gz"))
    assert out.shape == data.shape + (1,)
    check(data, affine, out.get_fdata()[..., 0])


def test_4d_image_is_refused_before_segmentation(tmp_path, fake_ts):
    src = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 2), np.int16), np.eye(4)), str(src))
    with pytest.raises(PatientError, match="not a 3D"):
        deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options())
    assert not fake_ts.calls and not (tmp_path / "out").exists()


@pytest.mark.parametrize("fill", ["air", "soft-tissue", "constant"])
def test_nrrd(tmp_path, fake_ts, fill):
    _, affine = ph.write_nifti(tmp_path / "tmp.nii.gz", "lps")
    src = tmp_path / "p02_IMG_CT.nrrd"
    img = sitk.ReadImage(str(tmp_path / "tmp.nii.gz"))
    img.SetMetaData("PatientName", "Mustermann")
    sitk.WriteImage(img, str(src))
    data = sitk.GetArrayFromImage(sitk.ReadImage(str(src))).astype(float)
    res = deface_image_file(src, "p02", tmp_path / "out", tmp_path / "qc", Options(fill=fill))
    assert res.pii == ["NRRD key 'PatientName'"]
    out_img = sitk.ReadImage(str(tmp_path / "out" / "p02_IMG_CT_deFaced.nrrd"))
    assert out_img.GetDirection() == img.GetDirection() and out_img.GetOrigin() == img.GetOrigin()
    assert out_img.GetPixelID() == img.GetPixelID()
    out = sitk.GetArrayFromImage(out_img).astype(float)
    # numpy (z, y, x) of the sitk array = nifti (i, j, k) transposed
    world = ph.nifti_world_lps(affine, data.shape[::-1]).transpose(2, 1, 0, 3)
    behind = world[..., 1] > -40
    assert np.array_equal(out[behind], data[behind])
    assert (out[ph.in_nose(world) & (world[..., 1] < -100)] < -850).all()
    assert "PatientName" not in out_img.GetMetaDataKeys()


def test_wrong_orientation_in_the_header_is_refused(tmp_path, fake_ts):
    src = tmp_path / "ct.nii.gz"
    data, affine = ph.write_nifti(src)
    fake_ts.mode = "back"
    with pytest.raises(PatientError, match="orientation"):
        deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options())
    assert not (tmp_path / "out").exists()


def test_empty_mask(tmp_path, fake_ts):
    fake_ts.mode = "empty"
    src = tmp_path / "head.nii.gz"
    ph.write_nifti(src)
    with pytest.raises(PatientError, match="EMPTY"):
        deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options())
    with pytest.raises(PatientError, match="includes the head"):
        deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options(allow_empty_mask=True))
    chest = tmp_path / "chest.nii.gz"
    data, _ = ph.write_nifti(chest, centre_z=-400.0)
    res = deface_image_file(chest, "ct", tmp_path / "out", tmp_path / "qc", Options(allow_empty_mask=True))
    assert any("EMPTY" in f and "no head" in f for f in res.flags)
    assert np.allclose(nib.load(str(tmp_path / "out" / "chest_deFaced.nii.gz")).get_fdata(), data)


def test_unknown_voxel_size_nifti_and_nrrd(tmp_path, fake_ts, monkeypatch):
    """No orientation and unit voxels (NIfTI) or no 'space' field (NRRD): dilation in voxels, flagged."""
    import defacing.segment
    monkeypatch.setattr(defacing.segment, "head_volume_ml", lambda *a: 0.0)   # accept the empty masks
    src = tmp_path / "ct.nii.gz"
    data, _ = ph.write_nifti(src)
    nib.save(nib.Nifti1Image(np.round(data).astype(np.int16), None), str(src))   # no orientation, 1 mm
    assert nib.load(str(src)).header["sform_code"] == 0
    fake_ts.mode = "empty"
    res = deface_image_file(src, "ct", tmp_path / "o1", tmp_path / "q1", Options(allow_empty_mask=True))
    assert any("voxel size unknown" in f for f in res.flags) and res.fill[0]["dilate_mm"] is None
    ok = tmp_path / "ok.nii.gz"
    ph.write_nifti(ok, centre_z=-400.0)
    res = deface_image_file(ok, "ok", tmp_path / "o2", tmp_path / "q2", Options(allow_empty_mask=True))
    assert not any("voxel size unknown" in f for f in res.flags) and res.fill[0]["dilate_mm"] == 3.0
    nrrd = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(np.full((8, 8, 8), -1000, np.int16)), str(nrrd))
    text = nrrd.read_bytes()
    assert b"space" in text                         # SimpleITK always writes a space: remove it
    head, _, body = text.partition(b"\n\n")
    lines = [ln for ln in head.split(b"\n") if not ln.startswith((b"space", b"space directions", b"space origin"))]
    nrrd.write_bytes(b"\n".join(lines) + b"\n\n" + body)
    res = deface_image_file(nrrd, "n", tmp_path / "o3", tmp_path / "q3", Options(allow_empty_mask=True))
    assert any("voxel size unknown" in f for f in res.flags)


def test_dry_run(tmp_path, fake_ts):
    src = tmp_path / "ct.nii.gz"
    ph.write_nifti(src, descrip="Mustermann")
    res = deface_image_file(src, "ct", tmp_path / "out", tmp_path / "qc", Options(), dry_run=True)
    assert res.status == "dry run" and res.pii == ["NIfTI header 'descrip' (free text)"]
    assert not fake_ts.calls and not (tmp_path / "out").exists() and not (tmp_path / "qc").exists()
