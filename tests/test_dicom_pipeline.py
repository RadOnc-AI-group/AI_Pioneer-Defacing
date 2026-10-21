"""
End-to-end de-facing of DICOM patients (TotalSegmentator replaced by the phantom's face
region). The output is read back with pydicom only and checked in patient coordinates,
so the checks do not depend on the reader, the slice order or the orientation.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import phantom as ph
from conftest import tree_hashes
from defacing.common import Options, PatientError
from defacing.ct import verify_written
from defacing.patient import deface_dicom_patient
from defacing.rt import roi_table, rtstruct_image_refs

BEHIND_FACE_Y = -40.0      # LPS y: voxels behind this are never part of the de-faced region


def run(tmp_path: Path, opts: Options | None = None, rt: bool = False, **ct_kwargs):
    """Write a patient, de-face it; returns (result, input CT, out dir, qc dir)."""
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT", **ct_kwargs)
    if rt:
        ph.write_rtstruct(pat / "RS.PAT01.dcm", ct, [("BODY", "EXTERNAL"), ("PTV_60", "PTV"), ("Eye_L", "ORGAN"),
                                                    ("Skin", "ORGAN"), ("Body-PTV", "PTV")])
    out, qc = tmp_path / "out" / "PAT01", tmp_path / "qc" / "PAT01"
    res = deface_dicom_patient(pat, "PAT01", out, qc, opts or Options())
    return res, ct, out, qc


def check_defaced(ct, out: Path, remove_shell=True, fill="air"):
    hu, world, dss = ph.read_series_hu(out / "dcm" if (out / "dcm").is_dir() else out)
    assert hu.shape == ct.hu.shape
    behind = world[..., 1] > BEHIND_FACE_Y
    assert np.array_equal(hu[behind], ct.hu[behind]), "voxels behind the face changed"
    face = ph.in_face(world)
    assert face.any()
    if fill in ("air", "constant"):
        assert (hu[face] < -850).all(), "face voxels not replaced by air"
    nose = ph.in_nose(world) & (world[..., 1] < -100)
    assert (hu[nose] < -850).all(), "the nose tip is still there"
    shell = ph.in_shell(world) & (np.abs(world[..., 0]) < 30) & (world[..., 2] > -15) & (world[..., 2] < 5)
    if remove_shell and shell.any():
        assert (hu[shell] < -850).all(), "the immobilisation mask in front of the face is still there"
    return hu, world, dss


@pytest.mark.parametrize("orientation", list(ph.ORIENTATIONS))
def test_every_orientation_is_defaced_at_the_front(tmp_path, fake_ts, orientation):
    res, ct, out, _ = run(tmp_path, orientation=orientation, shell=True)
    assert res.status == "done" and res.face_mask_ml[0] > 0
    check_defaced(ct, out)


def test_prone_scan_with_the_couch_in_front(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, orientation="axial_prone", prone=True)
    assert res.status == "done"
    hu, world, _ = check_defaced(ct, out)
    couch_in_front_of_face = ph.in_box(world, ((-20, 20), (-112, -106), (-10, 10)))
    assert (hu[couch_in_front_of_face] < -850).all()          # removed as well (README)


@pytest.mark.parametrize("kwargs", [
    dict(descending=True), dict(names="shuffled"), dict(names="plain"),
    dict(signed=False, bits_stored=12, intercept=-1024.0),
    dict(signed=False, intercept=-1024.0, slope=0.5),
    dict(compress="rle"),
    dict(spacing=(2.25, 2.25, 6.0), shape=(36, 112, 112)),       # finer grid
    dict(spacing=(4.5, 4.5, 3.0), shape=(72, 56, 56)),           # thinner slices
], ids=["descending", "shuffled-names", "no-extension", "uint12", "slope0.5", "rle", "fine", "thin"])
def test_slice_orders_names_and_pixel_encodings(tmp_path, fake_ts, kwargs):
    res, ct, out, _ = run(tmp_path, **kwargs)
    assert res.status == "done"
    _, _, dss = check_defaced(ct, out)
    for ds in dss:
        assert not ds.file_meta.TransferSyntaxUID.is_compressed
        assert ds.PixelRepresentation == (0 if kwargs.get("signed") is False else 1)
        assert float(ds.RescaleIntercept) == kwargs.get("intercept", 0.0)


def test_gantry_tilt(tmp_path, monkeypatch):
    """Tilted slices (shifted along the column direction) are read as an untilted stack;
    TotalSegmentator segments the image content, so the stand-in marks the face at the
    true slice positions."""
    import defacing.ct

    def content_based_ts(input_nii, seg_dir, device, log_path):
        img = nib.load(str(input_nii))
        world = ph.nifti_world_lps(img.affine, img.shape)
        k = np.arange(img.shape[2])[None, None, :]
        true = world + (k * 1.0)[..., None] * np.array([0.0, 1.0, 0.0])
        out = Path(seg_dir) / "face.nii.gz"
        out.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(ph.in_face(true).astype(np.uint8), img.affine), str(out))
        return out

    monkeypatch.setattr(defacing.ct, "run_totalsegmentator", content_based_ts)
    res, ct, out, _ = run(tmp_path, tilt=1.0)
    assert res.status == "done"
    check_defaced(ct, out)


@pytest.mark.parametrize("fill", ["constant", "soft-tissue"])
def test_other_fills(tmp_path, fake_ts, fill):
    res, ct, out, _ = run(tmp_path, Options(fill=fill, dilate_mm=9.0), shell=True)
    hu, world, _ = check_defaced(ct, out, fill=fill)
    face = ph.in_face(world)
    if fill == "constant":
        assert (hu[face] == -1000).all()
    else:
        assert res.fill[0]["voxels_tissue"] > 0 and abs(res.fill[0]["tissue_hu"] - ph.TISSUE_HU) < 10
        vals = hu[hu != ct.hu]                                  # every replaced voxel: air or soft tissue
        assert ((vals < -850) | ((vals > -200) & (vals < 200))).all()


def test_padding_outside_the_field_of_view_is_kept(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, fov_radius=110.0)
    hu, world, _ = check_defaced(ct, out)
    padding = ct.hu < -1100
    assert padding.any() and np.array_equal(hu[padding], ct.hu[padding])


def test_no_extend_anterior_keeps_what_lies_in_front_of_the_face(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, Options(extend_anterior=False), shell=True)
    hu, world, _ = check_defaced(ct, out, remove_shell=False)
    assert (hu[ph.in_shell(world)] > 100).all()


def test_dilation_in_mm(tmp_path, fake_ts):
    counts = {}
    for mm in (0.0, 3.0, 5.0, 10.0):
        res, _, _, qc = run(tmp_path / str(mm), Options(dilate_mm=mm, extend_anterior=False))
        counts[mm] = int((nib.load(str(qc / "CT_defaced.nii.gz")).get_fdata() < -850).sum())
        assert res.fill[0]["dilate_mm"] == mm
    assert counts[0.0] == counts[3.0]          # 3 mm < the 4.5 mm voxels: no neighbour that close
    assert counts[3.0] < counts[5.0] < counts[10.0]


def test_unknown_voxel_size_falls_back_to_voxels(tmp_path, monkeypatch):
    """Without PixelSpacing, the margin is given in voxels and the patient is flagged."""
    import defacing.ct

    def box_ts(input_nii, seg_dir, device, log_path):    # a face box in voxel space
        img = nib.load(str(input_nii))
        mask = np.zeros(img.shape, np.uint8)
        mask[20:36, 2:8, 10:20] = 1                        # (x, y = rows from the front, z)
        out = Path(seg_dir) / "face.nii.gz"
        out.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(mask, img.affine), str(out))
        return out

    monkeypatch.setattr(defacing.ct, "run_totalsegmentator", box_ts)
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    for p in ct.paths:
        ds = pydicom.dcmread(p)
        del ds.PixelSpacing
        ds.save_as(p)
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options(dilate=1))
    assert any("voxel size unknown" in f for f in res.flags)
    assert res.fill[0]["dilate_mm"] is None
    method = list(pydicom.dcmread(next((tmp_path / "out" / "dcm").iterdir())).DeidentificationMethod)
    assert "dilate=1vox" in method[1]


def test_pixel_encoding_that_cannot_store_air_fails_without_output(tmp_path, fake_ts):
    for fill in ("air", "constant"):
        with pytest.raises(PatientError, match="cannot store"):
            run(tmp_path / fill, Options(fill=fill), signed=False, intercept=0.0)
        assert not (tmp_path / fill / "out" / "PAT01").exists()
        assert not any((tmp_path / fill).rglob("*.partial"))


def test_input_is_never_modified(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT", compress="rle")
    ph.write_rtstruct(pat / "RS.dcm", ct)
    before = tree_hashes(pat)
    deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options())
    assert tree_hashes(pat) == before


def test_new_uids_and_deidentification_marks(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path)
    _, _, dss = ph.read_series_hu(out / "dcm")
    sops = {ds.SOPInstanceUID for ds in dss}
    assert len(sops) == len(dss) and not sops & set(ct.sop_uids)
    assert all(ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID for ds in dss)
    assert len({ds.SeriesInstanceUID for ds in dss}) == 1 and dss[0].SeriesInstanceUID != ct.series_uid
    assert all(ds.StudyInstanceUID == ct.study_uid and ds.FrameOfReferenceUID == ct.frame_uid for ds in dss)
    ds = dss[0]
    method = list(ds.DeidentificationMethod)
    assert method[0].startswith("De-faced") and "fill=air seed=" in method[-1]
    assert ds.DeidentificationMethodCodeSequence[-1].CodeValue == "113102"
    assert "de-faced" in str(ds.ImageComments).lower()
    assert all(len(v) <= 64 for v in method)                    # VR LO


def test_input_file_names_are_kept_and_icons_removed(tmp_path, fake_ts):
    """File names stay as in the input (the hospital can trace a file back to its patient); only
    an old SOP Instance UID in a name is replaced by the new one."""
    pat = tmp_path / "in" / "PAT01"
    icon = Dataset()
    icon.Rows = icon.Columns = 8
    icon.SamplesPerPixel, icon.PhotometricInterpretation = 1, "MONOCHROME2"
    icon.BitsAllocated = icon.BitsStored = 8
    icon.HighBit, icon.PixelRepresentation, icon.PixelData = 7, 0, bytes(64)
    ct = ph.write_ct_series(pat / "CT", names="plain", extra={"IconImageSequence": Sequence([icon])})
    ph.write_rtstruct(pat / "RS_Mustermann.dcm", ct)
    deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options())
    names = sorted(p.name for p in (tmp_path / "out" / "dcm").iterdir())
    assert names == sorted(p.name for p in ct.paths)                 # IM0000, IM0001, ...
    assert [p.name for p in (tmp_path / "out" / "rt").iterdir()] == ["RS_Mustermann.dcm"]
    assert all("IconImageSequence" not in pydicom.dcmread(p) for p in (tmp_path / "out" / "dcm").iterdir())

    pat2 = tmp_path / "in2" / "PAT01"                                   # CT.<uid>.dcm -> CT.<new uid>.dcm
    ct2 = ph.write_ct_series(pat2 / "CT", names="uid")
    deface_dicom_patient(pat2, "PAT01", tmp_path / "out2", tmp_path / "qc2", Options())
    for p in (tmp_path / "out2" / "dcm").iterdir():
        assert p.name == f"CT.{pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID}.dcm"
    assert not {p.name for p in (tmp_path / "out2" / "dcm").iterdir()} & {p.name for p in ct2.paths}


def test_same_file_name_in_two_folders(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT", names="plain")
    for i, p in enumerate(ct.paths):                                     # a/IM0000 and b/IM0000 ...
        (pat / ("a" if i % 2 else "b")).mkdir(exist_ok=True)
        p.rename(pat / ("a" if i % 2 else "b") / f"IM{i // 2:04d}")
    deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options())
    assert len(list((tmp_path / "out" / "dcm").iterdir())) == ct.shape[0]


def test_output_and_qc_folders(tmp_path, fake_ts):
    res, ct, out, qc = run(tmp_path, rt=True)
    assert sorted(p.name for p in out.iterdir()) == ["dcm", "rt"]
    assert not list(out.rglob("*.nii*"))                         # no face mask in the shareable output
    assert {p.name for p in qc.iterdir()} >= {"face_mask.nii.gz", "CT_defaced.nii.gz", "manifest.json",
                                               "roi_overlap.csv", "removed_rois", "tool.log"}
    import json
    manifest = json.loads((qc / "manifest.json").read_text())
    assert manifest["fill"] == "air" and manifest["ct_files"] == ct.shape[0]
    assert manifest["rois_removed"] == ["BODY", "Skin"]
    mask = nib.load(str(qc / "face_mask.nii.gz"))
    assert mask.get_fdata().sum() > 0


def test_rtstruct_body_and_skin_removed_and_relinked(tmp_path, fake_ts):
    res, ct, out, qc = run(tmp_path, rt=True)
    assert res.rois_removed == ["BODY", "Skin"] and res.rois_unclear == ["Body-PTV"]
    rs = [pydicom.dcmread(p) for p in (out / "rt").iterdir()]
    assert len(rs) == 1
    rs = rs[0]
    names = {name for name, _ in roi_table(rs).values()}
    assert names == {"PTV_60", "Eye_L", "Body-PTV"}
    new_ct = {pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID for p in (out / "dcm").iterdir()}
    assert rtstruct_image_refs(rs) and rtstruct_image_refs(rs) <= new_ct
    new_series = pydicom.dcmread(next((out / "dcm").iterdir()), stop_before_pixels=True).SeriesInstanceUID
    series_refs = {s.SeriesInstanceUID for f in rs.ReferencedFrameOfReferenceSequence
                   for st in f.RTReferencedStudySequence for s in st.RTReferencedSeriesSequence}
    assert series_refs == {new_series}
    # removed contours: in the QC folder only, linked to the de-faced CT
    removed = [pydicom.dcmread(p) for p in (qc / "removed_rois").iterdir()]
    assert len(removed) == 1 and {n for n, _ in roi_table(removed[0]).values()} == {"BODY", "Skin"}
    assert rtstruct_image_refs(removed[0]) <= new_ct
    # overlap report: the eye lies in the de-faced region, the PTV does not
    rows = {r["roi"]: r for r in res.overlap}
    assert rows["Eye_L"]["pct_in_defaced_region"] > 50
    assert rows["PTV_60"]["pct_in_defaced_region"] == 0


def test_rtdose_and_rtplan_removed_unless_kept(tmp_path, fake_ts):
    for keep in (False, True):
        pat = tmp_path / str(keep) / "PAT01"
        ct = ph.write_ct_series(pat / "CT")
        rs = ph.write_rtstruct(pat / "RS.dcm", ct)
        plan_ref = Dataset()
        plan_ref.ReferencedSOPClassUID, plan_ref.ReferencedSOPInstanceUID = ph.RTSTRUCT_STORAGE, rs.SOPInstanceUID
        plan = ph.write_simple(pat / "RP.dcm", "RTPLAN", ph.RTPLAN_STORAGE, ct,
                               ReferencedStructureSetSequence=Sequence([plan_ref]))
        dose_ref = Dataset()
        dose_ref.ReferencedSOPClassUID, dose_ref.ReferencedSOPInstanceUID = ph.RTPLAN_STORAGE, plan.SOPInstanceUID
        ph.write_simple(pat / "RD.dcm", "RTDOSE", ph.RTDOSE_STORAGE, ct, ReferencedRTPlanSequence=Sequence([dose_ref]))
        out = tmp_path / str(keep) / "out"
        res = deface_dicom_patient(pat, "PAT01", out, tmp_path / str(keep) / "qc",
                                   Options(keep_rtdose=keep, keep_rtplan=keep))
        mods = sorted(pydicom.dcmread(p).Modality for p in (out / "rt").iterdir())
        if not keep:
            assert mods == ["RTSTRUCT"] and res.rtdose_removed == 1 and res.rtplan_removed == 1
            continue
        assert mods == ["RTDOSE", "RTPLAN", "RTSTRUCT"]
        objs = {pydicom.dcmread(p).Modality: pydicom.dcmread(p) for p in (out / "rt").iterdir()}
        # every reference between the copied objects points to the new UIDs
        assert objs["RTPLAN"].ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID == objs["RTSTRUCT"].SOPInstanceUID
        assert objs["RTDOSE"].ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == objs["RTPLAN"].SOPInstanceUID
        assert objs["RTPLAN"].SOPInstanceUID != plan.SOPInstanceUID
        assert any("RTDOSE kept" in f for f in res.flags)


def test_several_ct_series(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    a = ph.write_ct_series(pat / "CT1")
    b = ph.write_ct_series(pat / "CT2", orientation="sagittal", study_uid=a.study_uid, frame_uid=a.frame_uid,
                           shape=(30, 56, 56))
    out, qc = tmp_path / "out", tmp_path / "qc"
    res = deface_dicom_patient(pat, "PAT01", out, qc, Options())
    assert res.ct_series == 2 and len(res.face_mask_ml) == 2 and len(fake_ts.calls) == 2
    assert sorted(p.name for p in (out / "dcm").iterdir()) == ["series_1", "series_2"]
    by_size = {len(list((out / "dcm" / s).iterdir())): s for s in ("series_1", "series_2")}
    check_defaced(a, out / "dcm" / by_size[36])
    check_defaced(b, out / "dcm" / by_size[30])
    assert (qc / "face_mask_series1.nii.gz").exists() and (qc / "CT_defaced_series2.nii.gz").exists()


def test_empty_face_mask_fails(tmp_path, fake_ts):
    fake_ts.mode = "empty"
    with pytest.raises(PatientError, match="EMPTY"):
        run(tmp_path)
    assert not (tmp_path / "out" / "PAT01").exists() and not fake_ts.head_calls


def test_empty_face_mask_in_a_scan_with_the_head_fails_even_when_allowed(tmp_path, fake_ts):
    fake_ts.mode = "empty"
    with pytest.raises(PatientError, match="includes the head"):
        run(tmp_path, Options(allow_empty_mask=True))
    assert fake_ts.head_calls and not (tmp_path / "out" / "PAT01").exists()


def test_empty_face_mask_accepted_per_series_without_head(tmp_path, fake_ts):
    """Head CT and chest CT of one patient: the chest CT has no face and no head."""
    pat = tmp_path / "in" / "PAT01"
    head = ph.write_ct_series(pat / "CT_head")
    chest = ph.write_ct_series(pat / "CT_chest", centre_z=-400.0, shape=(30, 56, 56), study_uid=head.study_uid)
    out = tmp_path / "out"
    with pytest.raises(PatientError, match="EMPTY"):
        deface_dicom_patient(pat, "PAT01", out, tmp_path / "qc", Options())
    res = deface_dicom_patient(pat, "PAT01", out, tmp_path / "qc", Options(allow_empty_mask=True))
    assert res.status == "done" and len(fake_ts.head_calls) == 1          # only for the series without a face
    assert [f for f in res.flags if "EMPTY" in f and "no head in the scan" in f]
    by_size = {len(list((out / "dcm" / s).iterdir())): s for s in ("series_1", "series_2")}
    check_defaced(head, out / "dcm" / by_size[36])
    hu, _, _ = ph.read_series_hu(out / "dcm" / by_size[30])
    assert np.array_equal(hu, chest.hu)


def test_qc_picture(tmp_path, fake_ts):
    import SimpleITK as sitk
    _, _, _, qc = run(tmp_path, shell=True)
    picture = sitk.GetArrayFromImage(sitk.ReadImage(str(qc / "QC_defacing.png")))
    assert picture.dtype == np.uint8 and picture.ndim == 2
    h = picture.shape[0]
    top, bottom = picture[: h // 2 - 2].astype(int), picture[h - (h // 2 - 2):].astype(int)
    assert (top != bottom).any() and top.max() > 200
    assert not (tmp_path / "out" / "PAT01" / "QC_defacing.png").exists()


def test_automatic_checks(tmp_path, fake_ts):
    res, _, _, _ = run(tmp_path / "clean")
    assert not [f for f in res.flags if "separate parts" in f or "burned-in" in f]
    res, _, _, _ = run(tmp_path / "text", text=True)
    assert [f for f in res.flags if "burned-in text" in f]
    fake_ts.mode = "two_parts"
    res, _, _, _ = run(tmp_path / "parts")
    assert [f for f in res.flags if "1 separate part(s) away from the face" in f]


def test_face_mask_at_the_back_of_the_head_is_refused(tmp_path, fake_ts):
    """Wrong orientation in the header: the 'anterior' extension would erase the head."""
    fake_ts.mode = "back"
    with pytest.raises(PatientError, match="orientation"):
        run(tmp_path)
    assert not (tmp_path / "out" / "PAT01").exists()
    fake_ts.mode = "back"
    res, _, _, _ = run(tmp_path / "noext", Options(extend_anterior=False))   # nothing extended: allowed
    assert res.status == "done"


def test_verification_detects_changed_voxels(tmp_path):
    """verify_written compares the files read back independently with what was intended."""
    ct = ph.write_ct_series(tmp_path / "CT", shape=(6, 24, 24), spacing=(10.0, 10.0, 30.0))
    image = np.stack([pydicom.dcmread(f).pixel_array.astype(float) for f in ct.paths])
    region = np.zeros(image.shape, bool)
    region[2, :5, :] = True
    target = image.copy()
    target[region] = -1000.0

    def write(changes):
        folder = tmp_path / f"out{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        for i, f in enumerate(ct.paths):
            d = pydicom.dcmread(f)
            arr = d.pixel_array.copy()
            arr[region[i]] = -1000
            for (k, r, c), v in changes.items():
                if k == i:
                    arr[r, c] = v
            d.PixelData = arr.tobytes()
            d.save_as(folder / f.name)
        return [folder / f.name for f in ct.paths]

    verify_written(ct.paths, write({}), image, target, region, "constant", 1.0)
    with pytest.raises(PatientError, match="OUTSIDE"):
        verify_written(ct.paths, write({(3, 10, 10): 999}), image, target, region, "constant", 1.0)
    with pytest.raises(PatientError, match="do not hold"):
        verify_written(ct.paths, write({(2, 0, 0): 0}), image, target, region, "constant", 1.0)
    with pytest.raises(PatientError, match="slice order"):
        verify_written(ct.paths, write({}), image[::-1], target, region, "constant", 1.0)


def test_dry_run_writes_nothing(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.dcm", ct)
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options(), dry_run=True)
    assert res.status == "dry run" and res.rois_removed == ["BODY"] and res.ct_files == ct.shape[0]
    assert not (tmp_path / "out").exists() and not (tmp_path / "qc").exists() and not fake_ts.calls


def test_rerun_after_interruption_starts_clean(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ph.write_ct_series(pat / "CT")
    out, qc = tmp_path / "out" / "PAT01", tmp_path / "qc" / "PAT01"
    stale = out.with_name("PAT01.partial") / "dcm"
    stale.mkdir(parents=True)
    (stale / "half_written.dcm").write_bytes(b"x")
    qc.mkdir(parents=True)
    (qc / "face_mask_series3.nii.gz").write_bytes(b"old")
    (qc / "my_notes.txt").write_text("kept")                    # not ours: never deleted
    deface_dicom_patient(pat, "PAT01", out, qc, Options())
    assert not out.with_name("PAT01.partial").exists()
    assert not (out / "dcm" / "half_written.dcm").exists()
    assert not (qc / "face_mask_series3.nii.gz").exists() and (qc / "my_notes.txt").exists()


def test_no_ct_series(tmp_path, fake_ts):
    pat = tmp_path / "PAT01"
    ph.write_simple(pat / "MR" / "mr1", "MR", ph.MR_IMAGE_STORAGE)
    with pytest.raises(PatientError, match="no CT series"):
        deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options())


def test_roi_overlap_uses_the_series_the_rtstruct_was_drawn_on(tmp_path, fake_ts):
    """Two series of one scan share the frame of reference (e.g. two reconstructions): the
    overlap must be computed on the series the RTSTRUCT refers to, not the largest one."""
    pat = tmp_path / "in" / "PAT01"
    sag = ph.write_ct_series(pat / "CT_sag", orientation="sagittal")                # largest
    axial = ph.write_ct_series(pat / "CT_ax", shape=(30, 56, 56), study_uid=sag.study_uid, frame_uid=sag.frame_uid)
    ph.write_rtstruct(pat / "RS.dcm", axial)
    qc = tmp_path / "qc"
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "out", qc, Options())
    assert not [f for f in res.flags if "overlap" in f]
    rows = {r["roi"]: r for r in res.overlap}
    assert set(rows) == {"PTV_60", "Eye_L"}
    assert rows["Eye_L"]["pct_in_defaced_region"] > 50 and rows["PTV_60"]["pct_in_defaced_region"] == 0
    import csv
    with open(qc / "roi_overlap.csv", encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert [r["roi"] for r in written] == ["PTV_60", "Eye_L"]
    assert written[1]["pct_in_defaced_region"] == str(rows["Eye_L"]["pct_in_defaced_region"])


def test_roi_overlap_columns_and_picture(tmp_path, fake_ts):
    import csv
    import SimpleITK as sitk
    res, ct, out, qc = run(tmp_path, rt=True)
    with open(qc / "roi_overlap.csv", encoding="utf-8") as fh:
        rows = {r["roi"]: r for r in csv.DictReader(fh)}
    eye = rows["Eye_L"]
    assert float(eye["pct_in_defaced_region"]) == 100.0
    voxel_ml = 4.5 * 4.5 * 6.0 / 1000
    assert float(eye["ml_in_defaced_region"]) == pytest.approx(int(eye["voxels"]) * voxel_ml, abs=0.01)
    assert 0 < float(eye["pct_in_face_mask"]) <= 100.0
    assert float(rows["PTV_60"]["ml_in_defaced_region"]) == 0.0 and float(rows["PTV_60"]["pct_in_face_mask"]) == 0
    pic = sitk.ReadImage(str(qc / "QC_roi_overlap.png"))
    assert pic.GetNumberOfComponentsPerPixel() == 3
    assert not list(out.rglob("*.png"))                               # never in the shareable output


def test_no_roi_picture_without_rois_near_the_region(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.dcm", ct, [("BODY", "EXTERNAL"), ("PTV_60", "PTV")])
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options())
    assert [r["roi"] for r in res.overlap] == ["PTV_60"]
    assert not list((tmp_path / "qc").glob("QC_roi_overlap*.png"))
