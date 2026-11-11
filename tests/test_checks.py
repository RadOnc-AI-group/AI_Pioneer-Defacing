"""Anatomy check, face check, signs of earlier de-facing and timing."""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import phantom as ph
import defacing.anatomy
import defacing.reid
from defacing.anatomy import assess as anatomy_assess
from defacing.common import Options, PatientError
from defacing.masks import Axes
from defacing.patient import deface_dicom_patient, deface_image_file
from defacing.prior import constant_block, evidence, header_evidence
from defacing.reid import assess as face_assess, check_face_check, render_views
from defacing.run import run_single
from defacing.timing import STAGES, Timer, summary_lines


# ---------------------------------------------------------------------------
# Anatomy check
# ---------------------------------------------------------------------------

AXES = Axes(spacing=(2.0, 2.0, 2.0), anterior=(1, False), si_axis=0)     # low index along axis 1 = anterior


def anatomy_case(tmp_path, *names):
    """A head with soft tissue from index 10 and landmarks at index 20; returns hu and a loader."""
    hu = np.full((30, 50, 30), -1000.0)
    hu[:, 10:45, 5:25] = 40.0
    tmp_path.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in names:
        m = np.zeros(hu.shape, bool)
        m[8:14, 20:24, 12:18] = True
        arrays[name] = m
        (tmp_path / f"{name}.nii.gz").touch()
    return hu, lambda p: arrays[Path(p).name.removesuffix(".nii.gz")]


def test_tissue_in_front_of_the_nasal_cavity_is_reported(tmp_path):
    hu, load = anatomy_case(tmp_path, "nasal_cavity_left")
    region = np.zeros_like(hu, bool)                           # the mask missed the nose
    results, flags = anatomy_assess(hu, region, region, AXES, load, tmp_path, "CT")
    assert results["nose"]["max_tissue_mm"] == pytest.approx(20.0) and results["nose"]["area_cm2"] > 0.5
    assert any("the nose may not be removed" in f for f in flags)


def test_removed_tissue_in_front_is_fine(tmp_path):
    hu, load = anatomy_case(tmp_path, "nasal_cavity_left", "teeth_upper", "eye_lens_left", "mandible")
    region = np.zeros_like(hu, bool)
    region[:, :20, :] = True
    results, flags = anatomy_assess(hu, region, region, AXES, load, tmp_path, "CT")
    assert not flags and results["nose"]["max_tissue_mm"] == 0.0


def test_tissue_in_front_of_the_rear_of_a_landmark_is_ignored(tmp_path):
    hu, _ = anatomy_case(tmp_path, "nasal_cavity_left")
    cavity = np.zeros(hu.shape, bool)
    cavity[8:14, 20:24, 12:18] = True                          # front of the cavity
    cavity[8:14, 30:34, 18:24] = True                          # rear part, 20 mm further back
    region = np.zeros_like(cavity)
    region[:, :20, 12:18] = True                               # the nose is gone, the maxilla is not
    results, flags = anatomy_assess(hu, region, region, AXES, lambda p: cavity, tmp_path, "CT")
    assert not flags and results["nose"]["max_tissue_mm"] == 0.0


def test_a_thin_shell_removed_from_the_nose_is_still_reported(tmp_path):
    hu, load = anatomy_case(tmp_path, "nasal_cavity_left")
    region = np.zeros_like(hu, bool)
    region[:, :12, :] = True                                   # only the outer 4 mm of the skin
    results, flags = anatomy_assess(hu, region, region, AXES, load, tmp_path, "CT")
    assert results["nose"]["max_tissue_mm"] == pytest.approx(16.0)
    assert any("the nose may not be removed" in f for f in flags)


def test_chin_is_only_reported_in_the_results(tmp_path):
    hu, load = anatomy_case(tmp_path, "mandible")
    region = np.zeros_like(hu, bool)
    results, flags = anatomy_assess(hu, region, region, AXES, load, tmp_path, "CT")
    assert not flags and results["chin"]["max_tissue_mm"] == pytest.approx(20.0)


def test_missing_landmarks_are_not_a_flag(tmp_path):
    hu, load = anatomy_case(tmp_path)
    region = np.zeros_like(hu, bool)
    results, flags = anatomy_assess(hu, region, region, AXES, load, tmp_path, "CT")
    assert not flags and results["nose"] == {"landmark_found": False}


def test_brain_in_the_region_and_orientation_are_reported(tmp_path):
    hu, _ = anatomy_case(tmp_path)
    nose, brain = np.zeros(hu.shape, bool), np.zeros(hu.shape, bool)
    nose[8:14, 35:40, 12:18] = True                            # the "nose" behind the brain: orientation wrong
    brain[2:28, 10:30, 8:22] = True
    arrays = {"nasal_cavity_left": nose, "brain": brain}
    for name in arrays:
        (tmp_path / f"{name}.nii.gz").touch()
    region = np.zeros_like(hu, bool)
    region[:, :20, :] = True
    results, flags = anatomy_assess(hu, region, region, AXES, lambda p: arrays[Path(p).name.removesuffix(".nii.gz")],
                                    tmp_path, "CT")
    assert results["brain_in_region_ml"] > 5 and any("brain inside" in f for f in flags)
    assert results["face_in_front_of_brain"] is False and any("orientation" in f for f in flags)


@pytest.fixture
def fake_landmarks(monkeypatch):
    """Landmarks of the phantom, written where TotalSegmentator would write them."""
    calls = []

    def fake(input_nii, seg_dir, device, log_path):
        calls.append(input_nii)
        img = nib.load(str(input_nii))
        world = ph.nifti_world_lps(img.affine, img.shape)
        seg_dir.mkdir(parents=True, exist_ok=True)
        nasal = ph.in_box(world, ((-10, 10), (-80, -62), (-20, 10)))
        nib.save(nib.Nifti1Image(nasal.astype(np.uint8), img.affine), str(seg_dir / "nasal_cavity_left.nii.gz"))
        brain = ph.in_box(world, ((-40, 40), (-40, 50), (-30, 50)))
        nib.save(nib.Nifti1Image(brain.astype(np.uint8), img.affine), str(seg_dir / "brain.nii.gz"))
        return seg_dir

    monkeypatch.setattr(defacing.anatomy, "segment_landmarks", fake)
    return calls


def test_anatomy_check_in_the_pipeline(tmp_path, fake_ts, fake_landmarks):
    pat = tmp_path / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1, anatomy_check=True))
    assert res.status == "done" and len(fake_landmarks) == 1
    assert res.checks[0]["anatomy"]["nose"]["landmark_found"] and res.timing["anatomy_check"] >= 0
    assert "nose" in res.anatomy_text()


def test_anatomy_check_is_off_by_default(tmp_path, fake_ts, fake_landmarks):
    pat = tmp_path / "PAT01"
    ph.write_ct_series(pat / "CT")
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1))
    assert res.status == "done" and not fake_landmarks and "anatomy" not in res.checks[0]


# ---------------------------------------------------------------------------
# Face check
# ---------------------------------------------------------------------------

def volume():
    z, y, x = np.meshgrid(np.arange(-72, 72, 4.0), np.arange(-126, 126, 4.0), np.arange(-110, 110, 4.0), indexing="ij")
    world = np.stack([x, y, z], axis=-1)
    hu = ph.phantom_hu(world, np.random.default_rng(1))
    return hu, ph.in_face(world), Axes(spacing=(4.0, 4.0, 4.0), anterior=(1, False), si_axis=0)


def test_render_views():
    hu, region, axes = volume()
    views = render_views(hu, region, axes)
    assert set(views) == set(defacing.reid.YAWS)
    assert views[0].ndim == 3 and views[0].dtype == np.uint8 and views[0].max() > 100 and views[0].min() == 0


def face(vector):
    return {"embedding": vector, "face_confidence": 0.99}


def stub(monkeypatch, before, after):
    """detect_face returns `before` for renders of the original and `after` for the de-faced ones."""
    seen = {"n": 0}

    def fake(image):
        seen["n"] += 1
        return before if seen["n"] % 2 == 1 else after

    monkeypatch.setattr(defacing.reid, "detect_face", fake)


def score(opts=None):
    hu, region, axes = volume()
    after = hu.copy()
    after[region] = -1000.0
    return face_assess(hu, after, region, axes, opts or Options(face_check=True), "CT")


def test_no_face_after_defacing(monkeypatch):
    stub(monkeypatch, face([1.0, 0.0]), None)
    result, flags = score()
    assert result["status"] == "no_face" and result["score"] == 0.0 and not flags


def test_a_matching_face_is_flagged(monkeypatch, caplog):
    stub(monkeypatch, face([1.0, 0.0]), face([1.0, 0.01]))
    result, flags = score()
    assert result["status"] == "match" and result["score"] > 0.9 and len(flags) == 1 and "matches" in flags[0]


def test_any_face_is_reported_but_not_flagged_by_default(monkeypatch, caplog):
    stub(monkeypatch, face([1.0, 0.0]), face([0.0, 1.0]))
    with caplog.at_level("WARNING", logger="deface"):
        result, flags = score()
    assert result["status"] == "face_found" and not flags and "a face was found" in caplog.text


def test_flag_rules_can_be_changed(monkeypatch):
    stub(monkeypatch, face([1.0, 0.0]), face([0.0, 1.0]))
    assert score(Options(face_check=True, face_check_flag="any"))[1]
    stub(monkeypatch, face([1.0, 0.0]), face([1.0, 0.01]))
    assert not score(Options(face_check=True, face_check_flag="none"))[1]


@pytest.mark.parametrize("rule, after, fails", [("never", [1.0, 0.01], False), ("match", [1.0, 0.01], True),
                                                ("match", [0.0, 1.0], False), ("any", [0.0, 1.0], True)])
def test_fail_rules(monkeypatch, rule, after, fails):
    stub(monkeypatch, face([1.0, 0.0]), face(after))
    opts = Options(face_check=True, face_check_fail=rule)
    if fails:
        with pytest.raises(PatientError, match="nothing was written"):
            score(opts)
    else:
        score(opts)


def test_no_face_in_the_original_is_inconclusive(monkeypatch):
    stub(monkeypatch, None, None)
    result, flags = score()
    assert result["status"] == "inconclusive" and result["score"] is None and not flags


def test_face_check_in_the_pipeline_only_changes_the_report(tmp_path, fake_ts, monkeypatch):
    stub(monkeypatch, face([1.0, 0.0]), None)
    pat = tmp_path / "in" / "PAT01"
    ph.write_ct_series(pat / "CT")
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1, face_check=True))
    assert res.status == "done" and res.face_check_text() == ("no_face", "0.000") and res.timing["face_check"] >= 0
    manifest = json.loads((tmp_path / "q" / "manifest.json").read_text())
    assert manifest["checks"][0]["face_check"]["status"] == "no_face"
    assert "embedding" not in json.dumps(manifest)


def test_missing_packages_stop_the_run(tmp_path, fake_ts, ts_on_path, monkeypatch):
    monkeypatch.setattr(defacing.run, "check_face_check", lambda: "install it")
    pat = tmp_path / "PAT01"
    ph.write_ct_series(pat / "CT")
    assert run_single(pat, tmp_path / "o", tmp_path / "q", Options(face_check=True), False) == 2
    assert not fake_ts.calls


def test_check_face_check_names_the_install_command(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert "requirements-reid.txt" in check_face_check()


# ---------------------------------------------------------------------------
# Earlier de-facing
# ---------------------------------------------------------------------------

def test_header_evidence():
    ds = Dataset()
    assert header_evidence(ds) is None
    ds.DeidentificationMethod = ["De-faced: TotalSegmentator face mask"]
    assert "DeidentificationMethod" in header_evidence(ds)
    ds = Dataset()
    ds.ImageComments = "scan defaced by the sender"
    assert "ImageComments" in header_evidence(ds)
    code = Dataset()
    code.CodeValue = "113102"
    ds = Dataset()
    ds.DeidentificationMethodCodeSequence = Sequence([code])
    assert "113102" in header_evidence(ds)


def test_constant_block():
    hu, region, axes = volume()
    assert constant_block(hu, axes) is None
    filled = hu.copy()
    filled[region] = -1000.0
    block = constant_block(filled, axes)
    assert block and block[0] == -1000.0 and block[1] >= 20
    clipped = np.where(hu < -500, -1024.0, hu)                 # all air is one value: not a block of de-facing
    assert constant_block(clipped, axes) is None


def test_file_name_is_evidence():
    hu, _, axes = volume()
    assert "file name" in evidence(hu, axes, path=Path("x_deFaced.nii.gz"))


def test_defaced_output_is_flagged_and_defaced_again(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ph.write_ct_series(pat / "CT")
    deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1))
    res = deface_dicom_patient(tmp_path / "o", "PAT01", tmp_path / "o2", tmp_path / "q2", Options(seed=1))
    assert res.status == "done" and any("already de-faced" in f for f in res.flags)


def test_empty_mask_on_a_defaced_scan_says_so(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ph.write_ct_series(pat / "CT", extra={"ImageComments": "De-faced"})
    fake_ts.mode = "empty"
    with pytest.raises(PatientError, match="appears to be already de-faced"):
        deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1))


def test_image_file_with_defaced_name_is_flagged(tmp_path, fake_ts):
    path = tmp_path / "P_deFaced.nii.gz"
    ph.write_nifti(path)
    res = deface_image_file(path, "P", tmp_path / "o", tmp_path / "q", Options(seed=1))
    assert res.status == "done" and any("already de-faced" in f for f in res.flags)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def test_timer_adds_up():
    t = Timer()
    with t.stage("fill"):
        pass
    with t.stage("fill"):
        pass
    assert set(t.seconds) == {"fill"} and t.seconds["fill"] >= 0


def test_summary_per_device():
    lines = summary_lines([("cpu", 10.0, {"segment": 8.0}), ("cpu", 20.0, {"segment": 16.0}), ("gpu", 5.0, {"segment": 1.0})])
    text = "\n".join(lines)
    assert "Timing on cpu (2 patient(s))" in text and "Timing on gpu (1 patient(s))" in text and "segment" in text


def test_timing_is_reported(tmp_path, fake_ts, ts_on_path):
    pat = tmp_path / "PAT01"
    ph.write_ct_series(pat / "CT")
    qc = tmp_path / "q"
    assert run_single(pat, tmp_path / "o", qc, Options(seed=1), False) == 0
    manifest = json.loads((qc / "manifest.json").read_text())
    assert set(manifest["timing_seconds"]) <= set(STAGES) and manifest["timing_seconds"]["segment"] >= 0
    rows = list(csv.DictReader(open(next(qc.glob("timing_*.csv")), encoding="utf-8")))
    assert float(rows[0]["total_s"]) > 0 and rows[0]["device"] == "cpu" and "segment_s" in rows[0]
