"""Organs at risk restored inside the de-faced region (--keep-oar, --keep-oar-ts)."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import defacing.anatomy
import defacing.oars
import phantom as ph
from defacing.common import Options
from defacing.masks import Axes
from defacing.oars import below_skin, plan_oars, restore
from defacing.patient import deface_dicom_patient
from defacing.run import run_single

AXES = Axes(spacing=(1.0, 1.0, 1.0), anterior=(1, False), si_axis=0)        # low index along axis 1 = anterior


@pytest.mark.parametrize("name, kind, restored", [
    ("Mandibula", "ORGAN", True), ("Eye_L", "ORGAN", True), ("Lippen", "ORGAN", True), ("N. opticus re", "ORGAN", True),
    ("Optic_Nerve_L", "ORGAN", True), ("Bulbus li", "ORGAN", True),
    ("Eye_L", "", False), ("Eye_L", "AVOIDANCE", False),            # only the type ORGAN counts
    ("Body", "ORGAN", False), ("Haut", "ORGAN", False), ("Maske", "ORGAN", False),
    ("PTV_60", "ORGAN", False), ("z_Eye", "ORGAN", False), ("Eye_PRV", "ORGAN", False), ("Eye+3mm", "ORGAN", False),
    ("Ring_Eye", "ORGAN", False), ("Parotis-PTV", "ORGAN", False), ("Teeth", "ORGAN", False), ("Zähne", "ORGAN", False),
])
def test_which_rois_are_restored(name, kind, restored):
    assert bool(plan_oars({1: (name, kind)}, Options(keep_oar=True)).keep) is restored


def test_not_oar_and_keep_roi_take_it_out_of_the_plan():
    rois = {1: ("Eye_L", "ORGAN"), 2: ("Mandible", "ORGAN")}
    opts = Options(keep_oar=True, not_oar=[re.compile("eye.*", re.I)], keep_roi=[re.compile("mandible", re.I)])
    assert not plan_oars(rois, opts).keep


def head(thickness=30):
    """Tissue from index 10 along the anterior axis, air in front of it."""
    hu = np.full((40, 60, 40), -1000.0)
    hu[:, 10:10 + thickness, 2:38] = 40.0
    return hu


def test_below_skin_measures_from_the_outside():
    hu = head()
    wanted = np.zeros_like(hu, bool)
    wanted[15:25, 10:30, 15:25] = True
    deep = below_skin(hu, wanted, AXES, 5.0)
    assert not deep[20, 14, 20] and deep[20, 15, 20] and deep[20, 25, 20]


def test_air_inside_the_head_is_not_outside():
    hu = head()
    hu[15:25, 18:24, 15:25] = -1000.0                                     # a sinus
    wanted = np.zeros_like(hu, bool)
    wanted[20, 25, 20] = True                                             # 1 mm behind the sinus
    assert below_skin(hu, wanted, AXES, 5.0)[20, 25, 20]


def test_thin_mask_in_front_of_the_face_is_not_the_patient():
    hu = head()
    hu[:, 4:6, :] = 40.0                                                  # a sheet of mask, 2 mm thick, air behind it
    hu[:, 3:7, 0:2] = 40.0                                                # joined to the head at the sides
    wanted = np.zeros_like(hu, bool)
    wanted[20, 12, 20] = True                                             # 2 mm below the real skin
    assert not below_skin(hu, wanted, AXES, 5.0)[20, 12, 20]


def test_zero_depth_restores_the_surface():
    hu = head()
    wanted = np.zeros_like(hu, bool)
    wanted[20, 10, 20] = True
    assert below_skin(hu, wanted, AXES, 0.0)[20, 10, 20]


def face_case():
    hu = head()
    face = np.zeros_like(hu, bool)
    face[:, 10:30, 2:38] = True                                           # tissue only: 40 x 20 x 36 voxels
    region = np.zeros_like(hu, bool)
    region[:, :30, :] = True
    return hu, face, region


def roi(slab):
    m = np.zeros((40, 60, 40), bool)
    m[slab] = True
    return m


def test_only_the_deep_part_of_an_organ_is_restored():
    hu, face, region = face_case()
    organ = roi((slice(15, 25), slice(10, 25), slice(15, 25)))
    out = restore([("rtstruct", "Eye", organ)], face, region, hu, AXES, Options(keep_oar=True), "CT")
    assert out.keep[20, 22, 20] and not out.keep[20, 12, 20]
    assert out.rows[0]["kept"] and out.rows[0]["ml_restored"] < out.rows[0]["ml_in_region"]
    assert any("ml of OAR voxels restored" in f for f in out.flags)


def test_a_face_contour_is_not_restored():
    hu, face, region = face_case()
    whole = roi((slice(5, 35), slice(10, 30), slice(5, 35)))              # far above 25% of the tissue
    out = restore([("rtstruct", "Gesicht", whole)], face, region, hu, AXES, Options(keep_oar=True), "CT")
    assert out.keep is None and not out.rows[0]["kept"] and out.rows[0]["pct_of_face_tissue"] >= 25
    assert any("NOT restored" in f for f in out.flags)


def test_the_share_is_taken_of_the_tissue_not_of_the_air_in_the_mask():
    hu, face, region = face_case()
    face[:, 0:10, :] = True                                               # as a real mask: mostly air
    face[:, 40:60, :] = True
    whole = roi((slice(5, 35), slice(10, 30), slice(5, 35)))
    out = restore([("rtstruct", "Gesicht", whole)], face, region, hu, AXES, Options(keep_oar=True), "CT")
    assert out.keep is None


def test_many_small_organs_together_are_reported():
    hu, face, region = face_case()
    parts = [("rtstruct", f"Part{i}", roi((slice(i * 8, i * 8 + 7), slice(10, 30), slice(2, 38)))) for i in range(5)]
    out = restore(parts, face, region, hu, AXES, Options(keep_oar=True), "CT")
    assert all(r["kept"] for r in out.rows) and any("together cover" in f for f in out.flags)


def test_no_organ_found_is_reported_unless_totalsegmentator_is_used():
    hu, face, region = face_case()
    assert any("no ROI typed ORGAN" in f for f in restore([], face, region, hu, AXES, Options(keep_oar=True), "CT").flags)
    assert not restore([], face, region, hu, AXES, Options(keep_oar=True, keep_oar_ts=True), "CT").flags


def test_targets_are_not_counted_again():
    hu, face, region = face_case()
    organ = roi((slice(15, 25), slice(16, 25), slice(15, 25)))
    out = restore([("rtstruct", "Eye", organ)], face, region, hu, AXES, Options(keep_oar=True), "CT", exclude=organ)
    assert out.keep is None


def run(tmp_path, rois, opts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.PAT01.dcm", ct, rois)
    out, qc = tmp_path / "out" / "PAT01", tmp_path / "qc" / "PAT01"
    res = deface_dicom_patient(pat, "PAT01", out, qc, opts)
    assert res.status == "done", res.error
    return res, ct, out, qc


def test_off_by_default_changes_nothing(tmp_path, fake_ts):
    rois = [("BODY", "EXTERNAL"), ("Eye_nose", "ORGAN")]
    res, ct, out, qc = run(tmp_path, rois, Options(seed=1))
    hu, world, _ = ph.read_series_hu(out / "dcm")
    box = ph.in_box(world, ((-7, 7), (-99, -71), (-16, 6)))
    assert box.any() and (hu[box] < -850).all()
    assert not res.oars and res.oar_kept_ml == [0.0] and not any("OAR" in f for f in res.flags)


def test_oar_voxels_are_restored_in_the_output(tmp_path, fake_ts):
    res, ct, out, qc = run(tmp_path, [("BODY", "EXTERNAL"), ("Eye_nose", "ORGAN")],
                           Options(seed=1, keep_oar=True, keep_oar_depth_mm=0.0))
    hu, world, dss = ph.read_series_hu(out / "dcm")
    box = ph.in_box(world, ((-7, 7), (-99, -71), (-16, 6)))
    assert box.any() and np.array_equal(hu[box], ct.hu[box])
    tip = ph.in_nose(world) & (world[..., 1] < -104)                       # in front of the box
    assert tip.any() and (hu[tip] < -850).all()
    assert res.oar_kept_ml[0] > 0 and res.oars[0]["roi"] == "Eye_nose" and res.oars[0]["source"] == "rtstruct"
    assert "Organ at risk voxels below the skin inside the region kept" in [str(v) for v in dss[0].DeidentificationMethod]
    manifest = json.loads((qc / "manifest.json").read_text())
    assert manifest["keep_oar"] is True and manifest["oar_kept_ml"] == res.oar_kept_ml
    assert any("ml of OAR voxels restored" in f for f in res.flags)


def test_depth_limit_reaches_the_output(tmp_path, fake_ts):
    rois = [("BODY", "EXTERNAL"), ("Eye_nose", "ORGAN")]
    shallow, *_ = run(tmp_path / "a", rois, Options(seed=1, keep_oar=True, keep_oar_depth_mm=0.0))
    deep, *_ = run(tmp_path / "b", rois, Options(seed=1, keep_oar=True, keep_oar_depth_mm=30.0))
    assert deep.oar_kept_ml[0] < shallow.oar_kept_ml[0]


def test_oars_stay_with_deface_targets(tmp_path, fake_ts):
    res, *_ = run(tmp_path, [("BODY", "EXTERNAL"), ("Eye_nose", "ORGAN"), ("PTV_nose", "PTV")],
                  Options(seed=1, keep_oar=True, keep_oar_depth_mm=0.0, deface_targets=True))
    assert res.oar_kept_ml[0] > 0 and res.target_kept_ml[0] == 0


def test_oars_stay_in_masked_rtdose(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.PAT01.dcm", ct, [("BODY", "EXTERNAL"), ("Eye_nose", "ORGAN")])
    ph.write_dose(pat / "RD.dcm", ct)
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q",
                               Options(seed=1, keep_oar=True, keep_oar_depth_mm=0.0, mask_rtdose=True))
    assert res.status == "done", res.error
    assert res.rtdose_masked == 1 and res.oar_kept_ml[0] > 0


def fake_seg(calls):
    def fake(task, input_nii, seg_dir, device, log_path, extra=()):
        calls.append(task)
        Path(seg_dir).mkdir(parents=True, exist_ok=True)
        if task == "head_glands_cavities":
            img = nib.load(str(input_nii))
            world = ph.nifti_world_lps(img.affine, img.shape)
            eye = ph.in_box(world, ((-7, 7), (-99, -71), (-16, 6)))
            nib.save(nib.Nifti1Image(eye.astype(np.uint8), img.affine), str(Path(seg_dir) / "eye_left.nii.gz"))
    return fake


def test_totalsegmentator_organs_on_dicom_and_shared_with_the_anatomy_check(tmp_path, fake_ts, monkeypatch):
    calls = []
    monkeypatch.setattr(defacing.anatomy, "run_task", fake_seg(calls))
    pat = tmp_path / "PAT01"
    ph.write_ct_series(pat / "CT")
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q",
                               Options(seed=1, keep_oar=True, keep_oar_ts=True, keep_oar_depth_mm=0.0, anatomy_check=True))
    assert res.status == "done", res.error
    assert res.oars[0]["source"] == "totalsegmentator" and res.oar_kept_ml[0] > 0
    assert calls.count("head_glands_cavities") == 1 and calls.count("craniofacial_structures") == 1 and "total" in calls


def test_not_oar_applies_to_totalsegmentator(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setattr(defacing.anatomy, "run_task", fake_seg([]))
    pat = tmp_path / "PAT01"
    ph.write_ct_series(pat / "CT")
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q",
                               Options(seed=1, keep_oar_ts=True, keep_oar=True, not_oar=[re.compile("eye_.*")]))
    assert res.status == "done" and not res.oars


def test_nifti_with_totalsegmentator(tmp_path, fake_ts, ts_on_path, monkeypatch):
    monkeypatch.setattr(defacing.anatomy, "run_task", fake_seg([]))
    image = tmp_path / "ct.nii.gz"
    ph.write_nifti(image)
    out, qc = tmp_path / "o", tmp_path / "q"
    code = run_single(image, out, qc, Options(seed=1, keep_oar=True, keep_oar_ts=True, keep_oar_depth_mm=0.0), False)
    assert code == 0
    before = nib.load(str(image))
    world = ph.nifti_world_lps(before.affine, before.shape)
    box = ph.in_box(world, ((-7, 7), (-99, -71), (-16, 6)))
    after = nib.load(str(next(out.glob("*_deFaced.nii.gz")))).get_fdata()
    assert box.any() and np.array_equal(after[box], before.get_fdata()[box])
    rows = list(csv.DictReader(open(next(qc.glob("patients_*.csv")), encoding="utf-8")))
    assert float(rows[0]["oar_kept_ml"]) > 0


def test_nifti_with_keep_oar_alone_replaces_everything(tmp_path, fake_ts, ts_on_path):
    image = tmp_path / "ct.nii.gz"
    ph.write_nifti(image)
    out, qc = tmp_path / "o", tmp_path / "q"
    assert run_single(image, out, qc, Options(seed=1, keep_oar=True), False) == 0
    assert json.loads((qc / "manifest.json").read_text())["oar_kept_ml"] == [0.0]


@pytest.mark.parametrize("args", [["--not-oar", "x"], ["--keep-oar-depth-mm", "3"], ["--keep-oar", "--keep-oar-depth-mm", "-1"]])
def test_bad_combinations_stop_the_run(tmp_path, args):
    repo = Path(__file__).resolve().parent.parent
    p = subprocess.run([sys.executable, str(repo / "deface.py"), "-i", str(tmp_path), "-o", str(tmp_path / "o"), *args],
                       capture_output=True, text=True)
    assert p.returncode == 2 and "keep-oar" in p.stderr
