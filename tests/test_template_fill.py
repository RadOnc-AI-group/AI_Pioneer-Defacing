"""--fill template (experimental): the fill and its plumbing, with the fit replaced where it does not matter."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

import defacing.template as tmpl
import phantom as ph
import test_fill
from defacing.common import Options, PatientError
from defacing.fill import PADDING_BELOW_HU, TemplateFront, apply_fill, blend_weight
from defacing.patient import deface_dicom_patient
from defacing.qc import _oriented
from defacing.run import run_single
from defacing.template import SYNTHETIC_NOTE, Fit, Pose, prepare_fill

GOLDEN = Path(__file__).parent / "golden" / "fill_template_v1.npz"
REPO = Path(__file__).resolve().parent.parent


def front_for(hu, axes, depth=0.7):
    """A flat front at `depth` of the anterior axis, for every column."""
    v, spacing = _oriented(hu, axes)
    return TemplateFront(np.full((v.shape[0], v.shape[2]), (v.shape[1] - 1) * spacing[1] * depth))


def case(depth=0.9, **kwargs):
    hu, region, axes = test_fill.volume(**kwargs)
    return hu, region, axes, front_for(hu, axes, depth)


def test_the_fill_needs_a_front():
    hu, region, axes = test_fill.volume()
    with pytest.raises(ValueError, match="front"):
        apply_fill(hu, region, axes, "template", seed=1)


def test_voxels_inside_the_region_do_not_matter():
    hu, region, axes, front = case()
    other = hu.copy()
    inside = region & (hu >= PADDING_BELOW_HU)
    other[inside] = np.random.default_rng(5).uniform(-1000, 1500, int(inside.sum()))
    a, _ = apply_fill(hu, region, axes, "template", 3, template=front)
    b, _ = apply_fill(other, region, axes, "template", 3, template=front)
    assert np.array_equal(a, b)


def test_only_the_region_changes():
    hu, region, axes, front = case()
    out, info = apply_fill(hu, region, axes, "template", 3, template=front)
    assert np.array_equal(out[~region], hu[~region].astype(np.float32)) and info["fill"] == "template"
    assert info["voxels_tissue"] > 0


def test_tissue_lies_behind_the_front_and_air_in_front_of_it():
    hu, region, axes, front = case()
    out, _ = apply_fill(hu, region, axes, "template", 3, template=front)
    v = _oriented(out, axes)[0]
    r = _oriented(region, axes)[0]
    spacing = _oriented(region, axes)[1]
    weight = blend_weight(r.any(axis=1), (spacing[0], spacing[2]))
    position = np.arange(v.shape[1]) * spacing[1]
    centre = weight == 1.0
    assert centre.any()
    z, lr = np.nonzero(centre)
    behind = position < front.front[z, lr][:, None] - spacing[1]
    ahead = position > front.front[z, lr][:, None] + spacing[1]
    rv, vv = r[z, :, lr], v[z, :, lr]
    assert (vv[rv & ahead] < -500).all()
    assert (vv[rv & behind] > -500).all()                       # no air gap between the filled tissue and the body


def test_the_front_blends_into_the_hull_at_the_outline():
    region2d = np.zeros((60, 60), bool)
    region2d[10:50, 10:50] = True
    weight = blend_weight(region2d, (1.0, 1.0))
    assert weight[10, 30] < 0.05 and weight[30, 30] == 1.0 and (weight[~region2d] == 0).all()


def test_columns_without_a_template_get_the_hull_front():
    hu, region, axes, front = case()
    nothing = TemplateFront(np.full_like(front.front, np.nan))
    a, _ = apply_fill(hu, region, axes, "template", 3, template=nothing)
    b, _ = apply_fill(hu, region, axes, "soft-tissue", 3)
    assert ((a[region] > -500) == (b[region] > -500)).mean() > 0.98


def test_the_surface_has_partial_volume():
    hu, region, axes, _ = case()
    v, spacing = _oriented(hu, axes)
    half = TemplateFront(np.full((v.shape[0], v.shape[2]), (v.shape[1] - 1) * spacing[1] * 0.9 + 0.3 * spacing[1]))
    out, _ = apply_fill(hu, region, axes, "template", 3, template=half)
    assert ((out[region] > -800) & (out[region] < -200)).any()


@pytest.mark.parametrize("flip, anterior, superior", [((1,), (1, True), True), ((0,), (1, False), False), ((0, 1), (1, True), False)])
def test_prone_and_feet_first_are_mirror_equivalent(flip, anterior, superior):
    hu, region, axes, front = case()
    a, _ = apply_fill(hu, region, axes, "template", 3, template=front)
    other_axes = type(axes)(axes.spacing, anterior, axes.si_axis, superior)
    b, _ = apply_fill(np.flip(hu, flip), np.flip(region, flip), other_axes, "template", 3, template=front)
    assert ((a[region] > -500) == (np.flip(b, flip)[region] > -500)).mean() > 0.995


def test_transposed_axes():
    hu, region, axes = test_fill.volume(transpose=True)
    out, _ = apply_fill(hu, region, axes, "template", 3, template=front_for(hu, axes))
    assert np.array_equal(out[~region], hu[~region].astype(np.float32)) and (out[region] > -500).any()


def test_golden():
    hu, region, axes = test_fill.volume(fov_radius=120, shell=True)
    out, _ = apply_fill(hu, region, axes, "template", 12345, template=front_for(hu, axes, 0.85))
    new = np.round(out[region]).astype(np.int16)
    golden = np.load(GOLDEN)["template"]
    assert new.shape == golden.shape and (new != golden).mean() < 0.005


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def rejected(*args, **kwargs):
    """No pose at all."""
    return Fit(False, "too few skin points around the face")


def weak(t, hu, face, region, axes):
    """A pose that misses the quality limits."""
    return Fit(**{**accepted(t, hu, face, region, axes).__dict__, "accepted": False, "reason": "the pose is ambiguous"})


def accepted(t, hu, face, region, axes):
    v, spacing = _oriented(hu, axes)
    return Fit(True, "", pose=Pose(1.0, np.eye(3), np.zeros(3)), rms_mm=1.0, points=999, coverage=1.0,
               front=np.full((v.shape[0], v.shape[2]), (v.shape[1] - 1) * spacing[1] * 0.85))


def good_skin(monkeypatch):
    monkeypatch.setattr(tmpl, "front_minus_skin", lambda *a: {"skin_p10_mm": -1.0, "skin_p50_mm": 0.0, "skin_p90_mm": 1.0})


def test_prepare_fill_leaves_other_fills_alone():
    assert prepare_fill(Options(fill="air"), None, None, None, None, "CT") == ("air", None, [], {})


def test_a_good_fit_is_not_flagged(monkeypatch):
    good_skin(monkeypatch)
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    hu, region, axes = test_fill.volume()
    mode, front, flags, info = prepare_fill(Options(fill="template"), hu, region, region, axes, "CT")
    assert mode == "template" and front is not None and not flags and info["template_quality"] == "ok"


def test_a_weak_fit_is_inserted_and_flagged_by_default(monkeypatch):
    good_skin(monkeypatch)
    monkeypatch.setattr(tmpl, "fit_template", weak)
    hu, region, axes = test_fill.volume()
    mode, front, flags, info = prepare_fill(Options(fill="template"), hu, region, region, axes, "CT")
    assert mode == "template" and front is not None and info["fill_used"] == "template"
    assert "does not fit well" in flags[0] and "inserted anyway" in flags[0] and info["template_quality"] == "the pose is ambiguous"


def test_a_front_far_from_the_real_skin_is_flagged_but_still_inserted(monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    monkeypatch.setattr(tmpl, "front_minus_skin", lambda *a: {"skin_p10_mm": 2.0, "skin_p50_mm": 12.0, "skin_p90_mm": 23.0})
    hu, region, axes = test_fill.volume()
    mode, _, flags, info = prepare_fill(Options(fill="template"), hu, region, region, axes, "CT")
    assert mode == "template" and "+12 mm from the real skin" in flags[0] and info["template_quality"] != "ok"


def test_the_real_skin_never_triggers_a_fallback(monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    monkeypatch.setattr(tmpl, "front_minus_skin", lambda *a: {"skin_p10_mm": 40.0, "skin_p50_mm": 50.0, "skin_p90_mm": 60.0})
    hu, region, axes = test_fill.volume()
    assert prepare_fill(Options(fill="template", template_fallback="soft-tissue"), hu, region, region, axes, "CT")[0] == "template"


def test_fallback_soft_tissue_replaces_a_weak_fit(monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", weak)
    hu, region, axes = test_fill.volume()
    mode, front, flags, info = prepare_fill(Options(fill="template", template_fallback="soft-tissue"), hu, region, region, axes, "CT")
    assert mode == "soft-tissue" and front is None and "soft-tissue used instead" in flags[0]
    assert info["fill_used"] == "soft-tissue" and info["template_fallback"] == "the pose is ambiguous" and len(info["template_sha256"]) == 64


def test_fallback_fail_fails_a_weak_fit(monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", weak)
    hu, region, axes = test_fill.volume()
    with pytest.raises(PatientError, match="nothing was written"):
        prepare_fill(Options(fill="template", template_fallback="fail"), hu, region, region, axes, "CT")


def test_without_a_pose_the_patient_fails_and_names_the_option(monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", rejected)
    hu, region, axes = test_fill.volume()
    with pytest.raises(PatientError, match="--fallback soft-tissue"):
        prepare_fill(Options(fill="template"), hu, region, region, axes, "CT")
    assert prepare_fill(Options(fill="template", template_fallback="soft-tissue"), hu, region, region, axes, "CT")[0] == "soft-tissue"


def test_unknown_orientation_is_treated_as_no_pose():
    hu, region, axes = test_fill.volume()
    with pytest.raises(PatientError, match="orientation"):
        prepare_fill(Options(fill="template"), hu, region, region, axes, "CT", orientation_known=False)


def test_the_bundled_template_loads():
    t = tmpl.load_template(tmpl.DEFAULT_TEMPLATE)
    assert len(t.vertices) == 5023 and (REPO / "defacing" / "data" / "flame_template.npz.NOTICE.txt").exists()


def test_the_note_fits_a_dicom_long_string():
    assert len(SYNTHETIC_NOTE) <= 64


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def patient(tmp_path, opts, rois=None):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    if rois:
        ph.write_rtstruct(pat / "RS.dcm", ct, rois)
    out, qc = tmp_path / "out", tmp_path / "qc"
    res = deface_dicom_patient(pat, "PAT01", out, qc, opts)
    assert res.status == "done", res.error
    return res, ct, out, qc


def test_dicom_says_the_face_is_synthetic(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    res, ct, out, qc = patient(tmp_path, Options(seed=1, fill="template"))
    _, _, dss = ph.read_series_hu(out / "dcm")
    methods = [str(v) for v in dss[0].DeidentificationMethod]
    assert SYNTHETIC_NOTE in methods and any("fill=template" in m for m in methods)
    assert all(len(m) <= 64 for m in methods)
    manifest = json.loads((qc / "manifest.json").read_text())
    assert manifest["fill_details"][0]["fill_used"] == "template" and "template_sha256" in manifest["fill_details"][0]
    assert list(qc.glob("QC_defacing*.png"))


def test_a_fallback_does_not_claim_a_synthetic_face(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", rejected)
    res, ct, out, qc = patient(tmp_path, Options(seed=1, fill="template", template_fallback="soft-tissue"))
    _, _, dss = ph.read_series_hu(out / "dcm")
    methods = [str(v) for v in dss[0].DeidentificationMethod]
    assert SYNTHETIC_NOTE not in methods and any("fill=soft-tissue" in m for m in methods)
    assert any("soft-tissue used instead" in f for f in res.flags)


def test_clip_body_uses_the_synthetic_tissue(tmp_path, fake_ts, monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    res, *_ = patient(tmp_path, Options(seed=1, fill="template", clip_body=True), [("BODY", "EXTERNAL")])
    assert res.rois_clipped == ["BODY"]


def test_kept_targets_in_front_of_the_face_are_reported(tmp_path, fake_ts, monkeypatch):
    def far_back(t, hu, face, region, axes):
        v, spacing = _oriented(hu, axes)
        return Fit(True, "", pose=Pose(1.0, np.eye(3), np.zeros(3)), front=np.full((v.shape[0], v.shape[2]), 0.0))
    monkeypatch.setattr(tmpl, "fit_template", far_back)
    res, *_ = patient(tmp_path, Options(seed=1, fill="template"), [("PTV_nose", "PTV")])
    assert any("show through" in f for f in res.flags)


def test_reports_show_the_fill_used(tmp_path, fake_ts, ts_on_path, monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", rejected)
    pat = tmp_path / "in"
    ph.write_ct_series(pat / "CT")
    out, qc = tmp_path / "o", tmp_path / "q"
    assert run_single(pat, out, qc, Options(seed=1, fill="template", template_fallback="soft-tissue"), False) == 0
    row = list(csv.DictReader(open(next(qc.glob("patients_*.csv")), encoding="utf-8")))[0]
    assert row["fill"] == "template" and row["fill_used"] == "soft-tissue" and row["template_quality"]


def test_nifti_says_the_face_is_synthetic(tmp_path, fake_ts, ts_on_path, monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    image = tmp_path / "ct.nii.gz"
    ph.write_nifti(image)
    out, qc = tmp_path / "o", tmp_path / "q"
    assert run_single(image, out, qc, Options(seed=1, fill="template"), False) == 0
    written = nib.load(str(next(out.glob("*_deFaced.nii.gz"))))
    assert written.header["descrip"].item().decode() == SYNTHETIC_NOTE


def test_nrrd_says_the_face_is_synthetic(tmp_path, fake_ts, ts_on_path, monkeypatch):
    monkeypatch.setattr(tmpl, "fit_template", accepted)
    nii = tmp_path / "ct.nii.gz"
    ph.write_nifti(nii)
    image = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.ReadImage(str(nii)), str(image))
    out, qc = tmp_path / "o", tmp_path / "q"
    assert run_single(image, out, qc, Options(seed=1, fill="template"), False) == 0
    assert sitk.ReadImage(str(next(out.glob("*_deFaced.nrrd")))).GetMetaData("DeidentificationNote") == SYNTHETIC_NOTE


@pytest.mark.parametrize("args, message", [
    (["--fill", "template", "--no-extend-anterior"], "anterior image edge"),
    (["--fill", "template", "--keep-oar"], "keep-oar"),
    (["--fill", "template", "--face-check", "--face-check-fail", "any"], "face-check-fail"),
    (["--fill", "template", "--template-model", "/no/such/file.npz"], "cannot read the template"),
    (["--fallback", "fail"], "need --fill template"),
    (["--template-model", "x.npz"], "need --fill template"),
])
def test_bad_combinations_stop_the_run(tmp_path, args, message):
    p = subprocess.run([sys.executable, str(REPO / "deface.py"), "-i", str(tmp_path), "-o", str(tmp_path / "o"), *args],
                       capture_output=True, text=True)
    assert p.returncode == 2 and message in p.stderr, p.stderr
