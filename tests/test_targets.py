"""Target volumes: which ROIs they are, and that their voxels stay unchanged."""
import csv
import json
import re

import numpy as np
import pydicom
import pytest

import phantom as ph
from defacing.common import Options
from defacing.masks import Axes
from defacing.patient import deface_dicom_patient
from defacing.targets import classify_target, exposed_area_cm2, plan_targets


@pytest.mark.parametrize("name, kind, expected", [
    # by name: English abbreviations
    ("PTV_70", "", "target"), ("PTV-70", "ORGAN", "target"), ("PTV 60Gy", "", "target"), ("GTVp", "", "target"),
    ("GTVn", "", "target"), ("GTV-T", "", "target"), ("GTV_LK", "", "target"), ("GTVLK", "", "target"),
    ("CTVn_L", "", "target"), ("CTV1", "", "target"), ("PTVboost", "", "target"), ("ITV", "", "target"),
    ("PTVges", "", "target"), ("PTV_eval", "", "target"), ("PTV_crop", "", "target"),
    # German terms
    ("LAW", "", "target"), ("LAW_re", "", "target"), ("LAW li", "", "target"), ("LAWre", "", "target"),
    ("CTV_LAW", "", "target"), ("PTV LAW", "", "target"), ("Lymphabflusswege", "", "target"),
    ("Lymphabfluss re", "", "target"), ("Zielvolumen 1", "", "target"), ("ZV", "", "target"), ("ZV_60Gy", "", "target"),
    ("Planungszielvolumen", "", "target"), ("Planungs-Zielvolumen", "", "target"),
    ("Klinisches Zielvolumen", "", "target"), ("Makroskopisches Tumorvolumen", "", "target"),
    # full English terms; by type only
    ("Planning Target Volume", "", "target"), ("Gross Tumour Volume", "", "target"), ("Whatever", "PTV", "target"),
    # helpers look like targets, but are reported instead
    ("z_PTV", "PTV", "rejected"), ("x-CTV", "", "rejected"), ("_PTV", "", "rejected"), ("Ring_PTV", "", "rejected"),
    ("PTV_ring", "", "rejected"), ("PTV_opt", "", "rejected"), ("PTVopt", "", "rejected"), ("NonPTV", "", "rejected"),
    ("non_PTV", "", "rejected"), ("PTV70-PTV60", "", "rejected"), ("PTV-GTV", "", "rejected"),
    ("PTV+5mm", "", "rejected"), ("PTV_minus_GTV", "", "rejected"), ("Hilfs_PTV", "", "rejected"),
    ("LAW-PTV", "", "rejected"), ("Boost_PTV", "", "rejected"),
    # Body, Skin and unclear names are never targets
    ("Body-PTV", "PTV", "rejected"), ("PTV-Haut", "PTV", "rejected"), ("PTV_Maske", "", "rejected"),
    ("Skin", "PTV", "rejected"), ("BODY", "CTV", "rejected"),
    # not targets
    ("Parotid_L", "ORGAN", ""), ("Brain", "", ""), ("Lawinen", "", ""), ("ZVK", "", ""), ("Boost", "", ""),
    ("LK re", "", ""), ("Lymphknoten", "", ""), ("Tumorbett", "", ""), ("Spinal Cord", "", ""),
])
def test_classify_target(name, kind, expected):
    assert classify_target(name, kind) == expected


def test_a_helper_in_the_region_is_replaced_and_reported(monkeypatch):
    import defacing.targets as targets
    region = np.zeros((4, 10, 10), bool)
    region[1:3, :5, :] = True
    contour = np.zeros((10, 10), bool)
    contour[:3, :3] = True
    monkeypatch.setattr(targets, "roi_table", lambda ds: {1: ("z_PTV", "PTV"), 2: ("Ring_PTV", "")})
    monkeypatch.setattr(targets, "roi_slices", lambda ds, grid: [(1, {1: contour}), (2, {0: contour})])     # 2: outside
    grid = type("G", (), {"spacing": (2.0, 2.0, 2.0)})()
    kept = targets.kept_target_voxels([("rs", object())], grid, region, region, np.full(region.shape, 40.0),
                                      Axes((2.0, 2.0, 2.0), (1, False), 0), Options(), "CT")
    assert kept.keep is None and not kept.rows and len(kept.flags) == 1
    assert "'z_PTV' (0.1 ml)" in kept.flags[0] and "Ring_PTV" not in kept.flags[0]


def test_german_umlauts_are_normalised():
    assert classify_target("PLANUNGSZIELVOLUMEN", "") == "target"
    assert classify_target("Körper-PTV", "") == "rejected"


ROIS = {1: ("BODY", "EXTERNAL"), 2: ("PTV_60", "PTV"), 3: ("z_PTV", "PTV"), 4: ("Lens_L", ""), 5: ("Boost", "")}


def test_plan_targets_default():
    plan = plan_targets(ROIS, Options())
    assert plan.keep == {2: "PTV_60"} and plan.rejected == ["z_PTV"] and not plan.forced


def test_plan_targets_keep_roi_and_not_target():
    opts = Options(keep_roi=[re.compile("boost", re.I)], not_target=[re.compile("ptv_60", re.I)])
    plan = plan_targets(ROIS, opts)
    assert plan.keep == {5: "Boost"} and plan.forced == {5}
    assert 2 not in plan_targets(ROIS, Options(remove_roi=[re.compile("PTV_60", re.I)])).keep  # removed: not kept


def test_body_cannot_be_forced():
    assert not plan_targets({1: ("BODY", "EXTERNAL")}, Options(keep_roi=[re.compile("body", re.I)])).keep


def test_exposed_area():
    hu = np.full((10, 40, 10), -1000.0)
    hu[:, 10:, :] = 40.0                                           # skin at index 10 of a 2 mm axis; anterior = low index
    axes = Axes(spacing=(1.0, 2.0, 1.0), anterior=(1, False), si_axis=0)
    keep = np.zeros_like(hu, bool)
    keep[2:6, 10:13, 2:8] = True                                    # at the skin: 6 x 4 pixels of 1 mm2
    assert exposed_area_cm2(hu, keep, axes) == pytest.approx(24 / 100)
    deep = np.zeros_like(hu, bool)
    deep[2:6, 20:23, 2:8] = True                                    # 20 mm of tissue in front
    assert exposed_area_cm2(hu, deep, axes) == 0.0
    # the mirrored image gives the same area
    flipped = Axes(spacing=(1.0, 2.0, 1.0), anterior=(1, True), si_axis=0)
    assert exposed_area_cm2(hu[:, ::-1], keep[:, ::-1], flipped) == pytest.approx(24 / 100)


def run(tmp_path, rois, opts=None):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.PAT01.dcm", ct, rois)
    out, qc = tmp_path / "out" / "PAT01", tmp_path / "qc" / "PAT01"
    res = deface_dicom_patient(pat, "PAT01", out, qc, opts or Options(seed=1))
    assert res.status == "done", res.error
    return res, ct, out, qc


def nose_box(world):
    return ph.in_box(world, ((-7, 7), (-99, -71), (-16, 6)))


def test_target_voxels_stay_and_the_rest_is_defaced(tmp_path, fake_ts):
    res, ct, out, qc = run(tmp_path, [("BODY", "EXTERNAL"), ("PTV_nose", "PTV")])
    hu, world, dss = ph.read_series_hu(out / "dcm")
    box = nose_box(world)
    assert box.any() and np.array_equal(hu[box], ct.hu[box])
    tip = ph.in_nose(world) & (world[..., 1] < -104)                # in front of the target
    assert tip.any() and (hu[tip] < -850).all()
    others = ph.in_face(world) & ~ph.in_box(world, ((-14, 14), (-108, -62), (-30, 16)))
    assert (hu[others] < -850).all()
    assert res.target_kept_ml[0] > 0 and res.targets[0]["roi"] == "PTV_nose" and res.targets[0]["kept"]
    assert "Target voxels inside the region kept" in [str(v) for v in dss[0].DeidentificationMethod]
    manifest = json.loads((qc / "manifest.json").read_text())
    assert manifest["deface_targets"] is False and manifest["target_kept_ml"] == res.target_kept_ml
    rows = list(csv.DictReader(open(qc / "roi_overlap.csv", encoding="utf-8")))
    assert rows[0]["is_target"] == "yes" and float(rows[0]["pct_in_defaced_region"]) == 0.0
    assert float(rows[0]["ml_kept_in_region"]) == pytest.approx(res.target_kept_ml[0])


def test_deface_targets_replaces_them(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, [("PTV_nose", "PTV")], Options(seed=1, deface_targets=True))
    hu, world, _ = ph.read_series_hu(out / "dcm")
    assert (hu[nose_box(world)] < -850).all() and not any(res.target_kept_ml)


def test_helper_structure_is_not_kept(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, [("z_PTV_nose", "PTV")])
    hu, world, _ = ph.read_series_hu(out / "dcm")
    assert (hu[nose_box(world)] < -850).all() and not res.targets


def test_target_covering_most_of_the_face_is_not_kept(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, [("PTV_large", "PTV")])
    hu, world, _ = ph.read_series_hu(out / "dcm")
    assert (hu[nose_box(world)] < -850).all() and not any(res.target_kept_ml)
    assert res.targets[0]["kept"] is False and any("NOT kept" in f for f in res.flags)


def test_keep_roi_overrules_the_size_guard(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, [("PTV_large", "PTV")], Options(seed=1, keep_roi=[re.compile("PTV_large")]))
    hu, world, _ = ph.read_series_hu(out / "dcm")
    box = nose_box(world)
    assert np.array_equal(hu[box], ct.hu[box]) and res.targets[0]["kept"]
    assert any("kept, so much of the face" in f for f in res.flags)


def test_medium_target_is_kept_and_reported(tmp_path, fake_ts):
    res, ct, out, _ = run(tmp_path, [("PTV_half", "PTV")])
    assert 25 <= res.targets[0]["pct_of_face_mask"] < 75 and res.targets[0]["kept"]
    assert any("kept, so much of the face" in f for f in res.flags)


def test_surface_exposure_is_reported(tmp_path, fake_ts):
    res, _, _, _ = run(tmp_path, [("PTV_half", "PTV")])
    assert res.target_exposed_cm2[0] > 5 and any("of the skin surface" in f for f in res.flags)
    res2 = run(tmp_path / "2", [("PTV_nose", "PTV")])[0]
    assert res2.target_exposed_cm2[0] < res.target_exposed_cm2[0]


def test_structure_set_without_its_series_is_reported(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    other = ph.write_ct_series(tmp_path / "other", seed=3)
    ph.write_rtstruct(pat / "RS.dcm", other, [("PTV_nose", "PTV")])
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "out", tmp_path / "qc", Options(seed=1))
    hu, world, _ = ph.read_series_hu(tmp_path / "out" / "dcm")
    assert res.status == "done" and (hu[nose_box(world)] < -850).all()
    assert any("could not be kept" in f for f in res.flags)


def test_dry_run_names_kept_and_rejected_targets(tmp_path, caplog):
    pat = tmp_path / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.dcm", ct, [("PTV_nose", "PTV"), ("z_PTV", "PTV")])
    with caplog.at_level("INFO", logger="deface"):
        res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(), dry_run=True)
    assert res.status == "dry run" and "PTV_nose" in caplog.text and "z_PTV" in caplog.text
    assert pydicom.dcmread(next(pat.glob("RS*")))                  # the input is only read
