"""Cohort face identification with a stubbed face model: ranks, metrics, the perturbed gallery, the cache."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import defacing.qc
import defacing.reid
from defacing.masks import Axes

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("face_identification", REPO / "tools" / "face_identification.py")
fid = importlib.util.module_from_spec(spec)
sys.modules["face_identification"] = fid
spec.loader.exec_module(fid)


def unit(*v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def head(shape=(40, 60, 60)):
    """A ball of tissue in air, on a (z, y, x) grid."""
    zz, yy, xx = np.indices(shape, float)
    c = (np.array(shape) - 1) / 2
    inside = ((zz - c[0]) / 18) ** 2 + ((yy - c[1]) / 25) ** 2 + ((xx - c[2]) / 25) ** 2 <= 1
    return np.where(inside, 40.0, -1000.0).astype(np.float32)


# ------------------------------- render_views parameters ---------------------------------

def _volume():
    hu = head()
    return hu, np.zeros(hu.shape, bool), Axes(spacing=(2.0, 1.0, 1.0), anterior=(1, False), si_axis=0, superior=True)


def test_render_views_defaults_are_the_face_check_views():
    hu, region, axes = _volume()
    default = defacing.reid.render_views(hu, region, axes)
    explicit = defacing.reid.render_views(hu, region, axes, skin_hu=defacing.qc.SKIN_HU, yaws=defacing.reid.YAWS)
    assert tuple(default) == (-30, 0, 30) == defacing.reid.YAWS
    assert all(np.array_equal(default[y], explicit[y]) for y in default)


def test_render_views_uses_the_given_threshold_and_yaws():
    hu, region, axes = _volume()
    hu[(hu > -1000) & (np.arange(hu.shape[0]).reshape(-1, 1, 1) < 8)] = -250.0     # above -300, below -200
    views = defacing.reid.render_views(hu, region, axes, skin_hu=-200.0, yaws=(-45, 15))
    assert tuple(views) == (-45, 15)
    assert (views[15] > 0).sum() < (defacing.reid.render_views(hu, region, axes, yaws=(15,))[15] > 0).sum()


# ---------------------------------- metrics -------------------------------------------------

GALLERY = {"a": [unit(1, 0, 0)], "b": [unit(0, 1, 0)], "c": [unit(0, 0, 1)]}


def test_genuine_ranks_first():
    probes = {"a": [unit(1, 0.05, 0)], "b": [unit(0.05, 1, 0)], "c": [unit(0, 0.05, 1)]}
    rows, s, scores = fid.evaluate(GALLERY, probes, 0.30)
    assert [r["rank"] for r in rows] == [1, 1, 1]
    assert s["rank1_rate"] == 1.0 and s["rank1_hits"] == 3 and s["mrr"] == 1.0 and s["cmc"] == [1.0, 1.0, 1.0]
    assert s["auc"] == 1.0 and s["chance_rank1"] == round(1 / 3, 3)
    assert len(scores["genuine"]) == 3 and len(scores["impostor"]) == 6 and scores["genuine"] == sorted(scores["genuine"])


def test_missing_faces_count_as_not_identified():
    probes = {"a": [unit(1, 0.05, 0)], "b": [], "c": [unit(0, 1, 0.5)]}      # b no face; c closer to b than to itself
    rows, s, _ = fid.evaluate(GALLERY, probes, 0.30)
    assert [r["rank"] for r in rows] == [1, None, 2]
    assert s["rank1_hits"] == 1 and s["probe_face_rate"] == round(2 / 3, 3)
    assert s["cmc"] == [round(1 / 3, 4), round(2 / 3, 4), round(2 / 3, 4)]
    assert s["mrr"] == round((1 + 0 + 0.5) / 3, 3)


def test_a_gallery_without_a_face_leaves_the_cohort():
    rows, s, _ = fid.evaluate({**GALLERY, "d": []}, {"a": [unit(1, 0, 0)]}, 0.30)
    assert s["cases"] == 3 and len(rows) == 3 and len(s["cmc"]) == 3


@pytest.mark.parametrize("only_ref, only_other, p", [(0, 0, 1.0), (6, 0, 0.03125), (5, 1, 0.21875), (1, 5, 0.21875)])
def test_mcnemar_exact(only_ref, only_other, p):
    cases = [f"c{i}" for i in range(only_ref + only_other + 3)]
    ref = {c: i < only_ref for i, c in enumerate(cases)}
    other = {c: only_ref <= i < only_ref + only_other for i, c in enumerate(cases)}
    r = fid.mcnemar_exact(ref, other)
    assert r == {"only_reference": only_ref, "only_this": only_other, "p": p}


# ------------------------------- perturbed gallery -------------------------------------------

def test_perturbation_is_deterministic_and_in_range():
    for case in ("CASE_01", "CASE_02", "CASE_03", "CASE_04"):
        pitch, yaw, roll = fid.perturbation_angles(case)
        assert (pitch, yaw, roll) == fid.perturbation_angles(case)
        assert 4 <= abs(pitch) <= 8 and 4 <= abs(yaw) <= 8 and 0 <= abs(roll) <= 3
    assert fid.perturbation_angles("CASE_01") != fid.perturbation_angles("CASE_02")


def _image():
    img = sitk.GetImageFromArray(head())
    img.SetSpacing((1.0, 1.0, 2.0))
    region = np.zeros(sitk.GetArrayFromImage(img).shape, bool)
    region[15:25, 5:30, 15:45] = True
    return img, region


def test_perturbed_gallery_grid_values_and_repeatability():
    img, region = _image()
    hu, moved, axes = fid.perturbed_gallery(img, region, "CASE_01")
    assert axes.spacing == (1.0, 1.0, 1.0) and hu.shape == moved.shape == (79, 60, 60)
    assert hu.min() >= -1000.0 and hu.max() <= 40.0001 and hu[0, 0, 0] == pytest.approx(-1000.0)
    assert moved.any()
    again, moved_again, _ = fid.perturbed_gallery(img, region, "CASE_01")
    assert np.array_equal(hu, again) and np.array_equal(moved, moved_again)
    other, _, _ = fid.perturbed_gallery(img, region, "CASE_02")
    assert not np.array_equal(hu, other)


# ------------------------------- whole run, cache --------------------------------------------

@pytest.fixture
def cohort(tmp_path):
    """Three cases; a stub model turns the mean of the image into an embedding per case and variant."""
    rows, seen = [], []
    for k, case in enumerate(("CASE_01", "CASE_02", "CASE_03")):
        hu = head()
        hu[hu > -1000] = 10.0 * k
        img = sitk.GetImageFromArray(hu)
        img.SetSpacing((1.0, 1.0, 2.0))
        mask = sitk.GetImageFromArray((hu > -1000).astype(np.uint8))
        mask.CopyInformation(img)
        ct, m, air = tmp_path / f"{case}.nii.gz", tmp_path / f"{case}_m.nii.gz", tmp_path / f"{case}_air.nii.gz"
        sitk.WriteImage(img, str(ct))
        sitk.WriteImage(mask, str(m))
        sitk.WriteImage(img, str(air))
        rows.append({"case": case, "gallery": ct.name, "gallery_mask": m.name, "mask": m.name, "air": air.name,
                     "original": ct.name})
    path = tmp_path / "cases.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path, seen


@pytest.fixture
def model(monkeypatch):
    calls = {"render": [], "detect": 0}

    def render(hu, region, axes, skin_hu=-300.0, yaws=(-30, 0, 30)):
        calls["render"].append((skin_hu, tuple(yaws), hu.shape))
        level = float(hu[hu > -500].mean())
        return {y: np.full((4, 4, 3), level) for y in yaws}

    def detect(image):
        calls["detect"] += 1
        k = float(image.mean()) / 10.0
        return {"embedding": np.cos(np.arange(1, 6) * (k + 0.3) * 1.3).tolist(), "face_confidence": 0.99}

    monkeypatch.setattr(defacing.reid, "render_views", render)
    monkeypatch.setattr(defacing.reid, "detect_face", detect)
    monkeypatch.setattr(defacing.reid, "check_face_check", lambda: None)
    return calls


def test_same_gallery_run_and_outputs(cohort, model, tmp_path):
    path, _ = cohort
    assert fid.main([str(path), "--out", str(tmp_path / "r" / "id")]) == 0
    s = json.loads((tmp_path / "r" / "id_summary.json").read_text())
    assert s["air"]["rank1_rate"] == 1.0 and s["_gallery"]["variant"] == "same"
    assert s["original"]["mcnemar_vs_air"] == {"only_reference": 0, "only_this": 0, "p": 1.0}
    assert "mcnemar_vs_air" not in s["air"]
    assert all(r[0] == -300.0 and r[1] == (-30, 0, 30) for r in model["render"])
    scores = json.loads((tmp_path / "r" / "id_scores.json").read_text())
    assert len(scores["air"]["impostor"]) == 6


def test_perturbed_only_changes_the_gallery(cohort, model, tmp_path):
    path, _ = cohort
    assert fid.main([str(path), "--out", str(tmp_path / "p"), "--gallery-variant", "perturbed"]) == 0
    gallery = [r for r in model["render"] if r[1] == (-45, -15, 15, 45)]
    probes = [r for r in model["render"] if r[1] == (-30, 0, 30)]
    assert len(gallery) == 3 and len(probes) == 6
    assert all(r[0] == -200.0 and r[2] == (79, 60, 60) for r in gallery)
    assert all(r[0] == -300.0 and r[2] == (40, 60, 60) for r in probes)
    assert json.loads((tmp_path / "p_summary.json").read_text())["_gallery"]["perturbation"]["blur_sigma_mm"] == 0.7


def test_cache_is_used_and_keyed_by_variant(cohort, model, tmp_path):
    path, _ = cohort
    cache = tmp_path / "cache" / "emb.json"
    args = [str(path), "--out", str(tmp_path / "c"), "--cache", str(cache)]
    assert fid.main(args) == 0
    renders = len(model["render"])
    assert fid.main(args) == 0
    assert len(model["render"]) == renders                                  # nothing recomputed
    assert fid.main([*args, "--gallery-variant", "perturbed"]) == 0
    assert len(model["render"]) == renders + 3                              # only the three galleries are new
    keys = list(json.loads(cache.read_text()))
    assert sum("|perturbed|CASE_01|" in k for k in keys) == 1
    assert sum(k.endswith("|same") for k in keys) == 6                      # the original probes are the gallery files
    assert '"blur_sigma_mm": 0.7' in next(k for k in keys if "perturbed" in k)


def test_cache_key_changes_with_perturbation_parameters(monkeypatch):
    before = fid.cache_key("a.nii.gz", "m.nii.gz", 260.0, "perturbed", "CASE_01")
    monkeypatch.setitem(fid.PERTURB, "blur_sigma_mm", 1.0)
    assert before != fid.cache_key("a.nii.gz", "m.nii.gz", 260.0, "perturbed", "CASE_01")
    assert fid.cache_key("a.nii.gz", "m.nii.gz", 260.0, "perturbed", "CASE_02") != fid.cache_key(
        "a.nii.gz", "m.nii.gz", 260.0, "perturbed", "CASE_01")
