"""RT structure sets: ROI classification and removal, UID remapping, contour rasterisation."""
import re

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import phantom as ph
from defacing.common import Options
from defacing.rt import (Grid, RegionStats, classify_roi, fill_polygon, plan_roi_removal, referenced_series,
                         remap_uids, roi_slices, roi_table, rtstruct_image_refs, split_rois)


@pytest.mark.parametrize("name, kind, expected", [
    # removed: Body/External (by type or by the whole name) ...
    ("whatever", "EXTERNAL", "body"), ("BODY", "", "body"), ("Body_1", "ORGAN", "body"), ("BODY (1)", "", "body"),
    ("External", "", "body"), ("Outline", "", "body"), ("Körper", "", "body"), ("KOERPER", "", "body"),
    ("Außenkontur", "", "body"), ("Aussenkontur", "", "body"), ("Patient", "", "body"),
    ("Patientenkontur", "", "body"), ("Körperkontur", "", "body"), ("  body  ", "", "body"),
    # ... and Skin/Haut, optionally with a number or thickness
    ("Skin", "ORGAN", "skin"), ("SKIN 5mm", "", "skin"), ("Haut_3mm", "", "skin"), ("Hautkontur", "", "skin"),
    ("skin-2", "", "skin"), ("Skin 0.5 cm", "", "skin"),
    # kept but reported: names that only contain the words, bolus, fixation devices
    ("Body-PTV", "PTV", "unclear"), ("z_Skin_ring", "", "unclear"), ("Glaskörper_L", "ORGAN", "unclear"),
    ("Wirbelkörper", "", "unclear"), ("Vertebral_Body", "", "unclear"), ("Maske", "", "unclear"),
    ("Bolus", "", "unclear"), ("anything", "BOLUS", "unclear"), ("Headrest", "FIXATION", "unclear"),
    ("Thermoplastic shell", "", "unclear"), ("Immobilisation", "", "unclear"), ("External_ROI", "", "unclear"),
    # kept, nothing to report
    ("PTV_60", "PTV", ""), ("GTV", "GTV", ""), ("Parotid_L", "ORGAN", ""), ("Brain", "ORGAN", ""),
    ("Couch", "SUPPORT", ""),
])
def test_classify_roi(name, kind, expected):
    assert classify_roi(name, kind) == expected


ROIS = {1: ("BODY", "EXTERNAL"), 2: ("Skin", ""), 3: ("PTV_60", "PTV"), 4: ("Body-PTV", ""), 5: ("Lens_L", "")}


def test_plan_roi_removal_defaults():
    plan = plan_roi_removal(ROIS, Options())
    assert plan.remove == {1: "BODY", 2: "Skin"} and plan.has_body and plan.unclear == ["Body-PTV"]


def test_plan_roi_removal_options():
    assert set(plan_roi_removal(ROIS, Options(keep_body=True)).remove) == {2}
    assert set(plan_roi_removal(ROIS, Options(keep_skin=True)).remove) == {1}
    plan = plan_roi_removal(ROIS, Options(remove_roi=[re.compile("body-ptv", re.I), re.compile("lens.*", re.I)]))
    assert set(plan.remove) == {1, 2, 4, 5} and plan.unclear == []
    # the pattern must match the WHOLE name
    assert 3 not in plan_roi_removal(ROIS, Options(remove_roi=[re.compile("PTV", re.I)])).remove
    assert not plan_roi_removal({3: ("PTV", "PTV")}, Options()).has_body


@pytest.fixture
def rtstruct(tmp_path):
    ct = ph.write_ct_series(tmp_path / "CT", shape=(12, 40, 40), spacing=(6.0, 6.0, 12.0))
    ph.write_rtstruct(tmp_path / "RS.dcm", ct)
    return ct, pydicom.dcmread(tmp_path / "RS.dcm")


def test_roi_table_and_split_rois(rtstruct):
    ct, ds = rtstruct
    assert roi_table(ds) == {1: ("BODY", "EXTERNAL"), 2: ("PTV_60", "PTV"), 3: ("Eye_L", "ORGAN")}
    old_uid = ds.SOPInstanceUID
    removed = split_rois(ds, {1})
    assert list(roi_table(ds)) == [2, 3] and list(roi_table(removed)) == [1]
    for seq in ("StructureSetROISequence", "ROIContourSequence", "RTROIObservationsSequence"):
        assert len(ds[seq].value) == 2 and len(removed[seq].value) == 1
    assert removed.SOPInstanceUID != old_uid and removed.file_meta.MediaStorageSOPInstanceUID == removed.SOPInstanceUID
    assert removed.SeriesInstanceUID != ds.SeriesInstanceUID
    assert ds.SOPInstanceUID == old_uid


def test_remap_uids_everywhere_including_nested_and_multivalued(rtstruct):
    ct, ds = rtstruct
    uid_map = {u: f"9.9.{i}" for i, u in enumerate(ct.sop_uids)}
    uid_map[ct.series_uid] = "9.8.1"
    ds.add_new(0x00081115, "SQ", Sequence([Dataset()]))          # ReferencedSeriesSequence
    ds.ReferencedSeriesSequence[0].add_new(0x00081160, "UI", [ct.sop_uids[0], "1.2.3"])  # multi-valued UI
    before = rtstruct_image_refs(ds)
    n = remap_uids(ds, uid_map)
    assert before == set(ct.sop_uids)
    assert rtstruct_image_refs(ds) == {uid_map[u] for u in ct.sop_uids}
    assert referenced_series(ds) == {"9.8.1"}
    assert list(ds.ReferencedSeriesSequence[0][0x00081160].value) == ["9.9.0", "1.2.3"]
    assert ds.StudyInstanceUID == ct.study_uid                      # not in the map: unchanged
    assert n >= len(ct.sop_uids) * 2


def test_grid_to_index_roundtrip_oblique():
    d = np.array([[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]])
    grid = Grid(origin=np.array([10.0, -20.0, 5.0]), spacing=np.array([0.8, 0.8, 3.0]), direction=d,
                size=(10, 10, 10))
    idx = np.array([[0, 0, 0], [3, 4, 5], [9.5, 2.25, 1]])
    world = (grid.origin[:, None] + (d * grid.spacing) @ idx.T).T
    assert np.allclose(grid.to_index(world), idx)


def test_fill_polygon_square_and_triangle():
    sq = fill_polygon(np.array([0.6, 4.4, 4.4, 0.6]), np.array([0.6, 0.6, 3.4, 3.4]), (6, 6))
    assert sq.sum() == 12 and sq[1:4, 1:5].all()
    tri = fill_polygon(np.array([0.0, 10.0, 0.0]), np.array([0.0, 0.0, 10.0]), (12, 12))
    assert tri[0, :11].all() and not tri[9, 3] and tri[9, 1]
    assert not fill_polygon(np.array([-5.0, -1.0, -1.0]), np.array([-5.0, -5.0, -1.0]), (4, 4)).any()


def test_roi_slices_rasterise_contours_and_respect_holes(rtstruct):
    import SimpleITK as sitk
    ct, ds = rtstruct
    img = sitk.ReadImage([str(p) for p in ct.paths])
    grid = Grid.of(img)
    slices = dict(roi_slices(ds, grid))
    body = slices[1]
    assert set(body) == {k for k in range(12) if abs(ct.origin[2] + 12.0 * k) < 0.95 * ph.HEAD_R[2]}
    z_mid = min(body, key=lambda k: abs(ct.origin[2] + 12.0 * k))
    area = body[z_mid].sum() * 36.0
    expected = np.pi * ph.HEAD_R[0] * ph.HEAD_R[1] * (1 - ((ct.origin[2] + 12.0 * z_mid) / ph.HEAD_R[2]) ** 2)
    assert abs(area - expected) / expected < 0.1
    # a second contour on the same slice inside the first is a hole
    item = ds.ROIContourSequence[1].ContourSequence[0]
    inner = [float(v) for v in item.ContourData]
    hole = Dataset()
    hole.ContourGeometricType = "CLOSED_PLANAR"
    hole.ContourData = [f"{v * (0.5 if i % 3 != 2 else 1):.3f}" for i, v in enumerate(inner)]
    ds.ROIContourSequence[1].ContourSequence.append(hole)
    z0 = next(iter(slices[2]))
    with_hole = dict(roi_slices(ds, grid))[2][z0]
    assert 0 < with_hole.sum() < slices[2][z0].sum()
    assert not with_hole[tuple(np.argwhere(slices[2][z0]).mean(axis=0).round().astype(int))]


def test_region_stats_overlap_and_distance():
    region = np.zeros((5, 20, 20), bool)
    region[2, :5, :] = True                      # rows 0-4 of slice 2
    stats = RegionStats(region, (3.0, 1.0, 1.0))
    inside = np.zeros((20, 20), bool)
    inside[3:7, 3:7] = True                      # 2 of 4 rows inside
    assert stats.measure({2: inside}) == (16, 8, 0.0)
    far = np.zeros((20, 20), bool)
    far[15, 10] = True
    assert stats.measure({2: far}) == (1, 0, pytest.approx(11.0))
    assert stats.measure({0: far})[2] == pytest.approx(np.hypot(6.0, 11.0))
    assert stats.measure({}) == (0, 0, stats.cap_mm)
    assert RegionStats(np.zeros_like(region), (1, 1, 1)).measure({2: far}) == (1, 0, 50.0)
