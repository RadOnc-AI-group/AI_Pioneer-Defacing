"""--clip-body and --mask-rtdose."""
import copy

import numpy as np
import pydicom
import pytest

import phantom as ph
from defacing.common import Options
from defacing.patient import deface_dicom_patient
from defacing.rt import Grid, clip_roi, fill_polygon, roi_slices, roi_table


def run(tmp_path, opts, rois=None, dose=None):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    ph.write_rtstruct(pat / "RS.PAT01.dcm", ct, rois or [("BODY", "EXTERNAL"), ("PTV_60", "PTV")])
    if dose is not None:
        dose = ph.write_dose(pat / "RD.PAT01.dcm", ct, **dose)
    out, qc = tmp_path / "out" / "PAT01", tmp_path / "qc" / "PAT01"
    res = deface_dicom_patient(pat, "PAT01", out, qc, opts)
    assert res.status == "done", res.error
    return res, ct, out, qc, dose


def grid_of(ct):
    direction = np.column_stack([ct.row, ct.col, ct.normal])
    return Grid(ct.origin, np.array(ct.spacing), direction, (ct.shape[2], ct.shape[1], ct.shape[0]))


def body_slices(rs, grid):
    number = next(n for n, (name, _) in roi_table(rs).items() if name == "BODY")
    return next(s for n, s in roi_slices(rs, grid) if n == number)


def test_clip_body(tmp_path, fake_ts):
    res, ct, out, qc, _ = run(tmp_path, Options(seed=1, clip_body=True))
    assert res.rois_clipped == ["BODY"] and not res.rois_removed
    rs = pydicom.dcmread(next((out / "rt").iterdir()))
    assert "BODY" in [n for n, _ in roi_table(rs).values()]
    grid = grid_of(ct)
    hu, world, _ = ph.read_series_hu(out / "dcm")
    region = (hu != ct.hu)                                          # the voxels that were replaced
    clipped = body_slices(rs, grid)
    original = body_slices(pydicom.dcmread(next((tmp_path / "in" / "PAT01").glob("RS*"))), grid)
    assert clipped, "the Body contour is gone"
    for z, mask in clipped.items():
        assert not (mask & region[z]).any(), f"the clipped Body lies in the de-faced region on slice {z}"
        assert np.array_equal(mask, original[z] & ~region[z]) or not region[z].any()
    # the nose is in the original outline only on the slices of the face
    nose = ph.in_nose(world) & (world[..., 1] < -100)
    z = int(np.flatnonzero(nose.any(axis=(1, 2)))[0])
    assert not (clipped[z] & nose[z]).any()
    assert list((qc / "removed_rois").glob("BEFORE_CLIP_*"))        # the original, for checks


def test_no_original_vertices_remain_on_clipped_slices(tmp_path, fake_ts):
    res, ct, out, _, _ = run(tmp_path, Options(seed=1, clip_body=True))
    rs = pydicom.dcmread(next((out / "rt").iterdir()))
    orig = pydicom.dcmread(next((tmp_path / "in" / "PAT01").glob("RS*")))
    hu, world, _ = ph.read_series_hu(out / "dcm")
    region = hu != ct.hu
    grid = grid_of(ct)

    def points(ds):
        out_points = {}
        number = next(n for n, (name, _) in roi_table(ds).items() if name == "BODY")
        item = next(r for r in ds.ROIContourSequence if int(r.ReferencedROINumber) == number)
        for c in item.ContourSequence:
            p = np.asarray(c.ContourData, float).reshape(-1, 3)
            z = int(round(float(np.median(grid.to_index(p)[:, 2]))))
            out_points.setdefault(z, []).append(p)
        return out_points

    new, old = points(rs), points(orig)
    assert len(new) == len(old)
    for z in new:
        if region[z].any():
            have = {tuple(v) for p in new[z] for v in np.round(p, 3)}
            assert not (have & {tuple(v) for p in old[z] for v in np.round(p, 3)})
        else:
            assert np.array_equal(np.vstack(new[z]), np.vstack(old[z]))     # unchanged slice


def test_clip_body_with_soft_tissue_fill_includes_the_new_front(tmp_path, fake_ts):
    res, ct, out, _, _ = run(tmp_path, Options(seed=1, clip_body=True, fill="soft-tissue"))
    rs = pydicom.dcmread(next((out / "rt").iterdir()))
    hu, world, _ = ph.read_series_hu(out / "dcm")
    region = hu != ct.hu
    clipped = body_slices(rs, grid_of(ct))
    tissue = region & (hu > -500)                                   # in this phantom the new front adds very little
    for z, mask in clipped.items():
        assert not (mask & region[z] & ~tissue[z]).any()              # air of the fill is outside the Body
        assert (mask & tissue[z]).sum() == tissue[z].sum()


def test_clip_includes_added_tissue():
    grid = small_grid()
    ds = contour_ds([(3.0, [(4, 4), (32, 4), (32, 32), (4, 32)])])
    remove = np.zeros((4, 20, 20), bool)
    remove[1, :8, :] = True                                         # the front is removed
    add = np.zeros((4, 20, 20), bool)
    add[1, 4:8, 6:14] = True                                        # the fill wrote tissue there again
    before = next(roi_slices(ds, grid))[1][1].copy()
    clip_roi(ds, 1, grid, remove, add)
    assert np.array_equal(next(roi_slices(ds, grid))[1][1], (before & ~remove[1]) | add[1])


def test_clip_body_without_its_series_removes_it(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ph.write_ct_series(pat / "CT")
    other = ph.write_ct_series(tmp_path / "other", seed=3)
    ph.write_rtstruct(pat / "RS.dcm", other, [("BODY", "EXTERNAL")])
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1, clip_body=True))
    assert res.status == "done" and res.rois_removed == ["BODY"] and not res.rois_clipped
    assert any("removed instead of clipped" in f for f in res.flags)


def small_grid():
    return Grid(np.zeros(3), np.array([2.0, 2.0, 3.0]), np.eye(3), (20, 20, 4))


def contour_ds(points_per_slice):
    ds = pydicom.dataset.Dataset()
    roi = pydicom.dataset.Dataset()
    roi.ReferencedROINumber = 1
    seq = []
    for z, pts in points_per_slice:
        c = pydicom.dataset.Dataset()
        c.ContourGeometricType = "CLOSED_PLANAR"
        c.ContourData = [f"{v:.3f}" for p in pts for v in (p[0], p[1], z)]
        c.NumberOfContourPoints = len(pts)
        seq.append(c)
    roi.ContourSequence = pydicom.sequence.Sequence(seq)
    ds.ROIContourSequence = pydicom.sequence.Sequence([roi])
    ds.StructureSetROISequence = pydicom.sequence.Sequence()
    return ds


def test_clip_handles_holes_and_splits():
    grid = small_grid()
    square = [(4, 4), (32, 4), (32, 32), (4, 32)]
    ds = contour_ds([(3.0, square), (6.0, square)])
    remove = np.zeros((4, 20, 20), bool)
    remove[1, 8:12, 2:18] = True                                    # cuts the square in two
    remove[1, 3:6, 8:12] = True                                     # and a notch
    before = next(roi_slices(ds, grid))[1][1].copy()
    info = clip_roi(ds, 1, grid, remove)
    after = next(roi_slices(ds, grid))[1]
    assert info["slices_clipped"] == 1 and np.array_equal(after[1], before & ~remove[1])
    xs, ys = np.array([p[0] / 2 for p in square]), np.array([p[1] / 2 for p in square])
    assert np.array_equal(after[2], fill_polygon(xs, ys, (20, 20)))


def test_clip_drops_contours_outside_an_image_plane():
    grid = small_grid()
    ds = contour_ds([(3.0, [(4, 4), (32, 4), (32, 32), (4, 32)])])
    bad = copy.deepcopy(ds.ROIContourSequence[0].ContourSequence[0])
    bad.ContourData = [f"{v:.3f}" for v in (4, 4, 3, 32, 4, 9, 32, 32, 3)]    # tilted
    ds.ROIContourSequence[0].ContourSequence.append(bad)
    remove = np.zeros((4, 20, 20), bool)
    remove[1, 8:12, :] = True
    assert clip_roi(ds, 1, grid, remove)["contours_dropped"] == 1


def read_dose(path):
    ds = pydicom.dcmread(path)
    return ds, ds.pixel_array.reshape(int(ds.NumberOfFrames), ds.Rows, ds.Columns)


@pytest.mark.parametrize("descending", [False, True])
def test_mask_rtdose(tmp_path, fake_ts, descending):
    res, ct, out, _, (_, world) = run(tmp_path, Options(seed=1, mask_rtdose=True), dose={"descending": descending})
    assert res.rtdose_masked == 1 and res.rtdose_removed == 0
    ds, dose = read_dose(next(p for p in (out / "rt").iterdir() if p.name.startswith("RD")))
    assert "RTDOSE set to zero" in str(ds.DeidentificationMethod)
    face = ph.in_face(world)
    assert face.any() and (dose[face] == 0).all(), "dose left in the face"
    far = (world[..., 1] > -20) & (np.abs(world[..., 2]) < 50)      # well behind the face
    assert (dose[far] == 1000).all()


def test_mask_rtdose_keeps_the_dose_of_kept_targets(tmp_path, fake_ts):
    res, ct, out, _, (_, world) = run(tmp_path, Options(seed=1, mask_rtdose=True), [("PTV_nose", "PTV")], dose={})
    _, dose = read_dose(next(p for p in (out / "rt").iterdir() if p.name.startswith("RD")))
    inside = ph.in_box(world, ((-6, 6), (-98, -72), (-16, 6)))
    assert inside.any() and (dose[inside] == 1000).all()
    assert (dose[ph.in_face(world) & ~ph.in_box(world, ((-24, 24), (-120, -60), (-40, 30)))] == 0).all()


def test_mask_rtdose_without_a_matching_frame_is_not_copied(tmp_path, fake_ts):
    pat = tmp_path / "in" / "PAT01"
    ct = ph.write_ct_series(pat / "CT")
    other = ph.write_ct_series(tmp_path / "other", seed=2)
    ph.write_dose(pat / "RD.dcm", other)
    res = deface_dicom_patient(pat, "PAT01", tmp_path / "o", tmp_path / "q", Options(seed=1, mask_rtdose=True))
    assert res.status == "done" and res.rtdose_removed == 1 and res.rtdose_masked == 0
    assert any("no de-faced CT with its Frame of Reference" in f for f in res.flags)
    assert not list((tmp_path / "o").rglob("RD*"))


def test_dose_not_copied_by_default(tmp_path, fake_ts):
    res, _, out, _, _ = run(tmp_path, Options(seed=1), dose={})
    assert res.rtdose_removed == 1 and not [p for p in (out / "rt").iterdir() if p.name.startswith("RD")]
