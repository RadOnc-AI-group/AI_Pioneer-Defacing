"""The experimental template fit, on a synthetic head (FLAME itself is not in the repository)."""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from defacing.masks import Axes
from defacing.template import (EYE, FACE, NECK, REST, Pose, Template, fit_template, front_map, front_minus_skin, load_template,
                               sample_triangles, umeyama)

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("make_flame_template", REPO / "tools" / "make_flame_template.py")
converter = importlib.util.module_from_spec(spec)
sys.modules["make_flame_template"] = converter
spec.loader.exec_module(converter)

SPACING = 2.0
AXES = Axes(spacing=(SPACING, SPACING, SPACING), anterior=(1, True), si_axis=0, superior=True)   # (si, ap, lr)
BUMPS = (((0.0, 0.75, 0.66), 7.0, 0.35), ((0.0, -0.8, 0.2), 8.0, 0.4), ((1.0, -0.05, 0.0), 9.0, 0.22),
         ((-1.0, -0.05, 0.0), 9.0, 0.22), ((0.55, 0.55, 0.62), 4.0, 0.25), ((-0.55, 0.55, 0.62), 2.0, 0.25),
         ((0.0, 0.85, 0.42), 7.0, 0.14), ((0.85, 0.35, 0.45), -5.0, 0.2), ((-0.85, 0.35, 0.45), -5.0, 0.2),
         ((0.85, -0.35, -0.45), 7.0, 0.2), ((-0.85, -0.35, -0.45), 7.0, 0.2), ((0.78, 0.45, -0.15), 6.0, 0.18),
         ((-0.78, 0.45, -0.15), 6.0, 0.18))
NOSE = ((0.0, 0.97, -0.25), 26.0, 0.17)


def radius(u: np.ndarray, nose: float = 1.0, shape_noise: float = 0.0) -> np.ndarray:
    """A star-shaped head with a brow, an occiput, ears and a nose. u: unit vectors (lr, ap, si)."""
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
    r = 1.0 / np.maximum(np.sqrt((u[..., 0] / 74.0) ** 2 + (u[..., 1] / 96.0) ** 2 + (u[..., 2] / 112.0) ** 2), 1e-9)
    for direction, height, width in BUMPS + ((NOSE[0], NOSE[1] * nose, NOSE[2]),):
        d = np.asarray(direction, float)
        cos = np.clip((u @ (d / np.linalg.norm(d))), -1, 1)
        r = r + height * np.exp(-(np.arccos(cos) / width) ** 2)
    if shape_noise:
        r = r + shape_noise * np.sin(3.1 * u[..., 0] + 1.0) * np.cos(2.3 * u[..., 2] + 0.5)
    return r


def face_of(u: np.ndarray) -> np.ndarray:
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
    return (u[..., 1] > 0.45) & (u[..., 2] > -0.72) & (u[..., 2] < 0.30)


def make_template() -> Template:
    lat = np.linspace(0.0, np.pi, 90)
    lon = np.linspace(0.0, 2 * np.pi, 180, endpoint=False)
    th, ph = np.meshgrid(lat, lon, indexing="ij")
    u = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1, 3)
    vertices = u * radius(u)[..., None]
    n_lat, n_lon = len(lat), len(lon)
    index = lambda i, j: i * n_lon + (j % n_lon)
    tris = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            tris += [(index(i, j), index(i + 1, j), index(i + 1, j + 1)), (index(i, j), index(i + 1, j + 1), index(i, j + 1))]
    labels = np.full(len(vertices), REST, np.uint8)
    labels[face_of(u)] = FACE
    labels[u[:, 2] < -0.8] = NECK
    return Template(vertices, np.array(tris), labels, "test")


TEMPLATE = make_template()


@pytest.fixture(autouse=True)
def _quicker_fit(monkeypatch):
    """Fewer points than on a real scan, so that the tests stay fast."""
    monkeypatch.setattr("defacing.template.SAMPLE_MM", 3.0)
    monkeypatch.setattr("defacing.template.MAX_POINTS", 3000)


def make_patient(pose: Pose | None = None, *, shape_noise=0.0, mask_shell=False, fov_lr_mm=None, face_lift=0.0):
    """A voxelised head (with neck) under `pose`, and its face mask and de-faced region. Axes (si, ap, lr)."""
    pose = pose or Pose(1.0, np.eye(3), np.array([140.0, 130.0, 170.0]))
    shape = (170, 140, 140)
    s, a, l = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    world = np.stack([l, a, s], axis=-1) * SPACING
    x = pose.invert(world.reshape(-1, 3)).reshape(shape + (3,))
    head = np.linalg.norm(x, axis=-1) <= radius(x, shape_noise=shape_noise)
    neck = (x[..., 2] < -60) & (np.hypot(x[..., 0] / 55.0, (x[..., 1] + 10.0) / 60.0) <= 1.0)
    body = head | neck
    hu = np.where(body, 40.0, -1000.0)
    u = x.copy()
    u[..., 2] = u[..., 2] - face_lift
    face = body & face_of(u) & head
    if mask_shell:
        grown = binary_dilation(body, iterations=3)
        hu[grown & ~binary_dilation(body, iterations=2) & (x[..., 1] > 0)] = 40.0
    if fov_lr_mm is not None:
        centre = pose.translation[0]
        hu[np.abs(world[..., 0] - centre) > fov_lr_mm] = -1000.0
    region = binary_dilation(face, iterations=2)
    return hu, face, np.maximum.accumulate(region, axis=1), None                   # extended to the anterior edge


def face_error(fit, pose: Pose) -> float:
    face = TEMPLATE.vertices[TEMPLATE.labels == FACE]
    return float(np.sqrt(((fit.pose.apply(face * np.array([fit.lr_sign, 1, 1])) - pose.apply(face)) ** 2).sum(axis=1).mean()))


def rotation(deg, axis=0):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fit_of(*args, **kwargs):
    hu, face, region, _ = make_patient(*args, **kwargs)
    return fit_template(TEMPLATE, hu, face, region, AXES)


def test_umeyama_recovers_a_similarity_and_never_mirrors():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(200, 3)) * 30
    truth = Pose(1.07, rotation(12) @ rotation(-7, 2), np.array([3.0, -8.0, 21.0]))
    found = umeyama(source, truth.apply(source))
    assert found.scale == pytest.approx(1.07) and np.allclose(found.rotation, truth.rotation)
    mirrored = umeyama(source, truth.apply(source) * np.array([-1, 1, 1]))
    assert np.linalg.det(mirrored.rotation) == pytest.approx(1.0)


def test_pose_invert():
    pose = Pose(0.9, rotation(10), np.array([1.0, 2.0, 3.0]))
    x = np.array([[10.0, -4.0, 7.0]])
    assert np.allclose(pose.invert(pose.apply(x)), x)


def test_triangles_are_sampled_finely():
    vertices = np.array([[0, 0, 0], [12, 0, 0], [0, 12, 0]], float)
    points, normals = sample_triangles(vertices, np.array([[0, 1, 2]]), 1.5)
    assert np.allclose(np.abs(normals[:, 2]), 1.0) and len(points) == len(normals) and len(points) > 30 and points[:, :2].min() >= 0 and (points[:, :2].sum(axis=1) <= 12 + 1e-9).all()


def test_the_identity_pose_is_recovered():
    pose = Pose(1.0, np.eye(3), np.array([140.0, 130.0, 170.0]))
    fit = fit_of(pose)
    assert fit.accepted, fit.reason
    assert face_error(fit, pose) < 3.0 and fit.scale_change < 0.05 and fit.coverage >= 0.95


@pytest.mark.parametrize("scale, tilt, shift", [(1.06, 0, (0, 0, 0)), (0.95, 0, (6, -5, 4)), (1.0, 8, (0, 0, 0)), (1.0, -8, (2, 3, -3))])
def test_a_known_transform_is_recovered(scale, tilt, shift):
    pose = Pose(scale, rotation(tilt), np.array([140.0, 130.0, 170.0]) + np.array(shift, float))
    fit = fit_of(pose)
    assert fit.accepted, fit.reason
    assert face_error(fit, pose) < 3.5


@pytest.mark.parametrize("tilt", [15, -15])
def test_strong_head_tilt_gives_a_correct_pose_or_a_rejection(tilt):
    pose = Pose(1.0, rotation(tilt), np.array([140.0, 130.0, 170.0]))
    fit = fit_of(pose)
    assert not fit.accepted or face_error(fit, pose) < 5.0


def test_a_different_head_shape_is_still_placed_well():
    pose = Pose(1.0, np.eye(3), np.array([140.0, 130.0, 170.0]))
    fit = fit_of(pose, shape_noise=2.0)
    assert fit.accepted, fit.reason
    assert face_error(fit, pose) < 4.0


def test_a_thermoplastic_mask_does_not_become_the_skin():
    pose = Pose(1.0, np.eye(3), np.array([140.0, 130.0, 170.0]))
    fit = fit_of(pose, mask_shell=True)
    assert fit.accepted, fit.reason
    assert face_error(fit, pose) < 3.5


def test_a_narrow_field_of_view_gives_a_correct_pose_or_a_rejection():
    pose = Pose(1.0, np.eye(3), np.array([140.0, 130.0, 170.0]))
    fit = fit_of(pose, fov_lr_mm=55.0)
    assert not fit.accepted or face_error(fit, pose) < 5.0


def test_stray_parts_of_the_face_mask_are_ignored():
    hu, face, region, _ = make_patient()
    clean = fit_template(TEMPLATE, hu, face, region, AXES)
    face2, region2 = face.copy(), region.copy()
    face2[5:8, 135:138, 60:64] = True                       # a false detection far from the face, as on real scans
    region2[5:8, 130:, 58:66] = True
    stray = fit_template(TEMPLATE, hu, face2, region2, AXES)
    assert stray.accepted and stray.coverage == pytest.approx(clean.coverage)
    assert face_error(stray, Pose(1.0, np.eye(3), np.array([140.0, 130.0, 170.0]))) < 3.5


def test_voxels_inside_the_region_do_not_matter():
    hu, face, region, _ = make_patient()
    other = hu.copy()
    other[region] = np.random.default_rng(3).uniform(-1000, 1500, int(region.sum()))
    a, b = fit_template(TEMPLATE, hu, face, region, AXES), fit_template(TEMPLATE, other, face, region, AXES)
    assert np.array_equal(a.pose.rotation, b.pose.rotation) and np.array_equal(a.pose.translation, b.pose.translation)
    assert a.pose.scale == b.pose.scale


@pytest.mark.parametrize("flip, axes", [
    ((0,), Axes((SPACING,) * 3, (1, True), 0, False)),          # feet first
    ((1,), Axes((SPACING,) * 3, (1, False), 0, True)),          # prone
    ((2,), AXES),                                               # left-right the other way
])
def test_mirrored_images_give_the_same_fit(flip, axes):
    hu, face, region, _ = make_patient()
    a = fit_template(TEMPLATE, hu, face, region, AXES)
    b = fit_template(TEMPLATE, np.flip(hu, flip), np.flip(face, flip), np.flip(region, flip), axes)
    assert a.accepted and b.accepted
    assert b.rms_mm == pytest.approx(a.rms_mm, abs=0.5)
    front_a = a.front if 2 not in flip else a.front[:, ::-1]
    both = np.isfinite(front_a) & np.isfinite(b.front)
    assert both.mean() > 0.1 and np.abs(front_a[both] - b.front[both]).mean() < 3.0


def test_rejections():
    hu, face, region, _ = make_patient()
    assert fit_template(TEMPLATE, hu, np.zeros_like(face), region, AXES).reason == "empty face mask"
    assert "orientation" in fit_template(TEMPLATE, hu, face, region, Axes((2.0,) * 3, (0, True), 0)).reason
    air = np.full_like(hu, -1000.0)
    assert "too few" in fit_template(TEMPLATE, air, face, region, AXES).reason


@pytest.mark.xfail(strict=True, reason="known limit: a template of the wrong shape still fits the skin around the face")
def test_a_head_of_another_shape_is_rejected():
    wrong = Template(TEMPLATE.vertices * np.array([1.0, 0.72, 1.3]), TEMPLATE.triangles, TEMPLATE.labels, "x")
    hu, face, region, _ = make_patient()
    fit = fit_template(wrong, hu, face, region, AXES)
    assert not fit.accepted and fit.rms_mm is not None


def test_a_scan_that_ends_at_the_top_of_the_face_is_rejected():
    """Without a forehead a pose far from the truth fits the skin just as well, so the fit must not be trusted."""
    hu, face, region, _ = make_patient()
    top = int(np.flatnonzero(face.any(axis=(1, 2)))[-1])
    fit = fit_template(TEMPLATE, hu[:top + 1], face[:top + 1], region[:top + 1], AXES)
    assert not fit.accepted and "top of the face" in fit.reason and fit.above_face_mm == 0.0


def test_the_front_is_compared_with_the_real_skin():
    hu, face, region, _ = make_patient()
    fit = fit_template(TEMPLATE, hu, face, region, AXES)
    numbers = front_minus_skin(fit, hu, face, region, AXES)
    assert abs(numbers["skin_p50_mm"]) < 3.0 and numbers["skin_p10_mm"] <= numbers["skin_p50_mm"] <= numbers["skin_p90_mm"]


def test_the_report_holds_numbers_only():
    numbers = fit_of().numbers()
    assert set(numbers) == {"accepted", "reason", "lr_sign", "points", "rms_mm", "moved_mm", "scale", "scale_change",
                            "spread_mm", "above_face_mm", "ap_constraint", "si_constraint", "coverage"}
    assert all(isinstance(v, (bool, int, float, str, type(None))) for v in numbers.values())


def test_front_map_takes_the_most_anterior_surface_and_marks_no_template():
    v = np.array([[0, 10, 0], [40, 10, 0], [0, 10, 40], [0, 30, 0], [40, 30, 0], [0, 30, 40]], float)
    front = front_map(v, np.array([[0, 1, 2], [3, 4, 5]]), (30, 30), (2.0, 2.0, 2.0))
    assert front[5, 5] == pytest.approx(30.0) and np.isnan(front[25, 25])


def test_front_map_closes_a_narrow_groove():
    z, x = np.meshgrid(np.arange(0.0, 41.0), np.arange(0.0, 41.0), indexing="ij")
    ap = 50.0 - 8.0 * (np.abs(z - 20.0) < 1.5)
    v = np.stack([x.ravel(), ap.ravel(), z.ravel()], axis=1)
    idx = lambda i, j: i * 41 + j
    tris = []
    for i in range(40):
        for j in range(40):
            tris += [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)), (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))]
    front = front_map(v, np.array(tris), (41, 41), (1.0, 1.0, 1.0))
    assert front[20, 20] > 48.0 and np.isfinite(front[20, 20])


def test_template_file_round_trip_and_checks(tmp_path):
    path = tmp_path / "t.npz"
    np.savez(path, vertices=TEMPLATE.vertices, triangles=TEMPLATE.triangles, labels=TEMPLATE.labels)
    loaded = load_template(path)
    assert len(loaded.vertices) == len(TEMPLATE.vertices) and len(loaded.sha256) == 64
    np.savez(path, vertices=TEMPLATE.vertices / 1000.0, triangles=TEMPLATE.triangles, labels=TEMPLATE.labels)
    with pytest.raises(ValueError, match="millimetres"):
        load_template(path)


def test_a_template_file_with_pickled_objects_is_refused(tmp_path):
    path = tmp_path / "t.npz"
    np.savez(path, vertices=np.array([object()], dtype=object), triangles=TEMPLATE.triangles, labels=TEMPLATE.labels)
    with pytest.raises(ValueError):
        load_template(path)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class ChumpyLike:
    def __init__(self, x):
        self.x = x

    def __reduce__(self):
        return (_rebuild, (), {"x": self.x})


def _rebuild():
    return converter._Stub()


def flame_pickle(path: Path, wrap=False):
    """A file shaped like FLAME: metres, x left-right, y up, z front, with two separate eye parts."""
    head = (TEMPLATE.vertices / 1000.0)[:, [0, 2, 1]]
    tris = TEMPLATE.triangles
    eye = np.array([[0.03, 0.03, 0.08], [0.04, 0.03, 0.08], [0.03, 0.04, 0.08]])
    v = np.concatenate([head, eye])
    f = np.concatenate([tris, [[len(head), len(head) + 1, len(head) + 2]]])
    with open(path, "wb") as fh:
        pickle.dump({"v_template": v, "f": f, "shapedirs": np.zeros(3)}, fh, protocol=2)
    return v, f


def test_converter_reads_metres_and_changes_the_axes(tmp_path):
    source = tmp_path / "flame.pkl"
    v, f = flame_pickle(source)
    vertices, triangles = converter.read_flame(source)
    assert np.allclose(vertices, v[:, [0, 2, 1]] * 1000.0) and np.array_equal(triangles, f)


def test_converter_labels_and_output(tmp_path, capsys):
    source, out = tmp_path / "flame.pkl", tmp_path / "t.npz"
    flame_pickle(source)
    assert converter.main([str(source), str(out)]) == 0
    template = load_template(out)
    assert (template.labels == EYE).sum() == 3
    tip = template.vertices[template.labels != EYE][np.argmax(template.vertices[template.labels != EYE][:, 1])]
    nearest = int(np.argmin(np.linalg.norm(template.vertices - tip, axis=1)))
    assert template.labels[nearest] == FACE
    assert template.labels[np.argmin(template.vertices[:, 2])] == NECK
    assert template.labels[np.argmax(template.vertices[:, 2])] == REST
    notice = (tmp_path / "t.npz.NOTICE.txt").read_text()
    assert "Creative Commons" in notice and "misleading" in notice and "Learning a" in notice


def test_converter_refuses_code_in_the_pickle(tmp_path):
    source = tmp_path / "evil.pkl"
    marker = tmp_path / "ran"
    source.write_bytes(b"cos\nsystem\n(S'touch " + str(marker).encode() + b"'\ntR.")
    with pytest.raises(pickle.UnpicklingError):
        converter.read_flame(source)
    assert not marker.exists()


def test_converter_replaces_chumpy_by_a_stub():
    class Unpickler(converter._AllowList):
        pass
    assert Unpickler.find_class(Unpickler(_empty()), "chumpy.ch", "Ch") is converter._Stub
    assert Unpickler.find_class(Unpickler(_empty()), "scipy.sparse.csc", "csc_matrix") is converter._Stub


def _empty():
    import io
    return io.BytesIO(b"")


def test_converter_reads_numbers_from_a_stub():
    stub = converter._Stub()
    stub.__setstate__({"x": np.arange(3.0)})
    assert np.array_equal(converter._array(stub), np.arange(3.0))
    with pytest.raises(ValueError):
        converter._array(converter._Stub())


def test_converter_refuses_a_file_not_in_metres(tmp_path):
    source = tmp_path / "flame.pkl"
    with open(source, "wb") as fh:
        pickle.dump({"v_template": TEMPLATE.vertices, "f": TEMPLATE.triangles}, fh, protocol=2)
    with pytest.raises(ValueError, match="metres"):
        converter.read_flame(source)
