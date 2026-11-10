"""
EXPERIMENTAL: fit a mean head surface (FLAME 2023 Open) to what remains of the patient, for
--fill template. tools/template_fit_report.py measures how often the fit can be trusted.

Only the mean shape with a neutral expression is used, with 7 degrees of freedom (scale,
rotation, translation). No HU value inside the de-faced region is used: the pose comes
from the skin outside the region. The face mask only gives a rough starting position.

Coordinates of a template and of a patient: (left-right, anterior, superior) in mm. The
left-right index of an image may run either way, so both mirror images are tried.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .common import Options, PatientError
from .fill import TemplateFront
from .masks import BODY_HU, Axes
from .oars import thin_free_disk
from .qc import _oriented

REST, FACE, NECK, EYE = 0, 1, 2, 3      # per-vertex labels of a template

DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "data" / "flame_template.npz"
SYNTHETIC_NOTE = "Synthetic face inserted (FLAME 2023 Open, CC-BY-4.0)"     # at most 64 characters: DICOM LO

SAMPLE_MM = 1.5             # template triangles are sampled this finely: its vertices lie several mm apart
NEAR_FACE_MM = 50.0         # patient surface points: at most this far from the face mask ...
EDGE_MM = 3.0               # ... and at least this far from the de-faced region
MAX_POINTS = 10_000
MIN_POINTS = 500
TRIM = 0.70                 # share of the closest point pairs used in each step
COARSE_STEPS = 10           # point-to-point steps of the fit ...
FINE_STEPS = 80             # ... followed by at most this many point-to-plane steps
MAX_SCALE_CHANGE = 0.10     # rejected if the head is this much larger or smaller than the mean head
MAX_TILT_DEG = 25.0         # ... or turned further than this from straight
MAX_RMS_MM = 4.0            # ... or the fit is worse than this
MAX_SPREAD_MM = 5.0         # ... or starts that fit about equally well put the face in different places
EQUAL_FIT = 1.25            # "about equally well": RMS up to this factor of the best
MIN_COVERAGE = 0.95
MIN_ABOVE_FACE_MM = 40.0    # scan needed above the face: without a forehead the skin at the sides leaves the pose
                            # open, and a pose up to 87 mm off still fits well, so nothing else would show it
# Flag only (the report reads the real face, so it can never change the fill): the front is reported as not fitting
# when it is further than this from the real skin. Empirical starting values, see the README.
MAX_SKIN_MEDIAN_MM = 5.0    # median over the columns of the face
MAX_SKIN_P90_MM = 15.0      # 10th and 90th percentile
CLOSING_MM = 3.0            # removes the groove of the mouth slit from the front surface
# starting positions: head tilt (degrees), anterior shift, superior shift (mm). A grid, because the face mask says
# only roughly where the face is, and a scan that ends near the forehead moves that estimate.
STARTS = ((8.0, 0.0, 0.0), (-8.0, 0.0, 0.0),
          *((0.0, float(ap), float(si)) for ap in (-20, 0, 20) for si in (-30, -15, 0, 15, 30)))


@dataclass(frozen=True)
class Template:
    vertices: np.ndarray    # (n, 3) in mm: left-right, anterior, superior
    triangles: np.ndarray   # (m, 3) vertex indices
    labels: np.ndarray      # REST, FACE, NECK or EYE per vertex
    sha256: str


def load_template(path: Path) -> Template:
    """A template written by tools/make_flame_template.py. Never unpickles."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        vertices, triangles, labels = data["vertices"].astype(np.float64), data["triangles"].astype(np.int64), data["labels"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("template: vertices and triangles must have three columns")
    if len(labels) != len(vertices) or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("template: labels or triangles do not match the vertices")
    if not (100.0 < float(np.ptp(vertices[:, 2])) < 500.0):
        raise ValueError("template: vertices must be in millimetres")
    if not (labels == REST).any() or not (labels == FACE).any():
        raise ValueError("template: needs vertices labelled face and rest of head")
    return Template(vertices, triangles, labels.astype(np.uint8), digest)


def sample_triangles(vertices: np.ndarray, triangles: np.ndarray, spacing: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points on the triangles about `spacing` (default SAMPLE_MM) apart, in a fixed order, and their unit normals."""
    spacing = spacing or SAMPLE_MM
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=0)
    steps = np.clip(np.ceil(longest / spacing).astype(int), 1, 40)
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    out, normals = [], []
    for n in np.unique(steps):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u, v = (i[keep] / n)[None, :, None], (j[keep] / n)[None, :, None]
        pick = steps == n
        out.append((a[pick][:, None] * (1 - u - v) + b[pick][:, None] * u + c[pick][:, None] * v).reshape(-1, 3))
        normals.append(np.repeat(normal[pick], int(keep.sum()), axis=0))
    return np.concatenate(out), np.concatenate(normals)


@dataclass(frozen=True)
class Pose:
    """template point -> patient point: scale * rotation @ x + translation"""
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scale * x @ self.rotation.T + self.translation

    def invert(self, y: np.ndarray) -> np.ndarray:
        return (y - self.translation) @ self.rotation / self.scale


def umeyama(source: np.ndarray, target: np.ndarray) -> Pose:
    """The similarity transform with a proper rotation that best maps source onto target."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(t.T @ s / len(s))
    fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        fix[2, 2] = -1.0
    rotation = u @ fix @ vt
    variance = float((s ** 2).sum() / len(s))
    scale = float(np.trace(np.diag(d) @ fix) / variance)
    return Pose(scale, rotation, mu_t - scale * rotation @ mu_s)


def _pairs(points: np.ndarray, tree: cKDTree, cloud: np.ndarray, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """For each patient point: index of the closest template point, and its distance (mm)."""
    _, idx = tree.query(pose.invert(points))
    return idx, np.linalg.norm(pose.apply(cloud[idx]) - points, axis=1)


def _rotation_vector(w: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = w / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _plane_step(points: np.ndarray, target: np.ndarray, normals: np.ndarray, pose: Pose) -> tuple[Pose, float]:
    """One linearised point-to-plane step of the similarity transform. Returns the pose and the size of the step."""
    arm = target - pose.translation
    a = np.c_[np.cross(arm, normals), normals, (arm * normals).sum(axis=1)]
    b = ((points - target) * normals).sum(axis=1)
    w0, w1, w2, t0, t1, t2, ds = np.linalg.lstsq(a, b, rcond=None)[0]
    w, dt = np.array([w0, w1, w2]), np.array([t0, t1, t2])
    return (Pose(pose.scale * (1.0 + ds), _rotation_vector(w) @ pose.rotation, pose.translation + dt),
            max(float(np.linalg.norm(w)) * 100.0, float(np.linalg.norm(dt)), abs(float(ds)) * 100.0))


def _icp(points: np.ndarray, tree: cKDTree, cloud: np.ndarray, normals: np.ndarray, pose: Pose) -> tuple[Pose, float]:
    """
    Trimmed ICP from patient points to template points: a few point-to-point steps (robust
    from a rough start), then point-to-plane steps, because point-to-point alone stops
    early when sliding along smooth skin. Returns the pose and the RMS (mm) of the kept pairs.
    """
    n_keep = max(7, int(TRIM * len(points)))
    for _ in range(COARSE_STEPS):
        idx, distance = _pairs(points, tree, cloud, pose)
        keep = np.argsort(distance, kind="stable")[:n_keep]
        pose = umeyama(cloud[idx[keep]], points[keep])
    for _ in range(FINE_STEPS):
        idx, distance = _pairs(points, tree, cloud, pose)
        keep = np.argsort(distance, kind="stable")[:n_keep]
        pose, step = _plane_step(points[keep], pose.apply(cloud[idx[keep]]), normals[idx[keep]] @ pose.rotation.T, pose)
        if step < 1e-3:
            break
    _, distance = _pairs(points, tree, cloud, pose)
    return pose, float(np.sqrt((np.sort(distance)[:n_keep] ** 2).mean()))


def main_component(mask: np.ndarray) -> np.ndarray:
    """The largest 26-connected part: stray false detections of the face mask are ignored."""
    labels, found = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if found <= 1:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


@dataclass(frozen=True)
class Patient:
    """Volumes with axes (superior, anterior, left-right); indices increase superiorly and anteriorly."""
    hu: np.ndarray
    face: np.ndarray            # the main part of the face mask, without margin or extension
    region: np.ndarray          # the de-faced region
    spacing: tuple[float, float, float]


def orient(hu: np.ndarray, face: np.ndarray, region: np.ndarray, axes: Axes) -> Patient | None:
    """None if the orientation of the image is unknown."""
    a, b, c = _oriented(hu, axes), _oriented(face, axes), _oriented(region, axes)
    if a is None or b is None or c is None:
        return None
    return Patient(a[0], main_component(b[0]), c[0], a[1])


def surface_points(p: Patient) -> tuple[np.ndarray, dict]:
    """
    Skin points of what remains of the patient near the face, in (left-right, anterior,
    superior) mm, and rough head measures for the starting position.
    Not skin: the cut surface of the region, air cavities it opens, and thin sheets such
    as a thermoplastic mask, which touch the skin and would otherwise become its outside.
    """
    s_si, s_ap, s_lr = p.spacing
    slices = np.flatnonzero(p.face.any(axis=(1, 2)))
    reach = int(np.ceil(NEAR_FACE_MM / s_si))
    z0, z1 = max(0, slices[0] - reach), min(len(p.hu), slices[-1] + reach + 1)
    face, region = p.face[z0:z1], p.region[z0:z1]
    kept = (p.hu[z0:z1] > BODY_HU) & ~region
    disk = thin_free_disk((s_ap, s_lr))
    for k in range(len(kept)):
        kept[k] = ndi.binary_opening(kept[k], structure=disk)
    labels, found = ndi.label(kept)
    if not found:
        return np.empty((0, 3)), {}
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body = labels == int(np.argmax(sizes))

    # outside air: reached from the sides of the image without crossing the region
    background = ~body & ~region
    labels, _ = ndi.label(background)
    border = np.unique(np.concatenate([labels[:, 0].ravel(), labels[:, -1].ravel(), labels[:, :, 0].ravel(),
                                       labels[:, :, -1].ravel()]))
    outside = np.isin(labels, border[border > 0])
    skin = body & ndi.binary_dilation(outside)

    # much of a face mask is air: only the slices where it holds tissue tell where the face is
    face_tissue = face & (p.hu[z0:z1] > BODY_HU)
    at_face = np.flatnonzero(face_tissue.any(axis=(1, 2)))
    if not at_face.size:
        return np.empty((0, 3)), {}
    in_slices = body[at_face[0]:at_face[-1] + 1]
    _, ap, _ = np.nonzero(in_slices)
    if not ap.size:
        return np.empty((0, 3)), {}
    skin[:, :int((ap.min() + ap.max()) / 2)] = False          # the anterior half of the head only
    skin[:at_face[0]] = False                                 # no neck or shoulders: the template has none to match

    a, b = np.nonzero(face.any(axis=0))
    pad_a, pad_b = int(np.ceil(NEAR_FACE_MM / s_ap)) + 1, int(np.ceil(NEAR_FACE_MM / s_lr)) + 1
    crop = (slice(None), slice(max(0, a.min() - pad_a), a.max() + pad_a + 1),
            slice(max(0, b.min() - pad_b), b.max() + pad_b + 1))
    near = ndi.distance_transform_edt(~face[crop], sampling=p.spacing) <= NEAR_FACE_MM
    away = ndi.distance_transform_edt(~region[crop], sampling=p.spacing) >= EDGE_MM
    wanted = np.zeros_like(skin)
    wanted[crop] = near & away
    zz, aa, bb = np.nonzero(skin & wanted)
    points = np.stack([bb * s_lr, aa * s_ap, (zz + z0) * s_si], axis=1).astype(np.float64)
    if len(points) > MAX_POINTS:
        points = points[np.sort(np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False))]
    face_lr = np.flatnonzero(face_tissue.any(axis=(0, 1)))
    measures = {"above_face_mm": float(len(p.hu) - 1 - (at_face[-1] + z0)) * s_si, "back": float(ap.min()) * s_ap, "centre_lr": float(face_lr[0] + face_lr[-1]) / 2 * s_lr,
                "si_low": float(at_face[0] + z0) * s_si, "si_high": float(at_face[-1] + z0) * s_si}
    return points, measures


def _start(t: Template, vertices: np.ndarray, m: dict) -> Pose:
    """The mean head, centred left-right and on the face, with the back of the head in place."""
    face = vertices[t.labels == FACE]
    low, high = face[:, 2].min(), face[:, 2].max()
    level = (t.labels != EYE) & (vertices[:, 2] >= low) & (vertices[:, 2] <= high)
    return Pose(1.0, np.eye(3), np.array([m["centre_lr"] - (face[:, 0].min() + face[:, 0].max()) / 2,
                                          m["back"] - float(vertices[level, 1].min()),
                                          (m["si_low"] + m["si_high"]) / 2 - (low + high) / 2]))


def _perturbed(pose: Pose, pivot: np.ndarray, tilt_deg: float, ap_mm: float, si_mm: float) -> Pose:
    """pose with the head tilted about the left-right axis through `pivot` (patient mm) and shifted."""
    angle = np.deg2rad(tilt_deg)
    rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    return Pose(pose.scale, rot @ pose.rotation, pivot + rot @ (pose.translation - pivot) + np.array([0.0, ap_mm, si_mm]))


@dataclass(frozen=True)
class Fit:
    accepted: bool
    reason: str                 # why it was rejected; "" if accepted
    pose: Pose | None = None
    lr_sign: int = 1
    rms_mm: float | None = None
    moved_mm: float | None = None
    scale_change: float | None = None
    spread_mm: float | None = None
    above_face_mm: float | None = None
    ap_constraint: float | None = None
    si_constraint: float | None = None
    points: int = 0
    coverage: float | None = None
    front: np.ndarray | None = None     # anterior position (mm) per (slice, left-right column), NaN = no template

    def numbers(self) -> dict:
        """What the report and the manifest may show: no image-derived pixels."""
        def num(v, digits=2):
            return None if v is None else round(float(v), digits)
        return {"accepted": self.accepted, "reason": self.reason, "lr_sign": self.lr_sign, "points": self.points,
                "rms_mm": num(self.rms_mm), "moved_mm": num(self.moved_mm), "scale": None if self.pose is None else num(self.pose.scale, 4),
                "scale_change": num(self.scale_change, 4), "spread_mm": num(self.spread_mm),
                "above_face_mm": num(self.above_face_mm, 1), "ap_constraint": num(self.ap_constraint, 3), "si_constraint": num(self.si_constraint, 3),
                "coverage": num(self.coverage, 4)}


def _fit_one(t: Template, vertices: np.ndarray, points: np.ndarray, m: dict) -> tuple:
    """The best pose of one mirror image: (pose, rms, moved mm, scale change, spread mm, AP and SI constraint)."""
    cloud, normals = sample_triangles(vertices, t.triangles[(t.labels[t.triangles] == REST).all(axis=1)])
    tree = cKDTree(cloud)
    face = vertices[t.labels == FACE]
    start = _start(t, vertices, m)
    pivot = start.apply(face.mean(axis=0))
    results = [_icp(points, tree, cloud, normals, _perturbed(start, pivot, *offset)) for offset in STARTS]
    best_pose, best_rms = min(results, key=lambda r: r[1])
    faces = best_pose.apply(face)
    spread = max(float(np.sqrt(((pose.apply(face) - faces) ** 2).sum(axis=1).mean()))
                 for pose, rms in results if rms <= best_rms * EQUAL_FIT)
    moved = float(np.linalg.norm(best_pose.apply(face).mean(axis=0) - pivot))
    idx, distance = _pairs(points, tree, cloud, best_pose)
    used = normals[idx[np.argsort(distance, kind="stable")[:max(7, int(TRIM * len(points)))]]] @ best_pose.rotation.T
    _, ap, si = (used ** 2).mean(axis=0)
    return best_pose, best_rms, moved, abs(best_pose.scale - 1.0), spread, float(ap), float(si)


def front_map(vertices: np.ndarray, triangles: np.ndarray, shape: tuple[int, int], spacing: tuple) -> np.ndarray:
    """
    Most anterior position (mm) of the posed template per (slice, left-right column), NaN where
    there is no template. Grooves narrower than CLOSING_MM (the mouth slit) are closed.
    """
    s_si, _, s_lr = spacing
    z, l, a = vertices[:, 2] / s_si, vertices[:, 0] / s_lr, vertices[:, 1]
    front = np.full(shape, -np.inf)
    for i, j, k in triangles:
        low_z, high_z = int(np.ceil(min(z[i], z[j], z[k]))), int(np.floor(max(z[i], z[j], z[k])))
        low_l, high_l = int(np.ceil(min(l[i], l[j], l[k]))), int(np.floor(max(l[i], l[j], l[k])))
        low_z, low_l, high_z, high_l = max(low_z, 0), max(low_l, 0), min(high_z, shape[0] - 1), min(high_l, shape[1] - 1)
        if low_z > high_z or low_l > high_l:
            continue
        zz, ll = np.meshgrid(np.arange(low_z, high_z + 1), np.arange(low_l, high_l + 1), indexing="ij")
        det = (z[j] - z[i]) * (l[k] - l[i]) - (z[k] - z[i]) * (l[j] - l[i])
        if abs(det) < 1e-9:
            continue
        u = ((zz - z[i]) * (l[k] - l[i]) - (z[k] - z[i]) * (ll - l[i])) / det
        v = ((z[j] - z[i]) * (ll - l[i]) - (zz - z[i]) * (l[j] - l[i])) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if inside.any():
            np.maximum.at(front, (zz[inside], ll[inside]), a[i] + u[inside] * (a[j] - a[i]) + v[inside] * (a[k] - a[i]))
    known = np.isfinite(front)
    if not known.any():
        return np.full(shape, np.nan)
    rz, rl = int(CLOSING_MM / s_si), int(CLOSING_MM / s_lr)
    y, x = np.ogrid[-rz:rz + 1, -rl:rl + 1]
    closed = ndi.grey_closing(np.where(known, front, front[known].min() - 1000.0),
                              footprint=(y * s_si) ** 2 + (x * s_lr) ** 2 <= CLOSING_MM ** 2)
    return np.where(known, closed, np.nan)


def fit_template(t: Template, hu: np.ndarray, face: np.ndarray, region: np.ndarray, axes: Axes) -> Fit:
    """The fit of the template to one image; `face` is the face mask without margin or extension."""
    p = orient(hu, face, region, axes)
    if p is None:
        return Fit(False, "unknown image orientation")
    if not p.face.any():
        return Fit(False, "empty face mask")
    points, measures = surface_points(p)
    if len(points) < MIN_POINTS:
        return Fit(False, "too few skin points around the face", points=len(points))
    above = measures["above_face_mm"]
    candidates = []
    for sign in (1, -1):
        mirrored = t.vertices * np.array([sign, 1.0, 1.0])
        candidates.append((*_fit_one(t, mirrored, points, measures), sign, mirrored))
    pose, rms, moved, scale_change, spread, ap, si, sign, vertices = min(candidates, key=lambda c: c[1])
    common = {"pose": pose, "lr_sign": sign, "rms_mm": rms, "moved_mm": moved, "scale_change": scale_change,
              "spread_mm": spread, "ap_constraint": ap, "si_constraint": si, "points": len(points),
              "above_face_mm": above}
    front = front_map(pose.apply(vertices), t.triangles, (p.hu.shape[0], p.hu.shape[2]), p.spacing)
    columns = (p.face & (p.hu > BODY_HU)).any(axis=1)
    coverage = float(np.isfinite(front[columns]).mean())
    common.update(coverage=coverage, front=front)
    if above < MIN_ABOVE_FACE_MM:
        return Fit(False, "the scan ends too close to the top of the face", **common)
    if scale_change > MAX_SCALE_CHANGE:
        return Fit(False, "the head is much larger or smaller than the template", **common)
    tilt = float(np.rad2deg(np.arccos(np.clip((np.trace(pose.rotation) - 1.0) / 2.0, -1.0, 1.0))))
    if tilt > MAX_TILT_DEG:
        return Fit(False, "the head is turned too far", **common)
    if rms > MAX_RMS_MM:
        return Fit(False, "skin does not match the template", **common)
    if spread > MAX_SPREAD_MM:
        return Fit(False, "the pose is ambiguous", **common)
    if coverage < MIN_COVERAGE:
        return Fit(False, "the template does not cover the face mask", **common)
    return Fit(True, "", **common)


def front_minus_skin(fit: Fit, hu: np.ndarray, face: np.ndarray, region: np.ndarray, axes: Axes) -> dict | None:
    """
    VALIDATION ONLY, for the fit report: how far the fitted front lies from the real skin (mm,
    positive = in front of it), per column of the face mask that holds tissue. This reads the
    face that is to be removed, so it must never influence a fit or a fill.
    """
    p = orient(hu, face, region, axes)
    if fit.front is None or p is None:
        return None
    tissue = p.hu > BODY_HU
    columns = (p.face & tissue).any(axis=1)
    disk = thin_free_disk((p.spacing[1], p.spacing[2]))
    for k in np.flatnonzero(columns.any(axis=1)):
        tissue[k] = ndi.binary_opening(tissue[k], structure=disk)
    skin = (tissue.shape[1] - 1 - np.argmax(tissue[:, ::-1, :], axis=1)) * p.spacing[1]
    used = columns & tissue.any(axis=1) & np.isfinite(fit.front)
    if not used.any():
        return None
    low, median, high = np.percentile(fit.front[used] - skin[used], [10, 50, 90])
    return {"skin_p10_mm": round(float(low), 1), "skin_p50_mm": round(float(median), 1), "skin_p90_mm": round(float(high), 1)}


def resolve_template_path(path: str | None) -> Path:
    """--template-model, else $DEFACE_TEMPLATE, else the template in the repository."""
    return Path(path or os.environ.get("DEFACE_TEMPLATE") or DEFAULT_TEMPLATE).expanduser()


@lru_cache(maxsize=4)
def _template(path: str) -> Template:
    return load_template(Path(path))


def prepare_fill(opts: Options, hu: np.ndarray, face: np.ndarray, region: np.ndarray, axes: Axes, label: str,
                 orientation_known: bool = True) -> tuple[str, TemplateFront | None, list[str], dict]:
    """
    (fill mode to use, template front or None, CHECK lines, details for the report). Other
    fills pass through unchanged. --fill template inserts the template whenever a pose exists:
    a fit below the quality limits is flagged, so that every case can be compared across
    fills. --fallback soft-tissue or fail acts on those fits instead. Without any pose the
    template cannot be placed, and the patient fails unless --fallback soft-tissue is given.
    """
    if opts.fill != "template":
        return opts.fill, None, [], {}
    template = _template(str(resolve_template_path(opts.template_model)))
    fit = fit_template(template, hu, face, region, axes) if orientation_known else Fit(False, "unknown image orientation")
    info: dict = {"template_sha256": template.sha256, "template_fit": fit.numbers()}
    if fit.pose is not None:
        info["template_pose"] = {"scale": float(fit.pose.scale), "rotation": fit.pose.rotation.tolist(),
                                 "translation_mm": fit.pose.translation.tolist()}
    placed = fit.front is not None
    if not fit.accepted and (opts.template_fallback != "none" or not placed):
        text = f"{label}: --fill template: {fit.reason}"
        if opts.template_fallback != "soft-tissue":
            hint = "--fallback fail" if opts.template_fallback == "fail" else "use --fallback soft-tissue to fill it with soft tissue"
            raise PatientError(f"{text}; nothing was written for this patient ({hint})")
        return "soft-tissue", None, [f"{text}: soft-tissue used instead, no synthetic face"], {
            **info, "fill_used": "soft-tissue", "template_fallback": fit.reason, "template_quality": fit.reason}
    flags, notes = [], [] if fit.accepted else [fit.reason]
    # measured on the real face, for the report only: the fill is already decided
    skin = front_minus_skin(fit, hu, face, region, axes)
    if skin:
        info["template_skin"] = skin
        if abs(skin["skin_p50_mm"]) > MAX_SKIN_MEDIAN_MM or max(abs(skin["skin_p10_mm"]), abs(skin["skin_p90_mm"])) > MAX_SKIN_P90_MM:
            notes.append(f"front {skin['skin_p50_mm']:+.0f} mm from the real skin at the median "
                         f"({skin['skin_p10_mm']:+.0f} to {skin['skin_p90_mm']:+.0f} mm)")
    if notes:
        flags.append(f"{label}: --fill template: the synthetic face does not fit well: {'; '.join(notes)}. "
                     "It was inserted anyway (--fallback soft-tissue replaces such fits); check the QC picture")
    return "template", TemplateFront(fit.front), flags, {**info, "fill_used": "template", "template_quality": "; ".join(notes) or "ok"}
