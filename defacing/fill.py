"""
Values written into the de-faced region.

air (default)
    Air sampled from the same scan: noise with the in-plane texture (noise power
    spectrum) of the scan's air, mapped onto the scan's own distribution of air values
    (including any clipping, e.g. at -1024 HU). The de-faced region can therefore not
    be found by selecting voxels with one exact value, and its outline (the shape of
    the face mask) is not revealed.

soft-tissue
    A face-independent replacement that reduces the gap between de-faced training data
    and clinical CTs (which still show a face) for segmentation models. In every slice
    perpendicular to the head-feet axis, the convex hull of the body that remains after
    de-facing forms a smooth new front of the head (no nose, lips or eyes). De-faced
    voxels inside the hull get soft tissue with the scan's noise level and texture, and
    a 2 mm skin layer at the hull surface; voxels outside the hull get air as above.
    Needs the region to reach the anterior image edge (otherwise concave parts of the
    face would leave a face-shaped tissue/air boundary inside the hull).

constant
    -1000 HU everywhere in the region. Selecting the voxels that are exactly -1000 HU
    then shows the outline of the face mask: use only if the data stays internal.

For air and soft-tissue, everything is computed from voxels OUTSIDE the de-faced region
plus a random seed, so the fill adds no information about the face. Voxels outside the
reconstructed field of view (padding below -1100 HU, e.g. -3024 HU) are left unchanged;
they never hold tissue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import FACE_VALUE_HU
from .masks import Axes

BODY_HU = -500.0            # body = above this
AIR_MAX_HU = -900.0         # real air = below this (and above the padding)
PADDING_BELOW_HU = -1100.0  # values below: outside the reconstructed field of view
STATS_MM = 50.0             # air/noise statistics: slices within this distance of the region
NEAR_MM = 20.0              # soft-tissue value: tissue within this distance of the region
SKIN_MM = 2.0
PATCH = 16                  # noise patches (pixels)
MAX_PATCHES = 400
MAX_AIR_SAMPLES = 500_000


@dataclass
class NoiseModel:
    std: float
    kernel: np.ndarray    # 2D in-plane filter with unit energy: white noise * kernel has unit std


def fill_seed(text: str) -> int:
    """Reproducible seed derived from a (random, non-identifying) UID."""
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def apply_fill(hu: np.ndarray, region: np.ndarray, axes: Axes, mode: str, seed: int) -> tuple[np.ndarray, dict]:
    """Return a copy of hu (in HU) with the de-faced region filled, and a description of the fill."""
    if mode == "constant":
        out = hu.astype(np.float32, copy=True)
        out[region] = FACE_VALUE_HU
        return out, {"fill": "constant", "value_hu": FACE_VALUE_HU}
    if mode not in ("air", "soft-tissue"):
        raise ValueError(f"unknown fill mode: {mode}")
    if not region.any():
        return hu.astype(np.float32, copy=True), {"fill": mode, "seed": seed, "voxels": 0}
    return _sampled_fill(hu, region, axes, mode, seed)


# ---------------------------------------------------------------------------
# Noise model and sampling
# ---------------------------------------------------------------------------

def estimate_noise(img: np.ndarray, region: np.ndarray, rng: np.random.Generator,
                   homogeneous_only: bool) -> NoiseModel | None:
    """
    Noise level and in-plane texture from PATCH x PATCH patches that lie entirely in
    `region` (numpy axes: slice, row, col). With homogeneous_only, only the half of the
    patches with the lowest variance is used, so that anatomy does not count as noise.
    Returns None if there are too few patches.
    """
    from scipy.ndimage import uniform_filter
    h = PATCH // 2
    frac = uniform_filter(region.astype(np.float32), size=(1, PATCH, PATCH), mode="constant")
    centres = np.argwhere(frac > 0.999)
    centres = centres[(centres[:, 1] >= h) & (centres[:, 1] + h <= img.shape[1])
                      & (centres[:, 2] >= h) & (centres[:, 2] + h <= img.shape[2])]
    if len(centres) < 20:
        return None
    pick = centres[rng.choice(len(centres), min(3 * MAX_PATCHES, len(centres)), replace=False)]
    patches = np.stack([img[z, a - h:a - h + PATCH, b - h:b - h + PATCH] for z, a, b in pick]).astype(np.float64)
    patches -= patches.mean(axis=(1, 2), keepdims=True)
    if homogeneous_only:
        stds = patches.std(axis=(1, 2))
        patches = patches[stds <= np.median(stds)]
    patches = patches[:MAX_PATCHES]
    nps = (np.abs(np.fft.fft2(patches)) ** 2).mean(axis=0)
    std = float(np.sqrt(nps.sum()) / PATCH ** 2)          # Parseval
    kernel = np.fft.fftshift(np.real(np.fft.ifft2(np.sqrt(nps))))
    norm = np.sqrt((kernel ** 2).sum())
    if std <= 0 or norm <= 0:
        return None
    return NoiseModel(std=std, kernel=kernel / norm)


def synthesize(shape: tuple, model: NoiseModel, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean noise with the model's level and in-plane texture (slices independent)."""
    from scipy.signal import fftconvolve
    white = rng.standard_normal(shape)
    if model.kernel.size == 1:
        return model.std * white
    return model.std * fftconvolve(white, model.kernel[None], mode="same", axes=(1, 2))


def _white(std: float) -> NoiseModel:
    return NoiseModel(std=std, kernel=np.ones((1, 1)))


def _robust_std(values: np.ndarray, default: float) -> float:
    if values.size > 100:
        mad = float(np.median(np.abs(values - np.median(values))))
        return 1.4826 * mad or default
    return default


def sample_like(values: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Map a unit-variance (correlated) Gaussian field onto the empirical distribution of
    `values`: the result has exactly that distribution and keeps the field's texture."""
    from scipy.special import ndtr
    srt = np.sort(values)
    idx = np.minimum((ndtr(field) * len(srt)).astype(np.int64), len(srt) - 1)
    return srt[idx]


# ---------------------------------------------------------------------------
# Sampled fills (air, soft-tissue)
# ---------------------------------------------------------------------------

def _hull_membership(body2d: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """For (n, 2) pixel indices: inside the convex hull of the body pixels of one slice?"""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import ConvexHull, QhullError
    edge = np.argwhere(body2d & ~binary_erosion(body2d))
    if len(edge) < 3:
        return np.zeros(len(pixels), bool)
    try:
        hull = ConvexHull(edge)
    except QhullError:
        return np.zeros(len(pixels), bool)
    eq = hull.equations                                    # n . x + c <= 0 inside
    return (pixels @ eq[:, :2].T + eq[:, 2] <= 0.5).all(axis=1)


def _sampled_fill(hu: np.ndarray, region: np.ndarray, axes: Axes, mode: str, seed: int) -> tuple[np.ndarray, dict]:
    from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, label

    rng = np.random.default_rng(seed)
    # head-feet axis first, so that h[z] are axial(-like) slices
    h = np.moveaxis(hu.astype(np.float32), axes.si_axis, 0)
    r = np.moveaxis(region, axes.si_axis, 0)
    spacing = (axes.spacing[axes.si_axis],) + tuple(s for i, s in enumerate(axes.spacing) if i != axes.si_axis)
    out = h.copy()

    # padding outside the reconstructed field of view keeps its value (it never holds
    # tissue, so this reveals nothing about the face)
    outside = ~r
    target_all = r & (h >= PADDING_BELOW_HU)
    info = {"fill": mode, "seed": seed}
    if not target_all.any():
        return np.moveaxis(out, 0, axes.si_axis), {**info, "voxels": 0}

    # slab of slices around the region, for the hull and the statistics
    zs = np.flatnonzero(r.any(axis=(1, 2)))
    margin = int(np.ceil(STATS_MM / spacing[0]))
    z0, z1 = max(0, zs[0] - margin), min(len(h), zs[-1] + margin + 1)
    sub, rs, outs, target = h[z0:z1], r[z0:z1], outside[z0:z1], target_all[z0:z1]

    # real air: outside every object (body, couch, devices, incl. their inner cavities);
    # objects are taken from outside the region only, so the face does not influence it
    objects = np.stack([binary_fill_holes(s) for s in (sub > BODY_HU) & outs])
    real_air = (outs & ~binary_dilation(objects, iterations=3)
                & (sub > PADDING_BELOW_HU) & (sub < AIR_MAX_HU))
    if real_air.sum() < 1000:                              # too little near the region: whole scan
        sub_all = h
        objs_all = np.stack([binary_fill_holes(s) for s in (h > BODY_HU) & outside])
        air_all = (outside & ~binary_dilation(objs_all, iterations=3)
                   & (h > PADDING_BELOW_HU) & (h < AIR_MAX_HU))
    else:
        sub_all, air_all = sub, real_air
    air_values = sub_all[air_all]
    if air_values.size > MAX_AIR_SAMPLES:
        air_values = rng.choice(air_values, MAX_AIR_SAMPLES, replace=False)
    air_noise = estimate_noise(sub_all, air_all, rng, homogeneous_only=False) or _white(1.0)

    # synthesis, in the bounding box of the voxels to fill
    box = tuple(slice(ix.min(), ix.max() + 1) for ix in np.nonzero(target))
    shape = tuple(s.stop - s.start for s in box)
    unit_air = synthesize(shape, NoiseModel(1.0, air_noise.kernel), rng)
    if air_values.size >= 100:
        air_fill = sample_like(air_values, unit_air)
        info.update(air_hu=round(float(np.median(air_values)), 1),
                    air_noise_std=round(_robust_std(air_values, 0.0), 1), air_source="scan")
    else:                                                  # no real air in the image at all
        air_fill = FACE_VALUE_HU + 10.0 * unit_air
        info.update(air_hu=FACE_VALUE_HU, air_noise_std=10.0, air_source="default")
    vals = out[z0:z1][box]                                 # a view into out
    tgt = target[box]

    if mode == "air":
        vals[tgt] = air_fill[tgt]
        info["voxels_air"] = int(tgt.sum())
        return np.moveaxis(out, 0, axes.si_axis), info

    # soft-tissue: hull of the remaining body (largest object outside the region)
    body = (sub > BODY_HU) & outs
    lab, n = label(body)
    if n > 1:                                              # drop couch, separate devices
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        body = lab == int(np.argmax(sizes))
    hull = np.zeros_like(rs)
    pad = int(np.ceil(SKIN_MM / min(spacing[1:]))) + 2
    for z in np.flatnonzero(rs.any(axis=(1, 2))):
        a, b = np.nonzero(rs[z])
        aa, bb = np.meshgrid(np.arange(max(0, a.min() - pad), min(rs.shape[1], a.max() + pad + 1)),
                             np.arange(max(0, b.min() - pad), min(rs.shape[2], b.max() + pad + 1)),
                             indexing="ij")
        pix = np.stack([aa.ravel(), bb.ravel()], axis=1)
        inside = _hull_membership(body[z], pix)
        hull[z, pix[inside, 0], pix[inside, 1]] = True
    tissue = target & hull
    skin = np.zeros_like(tissue)                           # tissue within SKIN_MM of the hull surface
    for z in np.flatnonzero(tissue.any(axis=(1, 2))):
        skin[z] = tissue[z] & (distance_transform_edt(hull[z], sampling=spacing[1:]) <= SKIN_MM)

    # tissue values: soft tissue near the region (distances in a crop around it)
    _, ai, bi = np.nonzero(rs)
    pa, pb = (int(np.ceil(NEAR_MM / s)) for s in spacing[1:])
    crop = (slice(None), slice(max(0, ai.min() - pa), ai.max() + pa + 1),
            slice(max(0, bi.min() - pb), bi.max() + pb + 1))
    near = body[crop] & (distance_transform_edt(~rs[crop], sampling=spacing) <= NEAR_MM)
    near_hu = sub[crop][near]
    soft = near_hu[(near_hu > -200) & (near_hu < 200)]
    tissue_hu = float(np.median(soft)) if soft.size > 100 else 20.0
    dermis = near_hu[(near_hu >= 0) & (near_hu <= 100)]
    skin_hu = float(np.median(dermis)) if dermis.size > 100 else 40.0
    homogeneous = body & (sub > -150) & (sub < 150)
    tissue_noise = estimate_noise(sub, homogeneous, rng, homogeneous_only=True) or _white(20.0)
    t_noise = synthesize(shape, tissue_noise, rng)

    t, s = tissue[box], skin[box]
    a = tgt & ~t
    vals[a] = air_fill[a]
    vals[t] = tissue_hu + t_noise[t]
    vals[s] = skin_hu + t_noise[s]
    info.update(tissue_hu=round(tissue_hu, 1), skin_hu=round(skin_hu, 1),
                tissue_noise_std=round(tissue_noise.std, 1),
                voxels_tissue=int(t.sum()), voxels_air=int(a.sum()))
    return np.moveaxis(out, 0, axes.si_axis), info
