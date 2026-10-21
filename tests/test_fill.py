"""Fill of the de-faced region: values, privacy properties, reproducibility."""
import numpy as np
import pytest

import phantom as ph
from defacing.fill import AIR_MAX_HU, PADDING_BELOW_HU, apply_fill, fill_seed, sample_like
from defacing.masks import Axes, extend_mask_anteriorly


def volume(fov_radius=None, shell=False, transpose=False):
    """Axial phantom (z, y, x) with 4 mm voxels; region = face extended to the front edge."""
    z, y, x = np.meshgrid(np.arange(-72, 72, 4.0), np.arange(-126, 126, 4.0), np.arange(-110, 110, 4.0),
                          indexing="ij")
    world = np.stack([x, y, z], axis=-1)
    hu = ph.phantom_hu(world, np.random.default_rng(1), fov_radius=fov_radius, shell=shell)
    region = extend_mask_anteriorly(ph.in_face(world), axis=1, increasing=False)
    axes = Axes(spacing=(4.0, 4.0, 4.0), anterior=(1, False), si_axis=0)
    if transpose:                            # head-feet along the last numpy axis
        hu, region = np.moveaxis(hu, 0, 2), np.moveaxis(region, 0, 2)
        axes = Axes(spacing=(4.0, 4.0, 4.0), anterior=(1, False), si_axis=2)
    return hu, region, axes


@pytest.mark.parametrize("mode", ["air", "soft-tissue", "constant"])
def test_only_the_region_changes(mode):
    hu, region, axes = volume()
    out, info = apply_fill(hu, region, axes, mode, seed=3)
    assert out.shape == hu.shape
    assert np.array_equal(out[~region], hu[~region].astype(np.float32))
    assert info["fill"] == mode


def test_constant_fill():
    hu, region, axes = volume()
    out, info = apply_fill(hu, region, axes, "constant", seed=0)
    assert np.all(out[region] == -1000.0) and info["value_hu"] == -1000.0


@pytest.mark.parametrize("transpose", [False, True])
def test_air_fill_looks_like_the_scans_air(transpose):
    hu, region, axes = volume(transpose=transpose)
    out, info = apply_fill(hu, region, axes, "air", seed=3)
    vals = out[region]
    assert vals.max() < AIR_MAX_HU and vals.min() >= -1024.0        # the scan's air, incl. its clipping
    assert abs(np.median(vals) - (-1000.0)) < 5
    assert 8 < vals.std() < 25                                      # the phantom's air noise is 15 HU
    # no single value marks the region (a constant fill would make it selectable)
    _, counts = np.unique(vals, return_counts=True)
    assert counts.max() / vals.size < 0.1
    assert info["air_source"] == "scan" and info["voxels_air"] == int(region.sum())


@pytest.mark.parametrize("mode", ["air", "soft-tissue"])
def test_fill_does_not_depend_on_the_face(mode):
    """Everything the fill uses comes from outside the region: two scans that differ only
    inside it (different faces) get exactly the same fill."""
    hu, region, axes = volume(shell=True)
    other = hu.copy()
    other[region] = np.random.default_rng(9).uniform(-1000, 1500, int(region.sum()))
    a, _ = apply_fill(hu, region, axes, mode, seed=5)
    b, _ = apply_fill(other, region, axes, mode, seed=5)
    assert np.array_equal(a, b)


def test_fill_is_reproducible_with_the_seed():
    hu, region, axes = volume()
    a, ia = apply_fill(hu, region, axes, "air", seed=11)
    b, _ = apply_fill(hu, region, axes, "air", seed=11)
    c, _ = apply_fill(hu, region, axes, "air", seed=12)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert ia["seed"] == 11


def test_padding_outside_the_field_of_view_keeps_its_value():
    hu, region, axes = volume(fov_radius=110)
    padding = region & (hu < PADDING_BELOW_HU)
    assert padding.any()
    for mode in ("air", "soft-tissue"):
        out, _ = apply_fill(hu, region, axes, mode, seed=1)
        assert np.array_equal(out[padding], hu[padding].astype(np.float32))
        assert (out[region & ~padding] >= PADDING_BELOW_HU).all()


def test_soft_tissue_fill_builds_a_smooth_front_of_the_head():
    hu, region, axes = volume()
    out, info = apply_fill(hu, region, axes, "soft-tissue", seed=2)
    tissue = region & (out > -200)
    assert info["voxels_tissue"] > 0 and info["voxels_air"] > 0
    assert info["voxels_tissue"] + info["voxels_air"] == int(region.sum())
    assert abs(info["tissue_hu"] - ph.TISSUE_HU) < 10
    assert abs(np.median(out[tissue]) - ph.TISSUE_HU) < 15
    # the nose sticks out of the convex hull of the remaining head: it becomes air
    z, y, x = np.meshgrid(np.arange(-72, 72, 4.0), np.arange(-126, 126, 4.0), np.arange(-110, 110, 4.0),
                          indexing="ij")
    nose_tip = ph.in_box(np.stack([x, y, z], axis=-1), ((-10, 10), (-110, -100), (-20, 10)))
    assert (out[nose_tip] < -800).all()


def test_empty_region_and_unknown_mode():
    hu, region, axes = volume()
    out, info = apply_fill(hu, np.zeros_like(region), axes, "air", seed=1)
    assert np.array_equal(out, hu.astype(np.float32)) and info["voxels"] == 0
    with pytest.raises(ValueError):
        apply_fill(hu, region, axes, "zeros", seed=1)


def test_scan_without_air_uses_the_default_air():
    hu, region, axes = volume()
    hu = np.maximum(hu, 0.0)                  # e.g. a reconstruction that clips at 0 HU
    out, info = apply_fill(hu, region, axes, "air", seed=1)
    assert info["air_source"] == "default"
    assert abs(np.median(out[region]) + 1000) < 5


def test_sample_like_reproduces_the_distribution():
    rng = np.random.default_rng(0)
    values = np.concatenate([np.full(300, -1024.0), rng.normal(-1000, 10, 700)])
    out = sample_like(values, rng.standard_normal(100_000))
    assert abs((out == -1024.0).mean() - 0.3) < 0.02
    assert set(np.unique(out)) <= set(values)


def test_fill_seed_is_deterministic_and_fits_numpy():
    assert fill_seed("1.2.3") == fill_seed("1.2.3") != fill_seed("1.2.4")
    np.random.default_rng(fill_seed("1.2.3"))
