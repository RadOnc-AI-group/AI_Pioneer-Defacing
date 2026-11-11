"""
The fills of tool version 5 must not change. tests/golden/fill_v5.npz holds what the
version 5 code wrote into the de-faced region; with --deface-targets the whole pipeline
must give the output of version 5.
"""
from pathlib import Path

import numpy as np
import pytest

import phantom as ph
import test_fill
from defacing.common import Options
from defacing.fill import apply_fill
from defacing.patient import deface_dicom_patient

GOLDEN = Path(__file__).parent / "golden" / "fill_v5.npz"


@pytest.mark.parametrize("mode", ["air", "soft-tissue", "constant"])
def test_fill_equals_version_5(mode):
    hu, region, axes = test_fill.volume(fov_radius=120, shell=True)
    out, _ = apply_fill(hu, region, axes, mode, seed=12345)
    golden = np.load(GOLDEN)[mode.replace("-", "_")]
    new = np.round(out[region]).astype(np.int16)
    assert new.shape == golden.shape
    # sampled values depend on floating-point FFT rounding; at most a handful may differ between platforms
    assert (new != golden).mean() < 0.005, "the fill changed: add a new fill mode instead (see README)"


def run_patient(tmp_path: Path, name: str, opts: Options, rois):
    pat = tmp_path / name / "in"
    ct = ph.write_ct_series(pat / "CT")
    if rois:
        ph.write_rtstruct(pat / "RS.dcm", ct, rois)
    out = tmp_path / name / "out"
    res = deface_dicom_patient(pat, name, out, tmp_path / name / "qc", opts)
    assert res.status == "done", res.error
    return ph.read_series_hu(out / "dcm")[0], res


@pytest.mark.parametrize("fill", ["air", "soft-tissue", "constant"])
def test_deface_targets_gives_the_output_without_targets(tmp_path, fake_ts, fill):
    """A target in the face changes nothing with --deface-targets: same voxels as without contours."""
    without, _ = run_patient(tmp_path, "a", Options(seed=7, fill=fill), None)
    with_rs, res = run_patient(tmp_path, "b", Options(seed=7, fill=fill, deface_targets=True),
                               [("BODY", "EXTERNAL"), ("PTV_nose", "PTV")])
    assert np.array_equal(without, with_rs) and not res.targets


def test_targets_only_change_the_kept_voxels(tmp_path, fake_ts):
    """Keeping a target leaves every other value of the fill exactly as without it."""
    without, _ = run_patient(tmp_path, "a", Options(seed=7), None)
    kept, res = run_patient(tmp_path, "b", Options(seed=7), [("PTV_nose", "PTV")])
    assert res.target_kept_ml[0] > 0
    ct = ph.write_ct_series(tmp_path / "ref")
    changed = without != kept
    assert changed.any() and np.array_equal(kept[changed], ct.hu[changed])
