"""
Optional: the real TotalSegmentator on a real head CT (also a site validation for a new
hospital, scanner or protocol). Skipped unless DEFACE_TEST_CT is set and TotalSegmentator
(with its license) is installed. Runs on the CPU unless DEFACE_TEST_DEVICE says otherwise.

    DEFACE_TEST_CT=/data/patient01 python -m pytest tests/test_totalsegmentator.py -v

DEFACE_TEST_CT: a DICOM folder or a .nii/.nii.gz/.nrrd file of a CT that shows the face.
DEFACE_TEST_FACE_ML (optional): the expected face mask volume (ml) of that CT, e.g. from an
earlier run; the test then fails if the mask differs by more than 10 % (other software
versions, other hardware).
The QC picture is kept in the folder printed by the test: look at it.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from defacing.common import Options, classify_suffix
from defacing.run import run_single
from defacing.segment import HEAD_MIN_ML, head_volume_ml

CT = os.environ.get("DEFACE_TEST_CT", "")
DEVICE = os.environ.get("DEFACE_TEST_DEVICE", "cpu")
FACE_ML_RANGE = (20.0, 3000.0)          # plausible face mask volumes

pytestmark = pytest.mark.skipif(not CT or shutil.which("TotalSegmentator") is None,
                                reason="set DEFACE_TEST_CT to a real head CT (needs TotalSegmentator)")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    work = tmp_path_factory.mktemp("real_ct")
    out, qc = work / "out", work / "qc"
    code = run_single(Path(CT).expanduser().resolve(), out, qc, Options(device=DEVICE), False)
    print(f"\nde-faced output: {out}\nQC folder (look at QC_defacing*.png): {qc}")
    return code, out, qc


def test_the_ct_is_defaced(result):
    code, out, qc = result
    assert code == 0, f"de-facing failed, see {qc}"
    manifest = json.loads((qc / "manifest.json").read_text(encoding="utf-8"))
    volumes = manifest["face_mask_ml"]
    assert volumes and all(FACE_ML_RANGE[0] <= v <= FACE_ML_RANGE[1] for v in volumes), volumes
    assert list(qc.glob("QC_defacing*.png"))
    expected = os.environ.get("DEFACE_TEST_FACE_ML")
    if expected:
        assert abs(volumes[0] - float(expected)) <= 0.1 * float(expected), (volumes[0], expected)


def test_the_head_is_detected(result, tmp_path):
    """The check behind --allow-empty-mask: the "total" model finds skull and brain."""
    _, _, qc = result
    image = Path(CT)
    if not classify_suffix(image):                 # DICOM: use the de-faced CT written to the QC folder
        image = sorted(qc.glob("CT_defaced*.nii.gz"))[0]
    ml = head_volume_ml(image, tmp_path / "head", DEVICE, tmp_path / "tool.log")
    assert ml >= HEAD_MIN_ML


def test_the_face_mask_holds_air(result):
    """Independent of the tool's own read-back check: in the de-faced volume (QC folder), the
    raw face mask holds air (the default fill) everywhere."""
    import nibabel as nib
    _, _, qc = result
    for mask_path in sorted(qc.glob("face_mask*.nii.gz")):
        defaced_path = qc / mask_path.name.replace("face_mask", "CT_defaced")
        if not defaced_path.exists():                # NIfTI/NRRD input: no CT_defaced in the QC folder
            continue
        mask_img, defaced = nib.load(str(mask_path)), nib.load(str(defaced_path))
        assert mask_img.shape == defaced.shape
        hu = np.asanyarray(defaced.dataobj)[np.asanyarray(mask_img.dataobj) > 0]
        assert (hu < -800).all(), f"{int((hu >= -800).sum())} face mask voxels hold more than -800 HU"
