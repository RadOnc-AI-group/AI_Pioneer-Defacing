"""The TotalSegmentator call, with a stand-in executable on PATH."""
import os
import sys

import pytest

from defacing.common import PatientError
from defacing.segment import check_empty_mask, check_totalsegmentator, head_volume_ml, run_totalsegmentator

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="stand-in executable is a POSIX script")

SCRIPT = """#!{python}
import sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("-o") + 1])
mode = {mode!r}
print("TotalSegmentator stand-in", " ".join(args))
if mode == "license" and "face" in args:
    print("ERROR: License number has not been set.")
    sys.exit(1)
out.mkdir(parents=True, exist_ok=True)
if mode == "ok" and "face" in args:
    (out / "face.nii.gz").write_bytes(b"mask")
if "total" in args and mode != "nomask":
    import nibabel as nib, numpy as np
    assert args[args.index("--roi_subset") + 1:args.index("--roi_subset") + 3] == ["skull", "brain"]
    assert "--fastest" in args
    for name, n in (("skull", 3000), ("brain", 9000)):          # 1 ml voxels: 3 l and 9 l, or nothing
        arr = np.zeros((40, 40, 40), np.uint8)
        arr.flat[:n if mode == "ok" else 0] = 1
        nib.save(nib.Nifti1Image(arr, np.diag([10.0, 10.0, 10.0, 1.0])), str(out / (name + ".nii.gz")))
"""


@pytest.fixture
def fake_exe(tmp_path, monkeypatch):
    def make(mode):
        bindir = tmp_path / f"bin_{mode}"
        bindir.mkdir()
        exe = bindir / "TotalSegmentator"
        exe.write_text(SCRIPT.format(python=sys.executable, mode=mode))
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        return exe
    return make


def test_missing_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert "not found" in check_totalsegmentator()


def test_success_returns_the_mask_and_logs_the_command(fake_exe, tmp_path):
    fake_exe("ok")
    assert check_totalsegmentator() is None
    log = tmp_path / "qc" / "tool.log"
    mask = run_totalsegmentator(tmp_path / "ct.nii.gz", tmp_path / "seg", "cpu", log)
    assert mask == tmp_path / "seg" / "face.nii.gz"
    text = log.read_text()
    assert "-ta face" in text and "-d cpu" in text and "stand-in" in text


def test_failure_mentions_the_license(fake_exe, tmp_path):
    fake_exe("license")
    with pytest.raises(PatientError, match="license missing"):
        run_totalsegmentator(tmp_path / "ct.nii.gz", tmp_path / "seg", "cpu", tmp_path / "tool.log")


def test_missing_mask(fake_exe, tmp_path):
    fake_exe("nomask")
    with pytest.raises(PatientError, match="not found"):
        run_totalsegmentator(tmp_path / "ct.nii.gz", tmp_path / "seg", "cpu", tmp_path / "tool.log")


def test_head_volume(fake_exe, tmp_path):
    fake_exe("ok")
    assert head_volume_ml(tmp_path / "ct.nii.gz", tmp_path / "seg", "cpu", tmp_path / "tool.log") == 12000.0
    fake_exe("nohead")
    assert head_volume_ml(tmp_path / "ct.nii.gz", tmp_path / "seg2", "cpu", tmp_path / "tool.log") == 0.0
    fake_exe("nomask")
    with pytest.raises(PatientError, match="head masks not found"):
        head_volume_ml(tmp_path / "ct.nii.gz", tmp_path / "seg3", "cpu", tmp_path / "tool.log")


def test_check_empty_mask(fake_exe, tmp_path):
    args = (tmp_path / "ct.nii.gz", tmp_path / "seg", "cpu", tmp_path / "tool.log", "CT series 1/1")
    assert check_empty_mask(250.0, False, *args) == []
    with pytest.raises(PatientError, match="--allow-empty-mask"):
        check_empty_mask(0.0, False, *args)
    fake_exe("ok")
    with pytest.raises(PatientError, match="includes the head"):
        check_empty_mask(0.0, True, *args)
    fake_exe("nohead")
    flags = check_empty_mask(0.0, True, tmp_path / "ct.nii.gz", tmp_path / "seg4", "cpu", tmp_path / "tool.log",
                             "CT series 2/2")
    assert len(flags) == 1 and "no head in the scan" in flags[0] and "CT series 2/2" in flags[0]
