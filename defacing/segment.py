"""Face segmentation with TotalSegmentator."""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

from .common import PatientError


def check_totalsegmentator() -> str | None:
    """None if TotalSegmentator can be run, else a message."""
    if shutil.which("TotalSegmentator") is None:
        return "'TotalSegmentator' not found on PATH. Run: conda activate totalseg-deface"
    return None


def run_totalsegmentator(input_nii: Path, seg_dir: Path, device: str, log_path: Path) -> Path:
    """Segment the face; TotalSegmentator's output goes to log_path. Returns the mask path."""
    # Note: the "face" task does not support --fast (TotalSegmentator raises ValueError).
    cmd = ["TotalSegmentator", "-i", str(input_nii), "-o", str(seg_dir), "-ta", "face", "-d", device]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        start = lf.tell()
        lf.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace")[start:]
        hint = ""
        if "license" in text.lower():
            hint = " (license missing? Run: totalseg_set_license -l <KEY>)"
        last = [ln for ln in text.strip().splitlines() if ln.strip()][-1:] or [""]
        raise PatientError(f"TotalSegmentator failed (exit {proc.returncode}){hint}: {last[0][:200]} "
                           f"(full output: {log_path})")
    mask_path = seg_dir / "face.nii.gz"
    if not mask_path.exists():
        raise PatientError(f"Expected face mask not found at {mask_path}")
    return mask_path
