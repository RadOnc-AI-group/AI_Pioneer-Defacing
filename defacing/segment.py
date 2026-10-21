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


# Classes of the "total" task that show that a scan includes the head; this task needs
# no license. Used to decide whether an empty face mask is plausible.
HEAD_CLASSES = ("skull", "brain")
HEAD_MIN_ML = 20.0          # less than this: no head in the scan


def _run(cmd: list[str], log_path: Path) -> None:
    """Run TotalSegmentator; its output goes to log_path."""
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


def run_totalsegmentator(input_nii: Path, seg_dir: Path, device: str, log_path: Path) -> Path:
    """Segment the face; TotalSegmentator's output goes to log_path. Returns the mask path."""
    # Note: the "face" task does not support --fast (TotalSegmentator raises ValueError).
    _run(["TotalSegmentator", "-i", str(input_nii), "-o", str(seg_dir), "-ta", "face", "-d", device], log_path)
    mask_path = seg_dir / "face.nii.gz"
    if not mask_path.exists():
        raise PatientError(f"Expected face mask not found at {mask_path}")
    return mask_path


def head_volume_ml(input_nii: Path, seg_dir: Path, device: str, log_path: Path) -> float:
    """Volume (ml) of skull and brain found by the fastest (6 mm) "total" model: tells whether
    the scan includes the head, i.e. whether there should be a face."""
    from .masks import mask_volume_ml
    _run(["TotalSegmentator", "-i", str(input_nii), "-o", str(seg_dir), "-ta", "total", "--fastest",
          "--roi_subset", *HEAD_CLASSES, "-d", device], log_path)
    paths = [seg_dir / f"{name}.nii.gz" for name in HEAD_CLASSES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise PatientError(f"Expected head masks not found: {', '.join(str(p) for p in missing)}")
    return sum(mask_volume_ml(p) for p in paths)


def check_empty_mask(face_ml: float, allow_empty_mask: bool, input_nii: Path, seg_dir: Path, device: str,
                     log_path: Path, label: str) -> list[str]:
    """
    An empty face mask fails the series, unless --allow-empty-mask is given AND the scan
    includes no head (skull, brain), e.g. a chest CT next to the head CT of the same patient.
    Returns the flags for the report.
    """
    if face_ml > 0:
        return []
    msg = f"face mask is EMPTY for {label}: nothing was removed"
    if not allow_empty_mask:
        raise PatientError(msg + " (use --allow-empty-mask if the scan includes no head)")
    head_ml = head_volume_ml(input_nii, seg_dir, device, log_path)
    if head_ml >= HEAD_MIN_ML:
        raise PatientError(f"face mask is EMPTY for {label} although the scan includes the head (skull/brain "
                           f"{head_ml:.0f} ml): the face was not found; check the scan in a viewer "
                           "(nothing was written for this patient)")
    return [f"{msg} (accepted: no head in the scan, skull/brain {head_ml:.0f} ml)"]
