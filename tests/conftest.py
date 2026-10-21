"""
Shared fixtures. TotalSegmentator is replaced by a stand-in that marks the phantom's face
region (tests/phantom.py), so the tests need neither the model, a GPU nor a license.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phantom as ph  # noqa: E402

import defacing.ct  # noqa: E402
import defacing.patient  # noqa: E402
import defacing.segment  # noqa: E402


class FakeTotalSegmentator:
    """Records its calls; `mode` selects the mask: 'face', 'empty', 'two_parts' (the face and a
    false detection below it) or 'back' (the back of the head instead of the face, as if the
    orientation in the header were wrong)."""

    def __init__(self):
        self.mode = "face"
        self.calls: list[Path] = []
        self.head_calls: list[Path] = []

    def __call__(self, input_nii: Path, seg_dir: Path, device: str, log_path: Path) -> Path:
        self.calls.append(Path(input_nii))
        out = Path(seg_dir) / "face.nii.gz"
        img = nib.load(str(input_nii))
        if self.mode == "two_parts":                              # face + a false detection below it
            world = ph.nifti_world_lps(img.affine, img.shape)
            extra = ph.in_box(world, ((-20, 20), (-62, -40), (-72, -58))) & ph.inside_head(world)
            mask = (ph.in_face(world) | extra).astype(np.uint8)
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask, img.affine), str(out))
        elif self.mode == "back":
            world = ph.nifti_world_lps(img.affine, img.shape)
            world = world * np.array([1.0, -1.0, 1.0])            # mirrored front to back
            mask = ph.in_face(world).astype(np.uint8)
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask, img.affine), str(out))
        else:
            ph.fake_face_mask(input_nii, out, empty=self.mode == "empty")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"fake TotalSegmentator {input_nii}\n")
        return out


    def head_volume_ml(self, input_nii: Path, seg_dir: Path, device: str, log_path: Path) -> float:
        """Like the "total" model: volume of the phantom's head (skull and brain) in the image."""
        self.head_calls.append(Path(input_nii))
        img = nib.load(str(input_nii))
        voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        return float(ph.inside_head(ph.nifti_world_lps(img.affine, img.shape)).sum()) * voxel_ml


@pytest.fixture
def fake_ts(monkeypatch) -> FakeTotalSegmentator:
    fake = FakeTotalSegmentator()
    monkeypatch.setattr(defacing.ct, "run_totalsegmentator", fake)
    monkeypatch.setattr(defacing.patient, "run_totalsegmentator", fake)
    monkeypatch.setattr(defacing.segment, "head_volume_ml", fake.head_volume_ml)
    return fake


@pytest.fixture
def ts_on_path(monkeypatch):
    """Pretend the TotalSegmentator executable is installed (for run_single/run_batch)."""
    import defacing.run
    monkeypatch.setattr(defacing.run, "check_totalsegmentator", lambda: None)


@pytest.fixture(autouse=True)
def _no_desktop_notification(monkeypatch):
    import defacing.run
    monkeypatch.setattr(defacing.run, "notify", lambda msg: None)


@pytest.fixture(autouse=True)
def _reset_logger():
    yield
    log = logging.getLogger("deface")
    for h in list(log.handlers):
        h.close()
    log.handlers.clear()
    log.propagate = True


def tree_hashes(root: Path) -> dict[str, str]:
    """sha256 of every file below root (to prove that the input is never modified)."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}
