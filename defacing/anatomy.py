"""
Anatomy check (--anatomy-check): does tissue remain in front of structures that lie behind
the face? TotalSegmentator's open tasks (no license needed) give landmarks that the face
mask does not depend on. The external nose lies in front of the nasal cavity, the lips in
front of the teeth, the eyelids in front of the lenses, the chin in front of the mandible.

The check catches a partial face mask, which the other checks cannot: unchanged tissue in
front of a landmark means the mask missed what lies there. It also reports brain inside the
de-faced region (over-removal) and a face that is not in front of the brain (orientation).

It only adds CHECK lines. The thresholds are starting values and should be calibrated on
real scans of each protocol: a mask that ends just behind the lips, for example, leaves
some tissue in front of the teeth without showing the face.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .masks import BODY_HU, Axes
from .segment import run_task

LANDMARK_TASKS = ("head_glands_cavities", "craniofacial_structures")
MIN_AREA_CM2 = 0.5          # tissue in front of a landmark over less than this is ignored
BRAIN_CHECK_ML = 5.0        # more brain than this inside the de-faced region is reported


@dataclass(frozen=True)
class Landmark:
    feature: str
    masks: tuple[str, ...]
    behind: str             # text of the CHECK line
    tissue_mm: float        # at least this much tissue in front of the landmark is reported
    check: bool = True      # False: only reported in the manifest


LANDMARKS = (Landmark("nose", ("nasal_cavity_left", "nasal_cavity_right"), "nasal cavity", 5.0),
             Landmark("lips", ("teeth_upper", "teeth_lower"), "teeth", 5.0),
             Landmark("eyelids", ("eye_lens_left", "eye_lens_right"), "lenses", 2.0),
             Landmark("chin", ("mandible",), "mandible", 5.0, check=False))


def segment_landmarks(input_nii: Path, seg_dir: Path, device: str, log_path: Path) -> Path:
    """Run the open TotalSegmentator tasks; every structure ends up as <name>.nii.gz in seg_dir."""
    for task in LANDMARK_TASKS:
        run_task(task, input_nii, seg_dir, device, log_path)
    run_task("total", input_nii, seg_dir, device, log_path, ("--fastest", "--roi_subset", "skull", "brain"))
    return seg_dir


def _anterior_first(vol: np.ndarray, axes: Axes) -> np.ndarray:
    """vol with the anterior axis first and index 0 at the most anterior voxels."""
    axis, increasing = axes.anterior
    return np.moveaxis(np.flip(vol, axis) if increasing else vol, axis, 0)


def tissue_in_front(landmark: np.ndarray, tissue: np.ndarray, axes: Axes) -> tuple[np.ndarray, np.ndarray]:
    """(lines through the landmark, tissue in mm in front of it on each line); 2D arrays."""
    ap = axes.spacing[axes.anterior[0]]
    mark, unchanged = _anterior_first(landmark, axes), _anterior_first(tissue, axes)
    lines = mark.any(axis=0)
    first = np.argmax(mark, axis=0)
    cum = np.cumsum(unchanged, axis=0, dtype=np.int32)
    upto = np.take_along_axis(cum, first[None], 0)[0] - np.take_along_axis(unchanged, first[None], 0)[0]
    return lines, upto * ap


def assess(hu: np.ndarray, full_region: np.ndarray, final_region: np.ndarray, axes: Axes,
           load: Callable[[Path], np.ndarray], seg_dir: Path, label: str) -> tuple[dict, list[str]]:
    """Measurements for the report and CHECK lines. `load` reads a mask onto the image grid."""
    results: dict = {}
    flags: list[str] = []
    tissue = (hu > BODY_HU) & ~full_region           # unchanged tissue; kept targets are reported separately
    ap = axes.anterior[0]
    pixel_cm2 = float(np.prod([s for i, s in enumerate(axes.spacing) if i != ap])) / 100.0
    centres: dict[str, float] = {}
    for mark in LANDMARKS:
        paths = [seg_dir / f"{name}.nii.gz" for name in mark.masks]
        masks = [load(p) for p in paths if p.exists()]
        landmark = np.any(masks, axis=0) if masks else None
        if landmark is None or not landmark.any():
            results[mark.feature] = {"landmark_found": False}
            continue
        lines, front_mm = tissue_in_front(landmark, tissue, axes)
        over = lines & (front_mm >= mark.tissue_mm)
        area = float(over.sum()) * pixel_cm2
        results[mark.feature] = {"landmark_found": True, "max_tissue_mm": round(float(front_mm[lines].max()), 1),
                                 "area_cm2": round(area, 2)}
        if mark.check and area >= MIN_AREA_CM2:
            flags.append(f"{label}: anatomy check: up to {results[mark.feature]['max_tissue_mm']:.0f} mm of unchanged "
                         f"tissue in front of the {mark.behind} over {area:.1f} cm2: the {mark.feature} may not be "
                         "removed; check the QC picture")
        if mark.feature == "nose":
            centres["nose"] = float(np.nonzero(_anterior_first(landmark, axes))[0].mean())

    brain_path = seg_dir / "brain.nii.gz"
    brain = load(brain_path) if brain_path.exists() else None
    if brain is not None and brain.any():
        ml = float((brain & final_region).sum()) * float(np.prod(axes.spacing)) / 1000.0
        results["brain_in_region_ml"] = round(ml, 1)
        if ml > BRAIN_CHECK_ML:
            flags.append(f"{label}: anatomy check: {ml:.0f} ml of brain inside the de-faced region; check the "
                         "QC picture")
        centres["brain"] = float(np.nonzero(_anterior_first(brain, axes))[0].mean())
    if len(centres) == 2:
        results["face_in_front_of_brain"] = centres["nose"] < centres["brain"]    # index 0 = most anterior
        if not results["face_in_front_of_brain"]:
            flags.append(f"{label}: anatomy check: the nasal cavity is not in front of the brain: the orientation "
                         "in the header is probably wrong")
    return results, flags


def summary(results: dict) -> str:
    """One short text for the patients report."""
    parts = []
    for mark in LANDMARKS:
        r = results.get(mark.feature, {})
        parts.append(f"{mark.feature} {r['max_tissue_mm']:.0f}mm" if r.get("landmark_found")
                     else f"{mark.feature} n/a")
    if "brain_in_region_ml" in results:
        parts.append(f"brain {results['brain_in_region_ml']:.0f}ml")
    return " ".join(parts)
