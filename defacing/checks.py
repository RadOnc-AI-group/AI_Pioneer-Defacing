"""The optional checks of a de-faced image: what they need, and their results."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .common import Options
from .masks import Axes
from .timing import Timer


def run_optional_checks(opts: Options, before: np.ndarray, after: np.ndarray, full_region: np.ndarray,
                        final_region: np.ndarray, axes: Axes, ct_nii: Path, tmp: Path, tool_log: Path, label: str,
                        load_mask: Callable[[Path], np.ndarray], timer: Timer) -> tuple[dict, list[str]]:
    """--anatomy-check and --face-check of one image. `full_region` is the de-faced area including
    kept target voxels, `final_region` the voxels actually replaced. Returns (results, CHECK lines)."""
    results: dict = {}
    flags: list[str] = []
    if opts.anatomy_check:
        from . import anatomy
        with timer.stage("anatomy_check"):
            seg_dir = anatomy.segment_landmarks(ct_nii, tmp / "anatomy_seg", opts.device, tool_log)
            results["anatomy"], found = anatomy.assess(before, full_region, final_region, axes, load_mask, seg_dir, label)
        flags += found
    if opts.face_check:
        from . import reid
        with timer.stage("face_check"):
            results["face_check"], found = reid.assess(before, after, full_region, axes, opts, label)
        flags += found
    return results, flags
