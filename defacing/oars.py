"""
Organs at risk in the de-faced region (--keep-oar, --keep-oar-ts): their voxels are restored
below the skin, like the target volumes, so that their contours still describe the image.

Which structures: every ROI of the RTSTRUCT typed ORGAN (RTROIInterpretedType), and with
--keep-oar-ts also TotalSegmentator's eyes, lenses, optic nerves, large salivary glands and
mandible. Not restored: anything that is a target, Body, Skin or unclear, helper structures
and planning risk volumes, and teeth (dental features identify a person).

Only voxels at least --keep-oar-depth-mm below the skin of the input are restored: the
facial surface, which is what identifies a person, stays removed. Lips are thin, so little
of them remains.

A structure that covers TOO_LARGE_SHARE of the tissue in the face mask is not an organ but a
face or head contour (the share is taken of the tissue, because most of a face mask is air).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .anatomy import segment_open_tasks
from .common import Options, log
from .masks import BODY_HU, Axes
from .rt import Grid, classify_roi, normalise, plan_roi_removal, roi_slices, roi_table
from .targets import KeptTargets, classify_target, exposed_area_cm2

ORGAN_TYPE = "ORGAN"
DEFAULT_DEPTH_MM = 5.0      # the depth used for the exposed skin area of targets as well
TOO_LARGE_SHARE = 0.25      # of the tissue in the face mask: a real organ covers far less
THIN_MM = 2.0               # sheets up to twice this thick (thermoplastic mask) are not the patient

TS_OARS = ("eye_left", "eye_right", "eye_lens_left", "eye_lens_right", "optic_nerve_left", "optic_nerve_right",
           "parotid_gland_left", "parotid_gland_right", "submandibular_gland_left", "submandibular_gland_right",
           "mandible")

_HELPER = re.compile(
    r"^(?:z|zz|x|xx)[\s_\-.]|^[_#.]"
    r"|ring|avoid|prv|hilf|help|dummy|kontroll|control"
    r"|(?<![a-z])opt(?:imi[sz]\w*)?(?![a-z])"                         # "opt", "optimised", not "opticus"
    r"|[\s_\-.]minus[\s_\-.]|[\s_\-.]ohne[\s_\-.]|\+"
    r"|-\s*(?:ptv|ctv|gtv|itv|zv|law)")
_TEETH = re.compile(r"teeth|tooth|dental|dent(?:es|ition)|zaehne|zahn|gebiss")

Candidate = tuple[str, str, np.ndarray]     # source, name, boolean mask on the image grid


@dataclass
class OarPlan:
    keep: dict[int, str] = field(default_factory=dict)      # ROINumber -> ROI name
    skipped: list[str] = field(default_factory=list)        # typed ORGAN, but not restored


def plan_oars(rois: dict[int, tuple[str, str]], opts: Options) -> OarPlan:
    """Which ROIs of a structure set are restored by --keep-oar."""
    plan = OarPlan()
    removed = plan_roi_removal(rois, opts).remove
    for number, (name, kind) in rois.items():
        if kind != ORGAN_TYPE or number in removed or any(p.fullmatch(name) for p in opts.not_oar + opts.keep_roi):
            continue                                            # --keep-roi keeps the whole ROI already
        n = normalise(name)
        if classify_roi(name, kind) or classify_target(name, kind) or _HELPER.search(n) or _TEETH.search(n):
            plan.skipped.append(name)
        else:
            plan.keep[number] = name
    return plan


def thin_free_disk(spacing: tuple[float, float]) -> np.ndarray:
    """In-plane structuring element: a binary opening with it removes sheets thinner than 2 * THIN_MM."""
    ry, rx = int(THIN_MM / spacing[0]), int(THIN_MM / spacing[1])
    y, x = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    return (y * spacing[0]) ** 2 + (x * spacing[1]) ** 2 <= THIN_MM ** 2


def below_skin(hu: np.ndarray, voxels: np.ndarray, axes: Axes, depth_mm: float) -> np.ndarray:
    """
    The part of `voxels` at least depth_mm from the outside of the patient. Air inside the
    head (sinuses) does not count as outside, and neither does the air under a thermoplastic
    mask: thin sheets are removed first, then each axial slice is filled.
    """
    if depth_mm <= 0 or not voxels.any():
        return voxels.copy()
    si = axes.si_axis
    tissue, wanted = np.moveaxis(hu > BODY_HU, si, 0), np.moveaxis(voxels, si, 0)
    spacing = np.array([axes.spacing[si], *(s for i, s in enumerate(axes.spacing) if i != si)])
    reach = np.ceil(np.ceil(depth_mm) / spacing).astype(int) + 1
    where = np.nonzero(wanted)
    lo = [max(0, int(c.min()) - r) for c, r in zip(where, reach)]
    hi = [min(n, int(c.max()) + r + 1) for c, r, n in zip(where, reach, wanted.shape)]
    disk = thin_free_disk((spacing[1], spacing[2]))
    body = np.zeros((hi[0] - lo[0], *wanted.shape[1:]), bool)
    for k in range(lo[0], hi[0]):
        body[k - lo[0]] = ndi.binary_fill_holes(ndi.binary_opening(tissue[k], structure=disk))
    body = body[:, lo[1]:hi[1], lo[2]:hi[2]]
    box = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    deep = np.zeros_like(wanted)
    # a nearest outside voxel beyond the box would be further away than depth_mm
    # (the distance runs to the centre of the air voxel: half a voxel less to the surface)
    depth = np.inf if body.all() else ndi.distance_transform_edt(body, sampling=spacing) - spacing.min() / 2
    deep[box] = wanted[box] & (depth >= depth_mm)
    return np.moveaxis(deep, 0, si)


def restore(candidates: Iterable[Candidate], face: np.ndarray, region: np.ndarray, hu: np.ndarray, axes: Axes,
            opts: Options, label: str, exclude: np.ndarray | None = None) -> KeptTargets:
    """The voxels of `region` to restore. `face` is the face mask without margin or extension,
    `exclude` the voxels that are kept anyway (targets)."""
    out = KeptTargets()
    voxel_ml = float(np.prod(axes.spacing)) / 1000.0
    face_tissue = face & (hu > BODY_HU)
    face_voxels = int(face_tissue.sum())
    found = 0
    keep = np.zeros_like(region)
    if face_voxels and region.any():
        deep = below_skin(hu, region, axes, opts.keep_oar_depth_mm)
        if exclude is not None:
            deep &= ~exclude
        for source, name, mask in candidates:
            found += 1
            inside = int((mask & region).sum())
            if not inside:
                continue
            share = int((mask & face_tissue).sum()) / face_voxels
            kept = share < TOO_LARGE_SHARE
            part = mask & deep if kept else np.zeros_like(mask)
            if kept:
                keep |= part
            else:
                out.flags.append(f"{label}: OAR {name!r} covers {100 * share:.0f}% of the tissue in the face mask: NOT "
                                 "restored, probably a face or head contour (--keep-roi NAME keeps all of it, "
                                 "--not-oar NAME silences this)")
            out.rows.append({"source": source, "roi": name, "ml_in_region": round(inside * voxel_ml, 2),
                             "ml_restored": round(int(part.sum()) * voxel_ml, 2),
                             "pct_of_face_tissue": round(100 * share, 1), "kept": kept})
    if face_voxels and not found and not opts.keep_oar_ts:
        out.flags.append(f"{label}: --keep-oar found no ROI typed ORGAN, so no organ at risk was restored "
                         "(--keep-oar-ts uses TotalSegmentator instead)")
    if keep.any():
        out.keep = keep
        out.kept_ml = round(float(keep.sum()) * voxel_ml, 2)
        out.exposed_cm2 = round(exposed_area_cm2(hu, keep, axes), 1)
        out.flags.append(f"{label}: {out.kept_ml:.1f} ml of OAR voxels restored at least {opts.keep_oar_depth_mm:g} mm "
                         f"below the skin ({out.exposed_cm2:.1f} cm2 near the skin seen from the front); check the QC "
                         "picture (--face-check measures whether a face can still be found)")
        share = int((keep & face_tissue).sum()) / face_voxels
        if share >= TOO_LARGE_SHARE:
            out.flags.append(f"{label}: the restored OARs together cover {100 * share:.0f}% of the tissue in the face "
                             "mask, so much of the face shape may remain; check the QC picture (--not-oar NAME)")
    return out


def rtstruct_candidates(structs: list[tuple[str, object]], grid: Grid, shape: tuple, opts: Options) -> Iterator[Candidate]:
    for _, ds in structs:
        plan = plan_oars(roi_table(ds), opts)
        if not plan.keep:
            continue
        for number, slices in roi_slices(ds, grid):
            if number in plan.keep:
                mask = np.zeros(shape, bool)
                for z, m in slices.items():
                    mask[z] |= m
                yield "rtstruct", plan.keep[number], mask


def ts_candidates(input_nii: Path, tmp: Path, tool_log: Path, opts: Options,
                  load: Callable[[Path], np.ndarray]) -> Iterator[Candidate]:
    """TotalSegmentator's organs at risk; the folder is shared with --anatomy-check."""
    seg_dir = segment_open_tasks(input_nii, tmp / "anatomy_seg", opts.device, tool_log)
    for name in TS_OARS:
        path = seg_dir / f"{name}.nii.gz"
        if path.exists() and not any(p.fullmatch(name) for p in opts.not_oar):
            yield "totalsegmentator", name, load(path)


def keep_oars(structs: list | None, grid: Grid | None, input_nii: Path, tmp: Path, tool_log: Path, face: np.ndarray,
              region: np.ndarray, hu: np.ndarray, axes: Axes, opts: Options, label: str,
              load: Callable[[Path], np.ndarray], exclude: np.ndarray | None = None) -> KeptTargets:
    """--keep-oar for one image: the contours of `structs` and, with --keep-oar-ts, TotalSegmentator."""
    def candidates() -> Iterator[Candidate]:
        if structs and grid is not None:
            yield from rtstruct_candidates(structs, grid, region.shape, opts)
        if opts.keep_oar_ts:
            yield from ts_candidates(input_nii, tmp, tool_log, opts, load)
    result = restore(candidates(), face, region, hu, axes, opts, label, exclude)
    for row in result.rows:
        log.info(f"    {label}: OAR {row['roi']!r} ({row['source']}): {row['ml_restored']:.2f} of "
                 f"{row['ml_in_region']:.2f} ml in the region restored")
    return result
