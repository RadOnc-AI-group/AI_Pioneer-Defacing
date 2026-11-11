"""
Target volumes (GTV, CTV, PTV, ITV, Lymphabflusswege, Zielvolumen): which ROIs they are,
and which voxels of the de-faced region stay unchanged because they belong to one.

A ROI counts as a target by its RTROIInterpretedType (PTV, CTV, GTV) or by its name, which
must START with a target word (German terms included). Optimisation and helper structures
("z_PTV", "PTV_opt", "PTV70-PTV60", "Ring_PTV") and everything classify_roi reports as
Body, Skin or unclear ("Body-PTV") are not targets: keeping one of those could preserve
much of the face. Their names are reported, and --keep-roi can overrule the decision.

A ROI that covers a large part of the tissue in the face mask is probably mislabelled, see
TOO_LARGE_SHARE and LARGE_SHARE. (Most of a face mask is air, so the share is taken of the tissue.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .common import Options
from .masks import BODY_HU, Axes
from .rt import Grid, classify_roi, normalise, plan_roi_removal, roi_slices, roi_table

TOO_LARGE_SHARE = 0.75      # a target covering this share of the face-mask tissue or more is NOT kept (reported)
LARGE_SHARE = 0.25          # ... this share or more is kept, and reported
SKIN_DEPTH_MM = 5.0         # kept tissue with at most this much tissue in front of it lies at the skin
EXPOSED_CHECK_CM2 = 5.0     # a kept skin area (seen from the front) above this is reported

TARGET_TYPES = ("PTV", "CTV", "GTV")

_SEP = r"[\s_\-.]*"
_TARGET_NAME = re.compile(
    "(?:"
    r"(?:ptv|ctv|gtv|itv)(?:[pnt]|lk|law|boost|high|low|hr|ir|lr|ges\w*)?"  # PTV_70, GTVp, CTVn, GTVLK, PTVges
    r"|law(?:re|li|bds)?|lymphabfluss\w*"                                   # LAW, LAW_re, Lymphabflusswege
    r"|zv|zielvol\w*"                                                       # ZV1, Zielvolumen
    rf"|planungs{_SEP}zielvol\w*|klinisch\w*{_SEP}zielvol\w*"
    rf"|makroskopisch\w*{_SEP}tumor\w*"
    rf"|planning{_SEP}target\w*|clinical{_SEP}target\w*|internal{_SEP}target\w*|gross{_SEP}tumou?r\w*"
    r")(?![a-z])")
_TARGET_WORD = re.compile(r"ptv|ctv|gtv|(?<![a-z])(?:itv|law|zv)(?![a-z])")   # anywhere in the name
_HELPER = re.compile(
    r"^(?:z|zz|x|xx)[\s_\-.]|^[_#.]"                                  # z_PTV, x_CTV, _PTV
    r"|ring|avoid|nonptv|non[\s_\-]*(?:ptv|ctv|gtv|target)|not[\s_\-]*(?:ptv|ctv|gtv)"
    r"|(?<![a-z])opt(?:i\w*)?(?![a-z])|hilf|help|dummy|kontroll|control"
    r"|[\s_\-.]minus[\s_\-.]|[\s_\-.]ohne[\s_\-.]|\+"                 # expansions and subtractions
    r"|-\s*(?:ptv|ctv|gtv|itv|zv|law)")                               # PTV70-PTV60, PTV-GTV


def classify_target(name: str, kind: str) -> str:
    """'target', 'rejected' (looks like one, but is a helper, or Body/Skin/unclear) or ''."""
    n = normalise(name)
    if kind in TARGET_TYPES or _TARGET_NAME.match(n):
        return "rejected" if classify_roi(name, kind) or _HELPER.search(n) else "target"
    return "rejected" if _TARGET_WORD.search(n) else ""


@dataclass
class TargetPlan:
    keep: dict[int, str] = field(default_factory=dict)      # ROINumber -> ROI name
    forced: set[int] = field(default_factory=set)           # kept because of --keep-roi
    rejected: list[str] = field(default_factory=list)       # look like targets, but are not kept


def plan_targets(rois: dict[int, tuple[str, str]], opts: Options) -> TargetPlan:
    """Which ROIs of a structure set count as targets. A contour that is removed from the
    structure set or is Body/Skin is never kept, whatever --keep-roi says."""
    plan = TargetPlan()
    removed = plan_roi_removal(rois, opts).remove
    for number, (name, kind) in rois.items():
        if number in removed or classify_roi(name, kind) in ("body", "skin"):
            continue
        if any(p.fullmatch(name) for p in opts.not_target):
            continue
        if any(p.fullmatch(name) for p in opts.keep_roi):
            plan.keep[number] = name
            plan.forced.add(number)
        else:
            verdict = classify_target(name, kind)
            if verdict == "target":
                plan.keep[number] = name
            elif verdict == "rejected":
                plan.rejected.append(name)
    return plan


@dataclass
class KeptTargets:
    keep: np.ndarray | None = None                          # voxels of the region that stay unchanged
    rows: list[dict] = field(default_factory=list)          # one per target inside the region
    flags: list[str] = field(default_factory=list)
    kept_ml: float = 0.0
    exposed_cm2: float = 0.0


def exposed_area_cm2(hu: np.ndarray, keep: np.ndarray, axes: Axes) -> float:
    """
    Area, seen from the front, of the kept voxels that lie at the skin of the input: lines
    along the anterior axis on which the most anterior kept tissue has at most SKIN_DEPTH_MM
    of tissue in front of it. A 3D rendering of the output shows the skin there, or a surface
    that follows it. Thin objects in front of the face (a thermoplastic mask) do not hide it.
    """
    axis, increasing = axes.anterior
    tissue, kept = hu > BODY_HU, keep & (hu > BODY_HU)
    if increasing:                                          # index 0 = most anterior
        tissue, kept = np.flip(tissue, axis), np.flip(kept, axis)
    tissue, kept = np.moveaxis(tissue, axis, 0), np.moveaxis(kept, axis, 0)
    has = kept.any(axis=0)
    if not has.any():
        return 0.0
    first = np.argmax(kept, axis=0)
    before = np.take_along_axis(np.cumsum(tissue, axis=0, dtype=np.int32), first[None], 0)[0] - 1
    lines = has & (before * axes.spacing[axis] <= SKIN_DEPTH_MM)
    others = [s for i, s in enumerate(axes.spacing) if i != axis]
    return float(lines.sum()) * others[0] * others[1] / 100.0


def kept_target_voxels(structs: list[tuple[str, object]], grid: Grid, face: np.ndarray, region: np.ndarray,
                       hu: np.ndarray, axes: Axes, opts: Options, label: str) -> KeptTargets:
    """The voxels of `region` inside the targets of the structure sets drawn on this CT series.
    `face` is the face mask itself on the grid (no margin, no extension)."""
    out = KeptTargets()
    face_tissue = face & (hu > BODY_HU)
    face_voxels = max(int(face_tissue.sum()), 1)
    if not region.any() or not face.any():
        return out
    voxel_ml = float(np.prod(grid.spacing)) / 1000.0
    keep = np.zeros_like(region)
    for rs_name, ds in structs:
        rois = roi_table(ds)
        plan = plan_targets(rois, opts)
        if not plan.keep and not plan.rejected:
            continue
        replaced = []
        for number, slices in roi_slices(ds, grid):
            if number not in plan.keep:
                if number in rois and rois[number][0] in plan.rejected:
                    inside = sum(int((m & region[z]).sum()) for z, m in slices.items())
                    if inside:
                        replaced.append(f"{rois[number][0]!r} ({inside * voxel_ml:.1f} ml)")
                continue
            name = plan.keep[number]
            inside = sum(int((m & region[z]).sum()) for z, m in slices.items())
            if not inside:
                continue
            share = sum(int((m & face_tissue[z]).sum()) for z, m in slices.items()) / face_voxels
            kept = share < TOO_LARGE_SHARE or number in plan.forced
            what = f"{label}: target {name!r} covers {100 * share:.0f}% of the tissue in the face mask"
            if not kept:
                out.flags.append(f"{what}: NOT kept, probably not a true target (keep it with --keep-roi NAME if "
                                 "it is one)")
            elif share >= LARGE_SHARE:
                out.flags.append(f"{what}: kept, so much of the face may remain; check the QC picture")
            if kept:
                for z, m in slices.items():
                    keep[z] |= m & region[z]
            out.rows.append({"rtstruct": rs_name, "roi": name, "ml_in_region": round(inside * voxel_ml, 2),
                             "pct_of_face_tissue": round(100 * share, 1), "kept": kept})
        if replaced:
            out.flags.append(f"{label}: ROI(s) that look like targets but count as helpers are replaced in the de-faced "
                             f"region: {', '.join(replaced)} (if one is a true target: --keep-roi NAME)")
    if keep.any():
        out.keep = keep
        out.kept_ml = round(float(keep.sum()) * voxel_ml, 2)
        out.exposed_cm2 = round(exposed_area_cm2(hu, keep, axes), 1)
        if out.exposed_cm2 > EXPOSED_CHECK_CM2:
            out.flags.append(f"{label}: kept target voxels show {out.exposed_cm2:.1f} cm2 of the skin surface (seen "
                             "from the front), so that part of the face remains; check the QC picture "
                             "(--deface-targets replaces them)")
    return out
