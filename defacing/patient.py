"""
One patient: a DICOM folder (CT + RT objects, searched recursively) or one NIfTI/NRRD image.

Output (shareable)                         QC folder (confidential, never share)
  <out>/dcm/           de-faced CT           face_mask*.nii.gz   raw TotalSegmentator mask
  <out>/dcm/series_<k>/  (several CTs)       CT_defaced*.nii.gz  de-faced CT for viewing
  <out>/rt/            RTSTRUCT (+ kept RT)  QC_defacing*.png    QC picture (shows the face)
  <out>/<name>_deFaced.nii.gz (image input)  QC_roi_overlap*.png ROIs in or near the region
                                             removed_rois/       removed contours (Body, Skin)
                                             manifest.json, roi_overlap.csv, tool.log

The output is written to "<out>.partial" and renamed when complete, so an interrupted
run never leaves a half-written output behind.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.uid import generate_uid

from . import prior as prior_defacing
from .anatomy import summary as anatomy_summary
from .checks import run_optional_checks
from .common import Options, PatientError, classify_suffix, log, safe_cast, strip_known_suffix
from .ct import SeriesResult, deface_ct_series
from .fill import apply_fill, fill_seed
from .inventory import Inventory, inventory
from .masks import applied_mask_nifti, applied_mask_sitk, mask_volume_ml, nifti_axes, sitk_axes
from .pii import IdentifierSummary, header_findings, warn_dicom
from .qc import MAX_ROIS as MAX_ROI_PICTURE
from .qc import ROI_NEAR_MM, automatic_checks, roi_label, write_qc_picture, write_roi_picture
from .rt import (RegionStats, clip_roi, plan_roi_removal, referenced_series, remap_uids, renamed, roi_slices,
                 roi_table, rtstruct_image_refs, split_rois)
from .rtdose import mask_dose, verify_dose
from .oars import keep_oars, plan_oars
from .segment import check_empty_mask, run_totalsegmentator
from .targets import KeptTargets, classify_target, plan_targets
from .timing import Timer, machine_info

PARTIAL_SUFFIX = ".partial"
SCRIPT_VERSION = 6          # stored in the manifest; bump when the output content changes


@dataclass
class PatientResult:
    rel: str
    status: str = "failed"           # done | dry run | already done | legacy output | failed
    seconds: float = 0.0
    ct_series: int = 0
    ct_files: int = 0
    rt_files: int = 0
    not_copied: int = 0
    rtdose_removed: int = 0
    rtplan_removed: int = 0
    rois_removed: list[str] = field(default_factory=list)
    rois_unclear: list[str] = field(default_factory=list)
    face_mask_ml: list[float] = field(default_factory=list)
    fill: list[dict] = field(default_factory=list)
    overlap: list[dict] = field(default_factory=list)   # one row per kept ROI
    unresolved_refs: int = 0
    pii: list[str] = field(default_factory=list)        # populated direct identifiers
    ids: IdentifierSummary | None = None
    flags: list[str] = field(default_factory=list)
    error: str = ""
    rois_clipped: list[str] = field(default_factory=list)   # Body, cut back to the remaining image
    rtdose_masked: int = 0
    targets: list[dict] = field(default_factory=list)       # one per target inside the de-faced area
    target_kept_ml: list[float] = field(default_factory=list)      # per CT series
    target_exposed_cm2: list[float] = field(default_factory=list)  # per CT series
    oars: list[dict] = field(default_factory=list)          # one per organ at risk inside the de-faced area
    oar_kept_ml: list[float] = field(default_factory=list)  # per image
    oar_exposed_cm2: list[float] = field(default_factory=list)     # per image
    checks: list[dict] = field(default_factory=list)        # optional checks, per CT series
    timing: dict[str, float] = field(default_factory=dict)  # seconds per stage
    machine: str = ""
    device: str = ""

    def add_series(self, s: SeriesResult) -> None:
        self.face_mask_ml.append(s.face_ml)
        self.fill.append(s.fill_info)
        self.flags += s.flags
        self.checks.append(s.checks)
        if s.targets is not None:
            self.targets += s.targets.rows
            self.target_kept_ml.append(s.targets.kept_ml)
            self.target_exposed_cm2.append(s.targets.exposed_cm2)
        if s.oars is not None:
            self.add_oars(s.oars)

    def add_oars(self, oars: KeptTargets) -> None:
        self.oars += oars.rows
        self.oar_kept_ml.append(oars.kept_ml)
        self.oar_exposed_cm2.append(oars.exposed_cm2)

    def face_check_text(self) -> tuple[str, str]:
        """(status, score) of the face check per CT series, for the report."""
        results = [c["face_check"] for c in self.checks if "face_check" in c]
        return (";".join(r["status"] for r in results),
                ";".join("" if r["score"] is None else f"{r['score']:.3f}" for r in results))

    def anatomy_text(self) -> str:
        return ";".join(anatomy_summary(c["anatomy"]) for c in self.checks if "anatomy" in c)

    def add_inventory(self, inv: Inventory) -> None:
        series = inv.ct_series
        self.ct_series, self.ct_files = len(series), sum(len(v) for v in series.values())
        self.rt_files, self.not_copied = len(inv.rt), len(inv.not_copied)
        self.rtdose_removed, self.rtplan_removed, self.ids = inv.rtdose_removed, inv.rtplan_removed, inv.ids


def tool_commit() -> str:
    repo = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "deface.py", "defacing"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+modified" if dirty else "") if out.returncode == 0 else ""
    except Exception:
        return ""


def write_manifest(qc_dir: Path, res: PatientResult, opts: Options) -> None:
    (qc_dir / "manifest.json").write_text(json.dumps({
        "patient": res.rel, "script_version": SCRIPT_VERSION, "tool_commit": tool_commit(),
        "finished": dt.datetime.now().isoformat(timespec="seconds"), "device": opts.device,
        **opts.settings(), "ct_series": res.ct_series, "ct_files": res.ct_files, "rt_files": res.rt_files,
        "face_mask_ml": [round(v, 1) for v in res.face_mask_ml], "fill_details": res.fill,
        "rois_removed": res.rois_removed, "rtdose_removed": res.rtdose_removed,
        "rtplan_removed": res.rtplan_removed, "rois_clipped": res.rois_clipped, "rtdose_masked": res.rtdose_masked,
        "targets": res.targets, "target_kept_ml": res.target_kept_ml, "target_exposed_cm2": res.target_exposed_cm2,
        "oars": res.oars, "oar_kept_ml": res.oar_kept_ml, "oar_exposed_cm2": res.oar_exposed_cm2,
        "anatomy_check": opts.anatomy_check, "face_check": opts.face_check, "checks": res.checks,
        "timing_seconds": {k: round(v, 2) for k, v in res.timing.items()}, "machine": res.machine,
    }, indent=2), encoding="utf-8")


OVERLAP_COLS = ["patient", "rtstruct", "roi", "type", "voxels", "pct_in_defaced_region", "ml_in_defaced_region",
                "pct_in_face_mask", "min_distance_mm", "is_target", "ml_kept_in_region"]


def write_overlap_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OVERLAP_COLS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# DICOM patient folder
# ---------------------------------------------------------------------------

def _log_plan(inv: Inventory, res: PatientResult, opts: Options, dry_run: bool) -> None:
    verb = "would not copy" if dry_run else "not copied"
    for rel, reason in inv.not_copied:
        log.info(f"    {verb}: {rel}  ({reason})")
    if inv.headerless:
        msg = (f"{inv.headerless} file(s) hold DICOM data without the DICOM file header and were not read; "
               "convert them first (e.g. 'gdcmconv' or 'dcmconv'), see the log")
        log.warning(f"    ! {msg}")
        res.flags.append(msg)
    if opts.mask_rtdose and any(f.modality == "RTDOSE" for f in inv.rt):
        log.info(f"    RTDOSE {'would be' if dry_run else 'is'} copied with the dose set to zero in the de-faced region")
    if inv.rtdose_removed:
        log.warning(f"    ! {inv.rtdose_removed} RTDOSE file(s) {'would be ' if dry_run else ''}removed "
                    "(the dose outlines the face)")
    elif opts.keep_rtdose and any(f.modality == "RTDOSE" for f in inv.rt):
        log.warning("    ! RTDOSE kept (--keep-rtdose): the dose distribution outlines the face")
        res.flags.append("RTDOSE kept although it outlines the face (--keep-rtdose)")
    for f in inv.rt:
        if f.modality != "RTSTRUCT":
            continue
        plan = plan_roi_removal(f.rois, opts)
        if plan.remove and dry_run:
            log.info(f"    {f.rel.name}: would remove ROI(s): {', '.join(sorted(plan.remove.values()))}")
        if plan.clip and dry_run:
            log.info(f"    {f.rel.name}: would clip ROI(s): {', '.join(sorted(plan.clip.values()))}")
        if not opts.deface_targets:
            targets = plan_targets(f.rois, opts)
            if targets.keep:
                log.info(f"    {f.rel.name}: target volumes {'would be ' if dry_run else ''}kept where they lie in the "
                         f"de-faced region: {', '.join(sorted(targets.keep.values()))}")
            if targets.rejected:
                log.warning(f"    ! {f.rel.name}: look like targets but are NOT kept (helper, Body or unclear names): "
                            f"{', '.join(sorted(targets.rejected))} (keep with --keep-roi NAME)")
        if opts.keep_oar:
            oars = plan_oars(f.rois, opts)
            log.info(f"    {f.rel.name}: organs at risk {'would be ' if dry_run else ''}restored below the skin where "
                     f"they lie in the de-faced region: {', '.join(sorted(oars.keep.values())) or 'none'}")
            if oars.skipped:
                log.info(f"    {f.rel.name}: typed ORGAN but not restored (helper, teeth or unclear names): "
                         f"{', '.join(sorted(oars.skipped))}")
        if not plan.has_body and not opts.keep_body:
            res.flags.append(f"{f.rel.name}: no Body/External ROI recognised; check the structure set "
                             "for a skin contour (remove it with --remove-roi NAME)")
        if plan.unclear:
            log.warning(f"    ! {f.rel.name}: kept ROI(s) whose name suggests they may follow the skin or "
                        f"face: {', '.join(plan.unclear)}")
            res.flags.append(f"{f.rel.name}: unclear ROI name(s), kept: {', '.join(plan.unclear)} "
                             "(remove with --remove-roi NAME if they follow the skin)")
            res.rois_unclear += plan.unclear
        if dry_run:
            res.rois_removed += sorted(plan.remove.values())


def _series_for(ds, series: list[SeriesResult]) -> SeriesResult | None:
    """The de-faced CT series an RTSTRUCT was drawn on: by its referenced series (old or, after
    remapping, new UID), else by the images its contours refer to, else by the frame of
    reference if only one series has it (several series of one scan share it)."""
    refs = referenced_series(ds)
    for s in series:
        if s.old_series_uid in refs or s.new_series_uid in refs:
            return s
    images = rtstruct_image_refs(ds)
    by_images = [s for s in series if images & (set(s.sop_map) | set(s.sop_map.values()))]
    if by_images:
        return max(by_images, key=lambda s: len(images & (set(s.sop_map) | set(s.sop_map.values()))))
    frames = ds.get("ReferencedFrameOfReferenceSequence") or []
    frame = str(frames[0].get("FrameOfReferenceUID", "")) if len(frames) else ""
    same_frame = [s for s in series if frame and s.frame_uid == frame]
    return same_frame[0] if len(same_frame) == 1 else None


def _overlap_rows(res: PatientResult, rs_name: str, ds, s: SeriesResult, skip: frozenset = frozenset()
                  ) -> tuple[list[dict], list]:
    """One report row per ROI (except `skip`), and (legend text, {slice: 2D mask}) of the ROIs
    in or near the region (most affected first) for the ROI picture."""
    rows, shown = [], []
    stats = RegionStats(s.region, tuple(s.grid.spacing[::-1]))
    voxel_ml = float(np.prod(s.grid.spacing)) / 1000.0
    names = roi_table(ds)
    for number, slices in roi_slices(ds, s.grid):
        name, kind = names.get(number, (str(number), ""))
        n, inside, d = stats.measure(slices)
        if not n or number in skip:
            continue
        in_face = sum(int((m & s.face[z]).sum()) for z, m in slices.items()) if s.face is not None else 0
        distance = f">={stats.cap_mm:.0f}" if d >= stats.cap_mm else round(d, 1)
        row = {"patient": res.rel, "rtstruct": rs_name, "roi": name, "type": kind, "voxels": n,
               "pct_in_defaced_region": round(100.0 * inside / n, 1),
               "ml_in_defaced_region": round(inside * voxel_ml, 2),
               "pct_in_face_mask": round(100.0 * in_face / n, 1), "min_distance_mm": distance,
               "is_target": "yes" if classify_target(name, kind) == "target" else "",
               "ml_kept_in_region": (round(sum(int((m & s.keep[z]).sum()) for z, m in slices.items()) * voxel_ml, 2)
                                     if s.keep is not None else 0.0)}
        rows.append(row)
        if inside or d < ROI_NEAR_MM:
            text = roi_label(name, row["pct_in_defaced_region"], row["ml_in_defaced_region"], distance)
            shown.append(((-row["pct_in_defaced_region"], -inside, d), text, slices))
    shown.sort(key=lambda e: e[0])
    return rows, [(text, slices) for _, text, slices in shown]


# Files this tool writes into a patient's QC folder; only these are removed before a
# patient is (re)processed, never anything else in a user-chosen --qc-dir.
QC_FILES = ("face_mask*.nii.gz", "CT_defaced*.nii.gz", "QC_defacing*.png", "QC_roi_overlap*.png", "removed_rois",
            "roi_overlap.csv", "manifest.json", "tool.log", "patients_*.csv", "identifying_tags_*.csv",
            "roi_overlap_*.csv", "timing_*.csv")


def clear_qc(qc_dir: Path) -> None:
    for pattern in QC_FILES:
        for p in qc_dir.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def _prepare(out_dir: Path, qc_dir: Path) -> Path:
    """Remove leftovers of an interrupted attempt; return the .partial output folder."""
    partial = out_dir.with_name(out_dir.name + PARTIAL_SUFFIX)
    if partial.exists():
        shutil.rmtree(partial)
    if qc_dir.exists():
        clear_qc(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    return partial


def deface_dicom_patient(patient_dir: Path, rel: str, out_dir: Path, qc_dir: Path, opts: Options,
                         dry_run: bool = False) -> PatientResult:
    res = PatientResult(rel=rel, device=opts.device, machine=machine_info(opts.device))
    timer = Timer()
    with timer.stage("inventory"):
        inv = inventory(patient_dir, opts)
    res.add_inventory(inv)
    rt_kinds = sorted({f.modality for f in inv.rt})
    log.info(f"    {res.ct_series} CT series ({res.ct_files} files) to de-face; RT files copied: "
             f"{res.rt_files} ({', '.join(rt_kinds) or 'none'})")
    _log_plan(inv, res, opts, dry_run)
    if inv.ids is not None:
        res.pii = warn_dicom(inv.ids, Path(rel).name)
    if not inv.ct:
        raise PatientError("no CT series found")
    if dry_run:
        res.status = "dry run"
        return res

    partial = _prepare(out_dir, qc_dir)
    tool_log = qc_dir / "tool.log"
    try:
        series_results: list[SeriesResult] = []
        groups = list(inv.ct_series.values())
        structs = _structures_by_series(inv, groups, res, opts)
        for k, files in enumerate(groups):
            label = f"CT series {k + 1}/{len(groups)} ({len(files)} files"
            label += f", '{files[0].series_desc}')" if files[0].series_desc else ")"
            log.info(f"    de-facing {label} ...")
            sub = "dcm" if len(groups) == 1 else f"dcm/series_{k + 1}"
            suffix = "" if len(groups) == 1 else f"_series{k + 1}"
            with tempfile.TemporaryDirectory(prefix="deface_") as tmp:
                s = deface_ct_series(files, label, partial / sub, qc_dir, suffix, opts, Path(tmp), tool_log,
                                     structs=structs.get(files[0].series_uid), timer=timer)
            series_results.append(s)
            res.add_series(s)

        with timer.stage("rt_copy"):
            _copy_rt(inv, res, partial / "rt", qc_dir, series_results, opts)
        if res.overlap:
            write_overlap_csv(qc_dir / "roi_overlap.csv", res.overlap)
        res.timing = timer.seconds
        write_manifest(qc_dir, res, opts)         # before the rename: no output without one
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    res.status = "done"
    return res


@dataclass
class _SeriesRef:
    """What _series_for needs to know about a CT series before it is de-faced."""
    old_series_uid: str
    new_series_uid: str
    frame_uid: str
    sop_map: dict[str, str]


def _structures_by_series(inv: Inventory, groups: list, res: PatientResult, opts: Options) -> dict[str, list]:
    """(file name, dataset) of the RTSTRUCTs drawn on each CT series (old Series UID), for
    keeping their target volumes. A structure set without a matching series is reported."""
    by_series: dict[str, list] = {}
    if opts.deface_targets and not opts.keep_oar:
        return by_series
    refs = [_SeriesRef(files[0].series_uid, "", files[0].frame_uid, {f.sop_uid: f.sop_uid for f in files})
            for files in groups]

    def needed(f) -> bool:
        if f.modality != "RTSTRUCT":
            return False
        plan = plan_targets(f.rois, opts)
        return bool((not opts.deface_targets and (plan.keep or plan.rejected))
                    or (opts.keep_oar and plan_oars(f.rois, opts).keep))

    for f in inv.rt:
        if not needed(f):
            continue
        ds = pydicom.dcmread(f.path)
        ref = _series_for(ds, refs)
        if ref is None:
            res.flags.append(f"{f.rel.name}: the CT series it was drawn on was not found, so its target volumes "
                             "and organs at risk could not be kept (their voxels were de-faced)")
        else:
            by_series.setdefault(ref.old_series_uid, []).append((f.rel.name, ds))
    return by_series


def _copy_rt(inv: Inventory, res: PatientResult, rt_dir: Path, qc_dir: Path,
             series: list[SeriesResult], opts: Options) -> None:
    """Copy the RT objects with new UIDs and all references remapped; strip removed ROIs."""
    uid_map: dict[str, str] = {}
    pictures: dict[str, int] = {}                  # ROI pictures per CT series
    old_ct_sops, new_ct_sops = set(), set()
    for s in series:
        uid_map.update(s.sop_map)
        uid_map[s.old_series_uid] = s.new_series_uid
        old_ct_sops |= set(s.sop_map)
        new_ct_sops |= set(s.sop_map.values())
    for f in inv.rt:
        uid_map.setdefault(f.sop_uid, generate_uid())
        if f.series_uid and f.series_uid not in uid_map:
            uid_map[f.series_uid] = generate_uid()

    for f in inv.rt:
        ds = pydicom.dcmread(f.path)
        dose_check = None
        if f.modality == "RTDOSE" and opts.mask_rtdose:
            frame = str(ds.get("FrameOfReferenceUID", ""))
            matching = [s for s in series if frame and s.frame_uid == frame]
            if not matching or "PixelData" not in ds:
                res.rtdose_removed += 1
                res.flags.append(f"{f.rel.name}: RTDOSE not copied: "
                                 + ("it has no dose grid" if matching else "no de-faced CT with its Frame of Reference"))
                continue
            dose_check = mask_dose(ds, matching)
            res.rtdose_masked += 1
        refs_before = rtstruct_image_refs(ds) if f.modality == "RTSTRUCT" else set()
        remap_uids(ds, uid_map)
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        name = renamed(f.rel.name, f.sop_uid, str(ds.SOPInstanceUID))
        if f.modality == "RTSTRUCT":
            missing_before = len(refs_before - old_ct_sops)
            missing_after = len(rtstruct_image_refs(ds) - new_ct_sops)
            res.unresolved_refs += missing_after
            if missing_after > missing_before:
                raise PatientError(f"{f.rel.name}: {missing_after} contour image reference(s) do not point "
                                   "to the de-faced CT after remapping")
            if missing_after:
                res.flags.append(f"{f.rel.name} references {missing_after} image(s) that are not in this "
                                 "folder (already the case in the original)")
            plan = plan_roi_removal(roi_table(ds), opts)
            s = _series_for(ds, series)
            if plan.clip and s is None:                # cannot be clipped without the image: removed instead
                res.flags.append(f"{f.rel.name}: Body contour removed instead of clipped (the CT series it was drawn "
                                 "on was not found among the de-faced series)")
                plan.remove.update(plan.clip)
                plan.clip.clear()
            if plan.clip:
                (qc_dir / "removed_rois").mkdir(exist_ok=True)
                split_rois(copy.deepcopy(ds), set(plan.clip)).save_as(qc_dir / "removed_rois" / f"BEFORE_CLIP_{name}")
                for number, roi in plan.clip.items():
                    info = clip_roi(ds, number, s.grid, s.region, s.added_tissue)
                    if info["contours_dropped"]:
                        res.flags.append(f"{f.rel.name}: {info['contours_dropped']} contour(s) of {roi} could not be "
                                         "assigned to an image slice and were dropped")
                res.rois_clipped += sorted(plan.clip.values())
                log.info(f"    clipped ROI(s) in {f.rel.name}: {', '.join(sorted(plan.clip.values()))}")
            if plan.remove:
                removed = split_rois(ds, set(plan.remove))
                (qc_dir / "removed_rois").mkdir(exist_ok=True)
                removed.save_as(qc_dir / "removed_rois" / f"REMOVED_{name}")
                names = sorted(plan.remove.values())
                res.rois_removed += names
                log.info(f"    removed ROI(s) from {f.rel.name}: {', '.join(names)}")
            if s is None:
                res.flags.append(f"{f.rel.name}: ROI overlap not computed (the CT series it was drawn on was not "
                                 "found among the de-faced series)")
            else:
                rows, shown = _overlap_rows(res, name, ds, s, frozenset(plan.clip))
                if not rows and roi_table(ds):
                    res.flags.append(f"{f.rel.name}: ROI overlap not computed (no contour lies in an image plane "
                                     "of the CT series it refers to)")
                res.overlap += rows
                if shown and s.before is not None:
                    pictures[s.qc_suffix] = pictures.get(s.qc_suffix, 0) + 1
                    k = pictures[s.qc_suffix]
                    write_roi_picture(s.before, s.region, shown[:MAX_ROI_PICTURE], s.axes,
                                      qc_dir / f"QC_roi_overlap{s.qc_suffix}{'' if k == 1 else f'_{k}'}.png",
                                      more=max(0, len(shown) - MAX_ROI_PICTURE))
        rt_dir.mkdir(parents=True, exist_ok=True)
        target = rt_dir / name
        if target.exists():
            target = rt_dir / f"{f.modality}_{generate_uid()[-8:]}_{name}"
        ds.save_as(target)
        if dose_check is not None:
            verify_dose(target, dose_check[1], dose_check[0])

    near = [r for r in res.overlap if r["pct_in_defaced_region"] > 0]
    if near:
        log.info("    ROIs overlapping the de-faced region: "
                 + ", ".join(f"{r['roi']} {r['pct_in_defaced_region']:.0f}%" for r in near[:8])
                 + (", ..." if len(near) > 8 else ""))


# ---------------------------------------------------------------------------
# NIfTI / NRRD image
# ---------------------------------------------------------------------------

def deface_image_file(input_path: Path, rel: str, out_dir: Path, qc_dir: Path, opts: Options,
                      dry_run: bool = False) -> PatientResult:
    res = PatientResult(rel=rel, ct_series=1, ct_files=1, device=opts.device, machine=machine_info(opts.device))
    timer = Timer()
    if not opts.deface_targets:
        log.info("    NIfTI/NRRD input has no contours: target volumes cannot be kept, the whole region is replaced")
    if opts.keep_oar and not opts.keep_oar_ts:
        log.info("    NIfTI/NRRD input has no contours: --keep-oar restores nothing (--keep-oar-ts uses TotalSegmentator)")
    if opts.pii_check:
        res.pii = header_findings(input_path)
    if dry_run:
        res.status = "dry run"
        return res
    suffix = classify_suffix(input_path)
    _check_3d(input_path, suffix)
    partial = _prepare(out_dir, qc_dir)
    partial.mkdir(parents=True)
    stem = strip_known_suffix(input_path)
    try:
        with tempfile.TemporaryDirectory(prefix="deface_") as tmp:
            tmp = Path(tmp)
            nii_path = input_path
            if suffix == ".nrrd":
                nii_path = tmp / f"{stem}.nii.gz"
                sitk.WriteImage(sitk.ReadImage(str(input_path)), str(nii_path))
            with timer.stage("read"):
                earlier = _prior_of_image(input_path, suffix)
            with timer.stage("segment"):
                mask_path = run_totalsegmentator(nii_path, tmp / "face_seg", opts.device, qc_dir / "tool.log")
                face_ml = mask_volume_ml(mask_path)
                res.flags += check_empty_mask(face_ml, opts.allow_empty_mask, nii_path, tmp / "head_seg", opts.device,
                                              qc_dir / "tool.log", "the image", prior=earlier)
            if earlier:
                res.flags.append(f"the image appears to be already de-faced ({earlier}); de-faced as usual, check the "
                                 "QC picture")
            seed = opts.seed if opts.seed is not None else fill_seed(generate_uid())
            run_ctx = _ImageRun(nii_path, tmp, qc_dir / "tool.log", timer)
            if suffix == ".nrrd":
                out_path = partial / f"{stem}_deFaced.nrrd"
                info, flags, checks = _deface_nrrd(input_path, mask_path, out_path, opts, seed, qc_dir, run_ctx)
            else:
                out_path = partial / f"{stem}_deFaced.nii.gz"
                info, flags, checks = _deface_nifti(input_path, mask_path, out_path, opts, seed, qc_dir, run_ctx)
            res.flags += flags
            res.checks.append(checks)
            if opts.keep_oar:
                res.add_oars(run_ctx.oars)
            shutil.copy2(mask_path, qc_dir / "face_mask.nii.gz")
        res.face_mask_ml, res.fill = [face_ml], [info]
        res.timing = timer.seconds
        write_manifest(qc_dir, res, opts)
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    res.status = "done"
    return res


@dataclass
class _ImageRun:
    ct_nii: Path                # the image as NIfTI, for TotalSegmentator
    tmp: Path
    tool_log: Path
    timer: Timer
    oars: KeptTargets = field(default_factory=KeptTargets)


def _oars_of_image(data: np.ndarray, region: np.ndarray, axes, mask_path: Path, opts: Options, ctx: _ImageRun,
                   load) -> tuple[np.ndarray, list[str]]:
    """The voxels of a NIfTI/NRRD image to replace (the region minus restored OARs), and CHECK lines."""
    if not opts.keep_oar:
        return region, []
    with ctx.timer.stage("keep_oar"):
        ctx.oars = keep_oars(None, None, ctx.ct_nii, ctx.tmp, ctx.tool_log, load(mask_path), region, data, axes, opts,
                             "the image", load)
    return region if ctx.oars.keep is None else region & ~ctx.oars.keep, ctx.oars.flags


def _prior_of_image(path: Path, suffix: str) -> str | None:
    """Why a NIfTI/NRRD image looks already de-faced, or None."""
    if suffix == ".nrrd":
        img = sitk.ReadImage(str(path))
        return prior_defacing.evidence(sitk.GetArrayFromImage(img).astype(np.float64), sitk_axes(img), path=path)
    img = nib.load(str(path))
    return prior_defacing.evidence(img.get_fdata().astype(np.float64).reshape(img.shape[:3]), nifti_axes(img), path=path)


def _check_3d(path: Path, suffix: str) -> None:
    """Only single-component 3D images can be de-faced (trailing dimensions of size 1 are fine)."""
    if suffix == ".nrrd":
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        shape, components = reader.GetSize(), reader.GetNumberOfComponents()
    else:
        shape, components = nib.load(str(path)).shape, 1
    if len(shape) < 3 or any(n != 1 for n in shape[3:]) or components != 1:
        raise PatientError(f"the image is not a 3D CT (size {tuple(shape)}, {components} component(s) per voxel)")


def _nifti_scaling(img: nib.Nifti1Image) -> tuple[float, float]:
    """scl_slope/scl_inter of a loaded image (nibabel keeps them on the data proxy)."""
    slope, inter = getattr(img.dataobj, "slope", 1.0), getattr(img.dataobj, "inter", 0.0)
    slope = 1.0 if slope is None or not np.isfinite(slope) or slope == 0 else float(slope)
    inter = 0.0 if inter is None or not np.isfinite(inter) else float(inter)
    return slope, inter


def _check_image(written: np.ndarray, original: np.ndarray, target: np.ndarray, region: np.ndarray) -> None:
    """Read-back check; values may differ by rounding when the input used non-integer scaling."""
    if not np.allclose(written[~region], original[~region], rtol=0, atol=0.5 + 1e-6):
        raise PatientError("verification: voxels OUTSIDE the face region were changed")
    if (np.abs(written[region] - target[region]) > 0.5 + 1e-6).any():
        raise PatientError("verification: de-faced voxels do not hold the fill value (data type range?)")


def _dilation(opts: Options, voxel_size_known: bool) -> tuple[float | None, list[str]]:
    """Dilation in mm, or None (in voxels) with a flag if the voxel size is unknown."""
    if voxel_size_known:
        return opts.dilate_mm, []
    flag = (f"the image: voxel size unknown: face mask dilated by {opts.dilate} voxels instead of "
            f"{opts.dilate_mm:g} mm")
    log.warning(f"    ! {flag}")
    return None, [flag]


def _finish(data: np.ndarray, out: np.ndarray, region: np.ndarray, axes, qc_dir: Path, info: dict,
            dilate_mm: float | None) -> None:
    info["dilate_mm"] = dilate_mm
    write_qc_picture(data, out, region, axes, qc_dir / "QC_defacing.png")


def _deface_nifti(image_path: Path, mask_path: Path, out_path: Path, opts: Options, seed: int,
                  qc_dir: Path, ctx: _ImageRun) -> tuple[dict, list[str], dict]:
    img = nib.load(str(image_path))
    orig_dtype = img.get_data_dtype()
    # the stored values keep their scaling (e.g. uint16 with scl_inter -1024, as written by
    # dcm2niix), so that air can be stored and the voxels outside the region stay identical
    slope, inter = _nifti_scaling(img)
    full_shape = img.shape
    data = img.get_fdata().astype(np.float64).reshape(full_shape[:3])
    zooms = np.array(img.header.get_zooms()[:3], float)
    no_orientation = int(img.header["qform_code"]) == 0 and int(img.header["sform_code"]) == 0
    known = bool(np.all(np.isfinite(zooms)) and np.all(zooms > 0) and not (no_orientation and np.all(zooms == 1)))
    dilate_mm, flags = _dilation(opts, known)
    timer = ctx.timer
    with timer.stage("mask"):
        region = applied_mask_nifti(nib.load(str(mask_path)), img, opts.dilate, opts.extend_anterior, hu=data,
                                    dilate_by_mm=dilate_mm)
    axes = nifti_axes(img)
    load = lambda p: applied_mask_nifti(nib.load(str(p)), img, 0, False)
    full_region = region
    region, oar_flags = _oars_of_image(data, region, axes, mask_path, opts, ctx, load)
    flags = flags + oar_flags
    with timer.stage("fill"):
        target, info = apply_fill(data, full_region, axes, opts.fill, seed)
    with timer.stage("write"):
        out = data.copy()
        out[region] = target[region]
        stored = (out - inter) / slope
        stored = safe_cast(stored, orig_dtype) if np.issubdtype(orig_dtype, np.integer) else stored.astype(orig_dtype)
        out_img = nib.Nifti1Image(stored.reshape(full_shape), img.affine, img.header)
        out_img.set_data_dtype(orig_dtype)
        out_img.header.set_slope_inter(slope, inter)
        nib.save(out_img, str(out_path))
    with timer.stage("verify"):
        written = nib.load(str(out_path)).get_fdata().reshape(full_shape[:3])
        _check_image(written, data, target, region)
    with timer.stage("qc"):
        _finish(data, written, region, axes, qc_dir, info, dilate_mm)
        flags = flags + automatic_checks(mask_path, written, axes.si_axis, "the image")
    checks, found = run_optional_checks(
        opts, data, written, full_region, region, axes, ctx.ct_nii, ctx.tmp, ctx.tool_log, "the image", load, timer)
    return info, flags + found, checks


def _deface_nrrd(image_path: Path, mask_path: Path, out_path: Path, opts: Options, seed: int,
                 qc_dir: Path, ctx: _ImageRun) -> tuple[dict, list[str], dict]:
    img = sitk.ReadImage(str(image_path))
    orig_dtype = sitk.GetArrayViewFromImage(img).dtype
    data = sitk.GetArrayFromImage(img).astype(np.float64)
    # without a 'space' field and with unit spacing, the file most likely has no voxel size
    known = img.HasMetaDataKey("NRRD_space") or img.GetSpacing() != (1.0, 1.0, 1.0)
    dilate_mm, flags = _dilation(opts, known)
    timer = ctx.timer
    with timer.stage("mask"):
        region = applied_mask_sitk(sitk.ReadImage(str(mask_path)), img, opts.dilate, opts.extend_anterior, hu=data,
                                   dilate_by_mm=dilate_mm)
    axes = sitk_axes(img)
    load = lambda p: applied_mask_sitk(sitk.ReadImage(str(p)), img, 0, False)
    full_region = region
    region, oar_flags = _oars_of_image(data, region, axes, mask_path, opts, ctx, load)
    flags = flags + oar_flags
    with timer.stage("fill"):
        target, info = apply_fill(data, full_region, axes, opts.fill, seed)
    with timer.stage("write"):
        out = data.copy()
        out[region] = target[region]
        out_img = sitk.GetImageFromArray(safe_cast(out, orig_dtype))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))
    with timer.stage("verify"):
        written = sitk.GetArrayFromImage(sitk.ReadImage(str(out_path))).astype(np.float64)
        _check_image(written, data, target, region)
    with timer.stage("qc"):
        _finish(data, written, region, axes, qc_dir, info, dilate_mm)
        flags = flags + automatic_checks(mask_path, written, axes.si_axis, "the image")
    checks, found = run_optional_checks(
        opts, data, written, full_region, region, axes, ctx.ct_nii, ctx.tmp, ctx.tool_log, "the image", load, timer)
    return info, flags + found, checks
