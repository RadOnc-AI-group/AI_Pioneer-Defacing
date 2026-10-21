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

from .common import Options, PatientError, classify_suffix, log, safe_cast, strip_known_suffix
from .ct import SeriesResult, deface_ct_series
from .fill import apply_fill, fill_seed
from .inventory import Inventory, inventory
from .masks import applied_mask_nifti, applied_mask_sitk, mask_volume_ml, nifti_axes, sitk_axes
from .pii import IdentifierSummary, header_findings, warn_dicom
from .qc import MAX_ROIS as MAX_ROI_PICTURE
from .qc import ROI_NEAR_MM, automatic_checks, roi_label, write_qc_picture, write_roi_picture
from .rt import (RegionStats, plan_roi_removal, referenced_series, remap_uids, renamed, roi_slices,
                 roi_table, rtstruct_image_refs, split_rois)
from .segment import check_empty_mask, run_totalsegmentator

PARTIAL_SUFFIX = ".partial"
SCRIPT_VERSION = 5          # stored in the manifest; bump when the output content changes
TARGET_TYPES = ("CTV", "GTV", "PTV")


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
        "rtplan_removed": res.rtplan_removed,
    }, indent=2), encoding="utf-8")


OVERLAP_COLS = ["patient", "rtstruct", "roi", "type", "voxels", "pct_in_defaced_region", "ml_in_defaced_region",
                "pct_in_face_mask", "min_distance_mm"]


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


def _overlap_rows(res: PatientResult, rs_name: str, ds, s: SeriesResult) -> tuple[list[dict], list]:
    """One report row per ROI, and (legend text, {slice: 2D mask}) of the ROIs in or near the
    region (most affected first) for the ROI picture."""
    rows, shown = [], []
    stats = RegionStats(s.region, tuple(s.grid.spacing[::-1]))
    voxel_ml = float(np.prod(s.grid.spacing)) / 1000.0
    names = roi_table(ds)
    for number, slices in roi_slices(ds, s.grid):
        name, kind = names.get(number, (str(number), ""))
        n, inside, d = stats.measure(slices)
        if not n:
            continue
        in_face = sum(int((m & s.face[z]).sum()) for z, m in slices.items()) if s.face is not None else 0
        distance = f">={stats.cap_mm:.0f}" if d >= stats.cap_mm else round(d, 1)
        row = {"patient": res.rel, "rtstruct": rs_name, "roi": name, "type": kind, "voxels": n,
               "pct_in_defaced_region": round(100.0 * inside / n, 1),
               "ml_in_defaced_region": round(inside * voxel_ml, 2),
               "pct_in_face_mask": round(100.0 * in_face / n, 1), "min_distance_mm": distance}
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
            "roi_overlap_*.csv")


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
    res = PatientResult(rel=rel)
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
        for k, files in enumerate(groups):
            label = f"CT series {k + 1}/{len(groups)} ({len(files)} files"
            label += f", '{files[0].series_desc}')" if files[0].series_desc else ")"
            log.info(f"    de-facing {label} ...")
            sub = "dcm" if len(groups) == 1 else f"dcm/series_{k + 1}"
            suffix = "" if len(groups) == 1 else f"_series{k + 1}"
            with tempfile.TemporaryDirectory(prefix="deface_") as tmp:
                s = deface_ct_series(files, label, partial / sub, qc_dir, suffix, opts, Path(tmp), tool_log)
            series_results.append(s)
            res.face_mask_ml.append(s.face_ml)
            res.fill.append(s.fill_info)
            res.flags += s.flags

        _copy_rt(inv, res, partial / "rt", qc_dir, series_results, opts)
        if res.overlap:
            write_overlap_csv(qc_dir / "roi_overlap.csv", res.overlap)
        write_manifest(qc_dir, res, opts)         # before the rename: no output without one
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    res.status = "done"
    return res


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
            if plan.remove:
                removed = split_rois(ds, set(plan.remove))
                (qc_dir / "removed_rois").mkdir(exist_ok=True)
                removed.save_as(qc_dir / "removed_rois" / f"REMOVED_{name}")
                names = sorted(plan.remove.values())
                res.rois_removed += names
                log.info(f"    removed ROI(s) from {f.rel.name}: {', '.join(names)}")
            s = _series_for(ds, series)
            if s is None:
                res.flags.append(f"{f.rel.name}: ROI overlap not computed (the CT series it was drawn on was not "
                                 "found among the de-faced series)")
            else:
                rows, shown = _overlap_rows(res, name, ds, s)
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
    res = PatientResult(rel=rel, ct_series=1, ct_files=1)
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
            mask_path = run_totalsegmentator(nii_path, tmp / "face_seg", opts.device, qc_dir / "tool.log")
            face_ml = mask_volume_ml(mask_path)
            res.flags += check_empty_mask(face_ml, opts.allow_empty_mask, nii_path, tmp / "head_seg", opts.device,
                                          qc_dir / "tool.log", "the image")
            seed = opts.seed if opts.seed is not None else fill_seed(generate_uid())
            if suffix == ".nrrd":
                out_path = partial / f"{stem}_deFaced.nrrd"
                info, flags = _deface_nrrd(input_path, mask_path, out_path, opts, seed, qc_dir)
            else:
                out_path = partial / f"{stem}_deFaced.nii.gz"
                info, flags = _deface_nifti(input_path, mask_path, out_path, opts, seed, qc_dir)
            res.flags += flags
            shutil.copy2(mask_path, qc_dir / "face_mask.nii.gz")
        res.face_mask_ml, res.fill = [face_ml], [info]
        write_manifest(qc_dir, res, opts)
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    res.status = "done"
    return res


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
                  qc_dir: Path) -> tuple[dict, list[str]]:
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
    region = applied_mask_nifti(nib.load(str(mask_path)), img, opts.dilate, opts.extend_anterior, hu=data,
                                dilate_by_mm=dilate_mm)
    axes = nifti_axes(img)
    target, info = apply_fill(data, region, axes, opts.fill, seed)
    out = data.copy()
    out[region] = target[region]
    stored = (out - inter) / slope
    stored = safe_cast(stored, orig_dtype) if np.issubdtype(orig_dtype, np.integer) else stored.astype(orig_dtype)
    out_img = nib.Nifti1Image(stored.reshape(full_shape), img.affine, img.header)
    out_img.set_data_dtype(orig_dtype)
    out_img.header.set_slope_inter(slope, inter)
    nib.save(out_img, str(out_path))
    written = nib.load(str(out_path)).get_fdata().reshape(full_shape[:3])
    _check_image(written, data, target, region)
    _finish(data, written, region, axes, qc_dir, info, dilate_mm)
    return info, flags + automatic_checks(mask_path, written, axes.si_axis, "the image")


def _deface_nrrd(image_path: Path, mask_path: Path, out_path: Path, opts: Options, seed: int,
                 qc_dir: Path) -> tuple[dict, list[str]]:
    img = sitk.ReadImage(str(image_path))
    orig_dtype = sitk.GetArrayViewFromImage(img).dtype
    data = sitk.GetArrayFromImage(img).astype(np.float64)
    # without a 'space' field and with unit spacing, the file most likely has no voxel size
    known = img.HasMetaDataKey("NRRD_space") or img.GetSpacing() != (1.0, 1.0, 1.0)
    dilate_mm, flags = _dilation(opts, known)
    region = applied_mask_sitk(sitk.ReadImage(str(mask_path)), img, opts.dilate, opts.extend_anterior, hu=data,
                               dilate_by_mm=dilate_mm)
    axes = sitk_axes(img)
    target, info = apply_fill(data, region, axes, opts.fill, seed)
    out = data.copy()
    out[region] = target[region]
    out_img = sitk.GetImageFromArray(safe_cast(out, orig_dtype))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))
    written = sitk.GetArrayFromImage(sitk.ReadImage(str(out_path))).astype(np.float64)
    _check_image(written, data, target, region)
    _finish(data, written, region, axes, qc_dir, info, dilate_mm)
    return info, flags + automatic_checks(mask_path, written, axes.si_axis, "the image")
