"""
De-facing of one CT DICOM series, with verification of the written files.

The series is read once with SimpleITK (GDCM). That same volume is written to NIfTI for
TotalSegmentator, so the face mask is computed on exactly the grid that is written back
(no second converter whose slice order or orientation could differ). After writing, the
output is read back independently (pydicom, slices sorted by position) and compared with
the input: outside the de-faced region every value must be unchanged, inside it every
value must be the intended fill value. Otherwise the patient fails.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from .common import Options, PatientError, log, safe_cast
from .fill import apply_fill, fill_seed
from .masks import applied_mask_sitk, mask_volume_ml, sitk_axes
from .rt import Grid, renamed
from .segment import run_totalsegmentator


@dataclass
class SeriesResult:
    old_series_uid: str
    new_series_uid: str
    frame_uid: str
    sop_map: dict[str, str]                   # old SOP Instance UID -> new
    grid: Grid
    region: np.ndarray                        # de-faced voxels (z, y, x)
    face_ml: float
    fill_info: dict
    flags: list[str] = field(default_factory=list)


def _deid_method(opts: Options, fill_info: dict) -> list[str]:
    seed = f" seed={fill_info['seed']}" if "seed" in fill_info else ""
    return ["De-faced: TotalSegmentator face mask",
            f"dilate={opts.dilate} extend_anterior={'yes' if opts.extend_anterior else 'no'}",
            f"fill={opts.fill}{seed}"]


def _mark_defaced(ds: Dataset, opts: Options, fill_info: dict) -> None:
    """Record the de-facing in the image header (DICOM PS3.15 de-identification method)."""
    if "ImageComments" in ds:
        ds.ImageComments = str(ds.ImageComments) + " | de-faced (TotalSegmentator-face)"
    else:
        ds.ImageComments = "De-faced (TotalSegmentator-face)"
    prev = ds.get("DeidentificationMethod")
    previous = [] if not prev else ([str(prev)] if isinstance(prev, str) else [str(v) for v in prev])
    ds.DeidentificationMethod = previous + _deid_method(opts, fill_info)
    code = Dataset()
    code.CodeValue, code.CodingSchemeDesignator = "113102", "DCM"
    code.CodeMeaning = "Clean Recognizable Visual Features Option"
    codes = list(ds.DeidentificationMethodCodeSequence) if "DeidentificationMethodCodeSequence" in ds else []
    ds.DeidentificationMethodCodeSequence = Sequence(codes + [code])


def _rescale(ds: Dataset) -> tuple[float, float]:
    return float(getattr(ds, "RescaleSlope", 1) or 1), float(getattr(ds, "RescaleIntercept", 0) or 0)


def stored_range(ds: Dataset, dtype: np.dtype) -> tuple[int, int]:
    """Smallest and largest pixel value the file can store (BitsStored, sign, data type)."""
    info = np.iinfo(dtype)
    bits = int(getattr(ds, "BitsStored", info.bits) or info.bits)
    if int(getattr(ds, "PixelRepresentation", 0)) == 1:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    return max(lo, int(info.min)), min(hi, int(info.max))


def _encoding(ds: Dataset) -> str:
    slope, intercept = _rescale(ds)
    return (f"BitsAllocated {getattr(ds, 'BitsAllocated', '?')}, BitsStored {getattr(ds, 'BitsStored', '?')}, "
            f"{'signed' if int(getattr(ds, 'PixelRepresentation', 0)) == 1 else 'unsigned'}, "
            f"slope {slope:g}, intercept {intercept:g}")


def _load_sorted(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[Dataset]]:
    """HU volume (slices sorted by position along the slice normal), the sort order and the datasets."""
    dss = [pydicom.dcmread(p) for p in paths]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    order = np.argsort([float(np.dot(normal, np.array(d.ImagePositionPatient, float))) for d in dss],
                       kind="stable")
    vol = np.stack([dss[k].pixel_array.astype(np.float64) * _rescale(dss[k])[0] + _rescale(dss[k])[1]
                    for k in order])
    return vol, order, dss


def verify_written(in_paths: list[Path], out_paths: list[Path], image_hu: np.ndarray,
                   target_hu: np.ndarray, region: np.ndarray, fill: str, max_slope: float) -> None:
    """Independent read-back check of the written series (see module docstring)."""
    vin, order_in, dss_in = _load_sorted(in_paths)
    vout, order_out, _ = _load_sorted(out_paths)
    if vin.shape != image_hu.shape or not np.allclose(vin, image_hu, atol=0.5):
        diff = float(np.abs(vin - image_hu).max()) if vin.shape == image_hu.shape else float("nan")
        raise PatientError("verification: the slice order/orientation of the series reader differs from "
                           "the slice positions in the files (gantry tilt? duplicate slices?); "
                           f"max difference {diff:g} HU; nothing was written for this patient")
    if not np.array_equal(order_in, order_out):
        raise PatientError("verification: slice positions changed in the written files")
    if not np.array_equal(vout[~region], vin[~region]):
        n = int((vout[~region] != vin[~region]).sum())
        raise PatientError(f"verification: {n} voxels OUTSIDE the face region were changed")
    # stored values are rounded, so the HU value can be off by half a rescale step
    diff = np.abs(vout[region] - target_hu[region])
    bad = diff > 0.5 * abs(max_slope) + 1e-3
    if bad.any():
        what = "-1000 HU" if fill == "constant" else "the fill values"
        examples = ", ".join(f"{t:g} -> {w:g}" for t, w in
                             list(zip(target_hu[region][bad], vout[region][bad]))[:3])
        encodings = sorted({_encoding(d) for d in dss_in})
        raise PatientError(
            f"verification: {int(bad.sum())} de-faced voxels do not hold {what}; the pixel encoding probably "
            f"cannot store it (fill {fill}; intended -> written HU: {examples}; max difference "
            f"{float(diff.max()):g} HU; encoding: {' | '.join(encodings[:3])}{' | ...' if len(encodings) > 3 else ''})")


def deface_ct_series(files: list, label: str, out_dir: Path, qc_dir: Path, qc_suffix: str,
                     opts: Options, tmp: Path, tool_log: Path) -> SeriesResult:
    """De-face one CT series (list of inventory.DicomFile) into out_dir (flat)."""
    stage = tmp / "in"
    stage.mkdir(parents=True)
    staged = {}                                        # staged name -> inventory.DicomFile
    for i, f in enumerate(files):                      # copied to local disk; CT files only
        name = f.rel.name if f.rel.name not in staged else f"{i:05d}_{f.rel.name}"
        shutil.copy2(f.path, stage / name)
        staged[name] = f

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(stage))
    if len(series_ids) != 1:
        raise PatientError(f"{label}: expected one DICOM series, the reader found {len(series_ids)}")
    file_names = list(reader.GetGDCMSeriesFileNames(str(stage), series_ids[0]))
    if len(file_names) != len(files):
        raise PatientError(f"{label}: the series reader uses {len(file_names)} of {len(files)} files "
                           "(duplicate or inconsistent slices?)")
    reader.SetFileNames(file_names)
    image = reader.Execute()

    ct_nii = tmp / "ct.nii.gz"
    sitk.WriteImage(image, str(ct_nii))
    seg_dir = tmp / "face_seg"
    mask_path = run_totalsegmentator(ct_nii, seg_dir, opts.device, tool_log)
    face_ml = mask_volume_ml(mask_path)
    flags = []
    if face_ml == 0:
        msg = f"face mask is EMPTY for {label}: nothing was removed"
        if not opts.allow_empty_mask:
            raise PatientError(msg + " (use --allow-empty-mask if the CT has no face)")
        flags.append(msg)

    region = applied_mask_sitk(sitk.ReadImage(str(mask_path)), image, opts.dilate, opts.extend_anterior)
    image_hu = sitk.GetArrayFromImage(image).astype(np.float64)
    new_series_uid = generate_uid()
    seed = opts.seed if opts.seed is not None else fill_seed(new_series_uid)
    target_hu, fill_info = apply_fill(image_hu, region, sitk_axes(image), opts.fill, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    sop_map: dict[str, str] = {}
    out_paths: list[Path] = []
    frame_uid = ""
    max_slope = 1.0
    n_fitted = n_written = 0
    for i, fname in enumerate(file_names):
        src = staged[Path(fname).name]
        ds = pydicom.dcmread(fname)
        frame_uid = frame_uid or str(ds.get("FrameOfReferenceUID", ""))
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            # Decompresses PixelData in place and switches to an uncompressed transfer
            # syntax; writing decoded bytes under a compressed syntax would corrupt the file.
            ds.decompress()
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesInstanceUID = new_series_uid
        slope, intercept = _rescale(ds)
        max_slope = max(max_slope, abs(slope))
        pixels = ds.pixel_array.copy()
        # only voxels whose value changes are written; the others (e.g. padding outside the
        # field of view) keep their stored value bit for bit
        m = region[i] & (np.abs(target_hu[i] - image_hu[i]) > 1e-3)
        if m.any():
            n_written += int(m.sum())
            lo, hi = stored_range(ds, pixels.dtype)
            vals = target_hu[i][m]
            if opts.fill != "constant":
                # sampled values come from several slices; a slice may store a smaller range
                # (e.g. another rescale intercept). The few extreme values are fitted in (checked
                # below); -1000 HU (constant) is never adjusted, so an encoding that cannot store
                # it fails the verification.
                hu_lo, hu_hi = sorted((lo * slope + intercept, hi * slope + intercept))
                outside = (vals < hu_lo) | (vals > hu_hi)
                if outside.any():
                    n_fitted += int(outside.sum())
                    vals = np.clip(vals, hu_lo, hu_hi)
                    target_hu[i][m] = vals
            pixels[m] = safe_cast(np.clip(np.round((vals - intercept) / slope), lo, hi), pixels.dtype)
        ds.PixelData = pixels.tobytes()
        if n_fitted > 0.05 * max(n_written, 1) and n_fitted > 50:
            # not a few extreme values but the fill itself: a solid block of the nearest storable
            # value would show the outline of the face mask
            raise PatientError(f"{label}: the pixel encoding cannot store the fill values ({n_fitted} of "
                               f"{n_written} written voxels so far are outside the storable range; encoding: "
                               f"{_encoding(ds)}); nothing was written for this patient")
        _mark_defaced(ds, opts, fill_info)
        target = out_dir / renamed(src.rel.name, src.sop_uid, ds.SOPInstanceUID)
        if target.exists():                            # same file name in two input folders
            target = target.with_name(f"{i:05d}_{target.name}")
        ds.save_as(target)
        sop_map[src.sop_uid] = ds.SOPInstanceUID
        out_paths.append(target)

    verify_written([Path(f) for f in file_names], out_paths, image_hu, target_hu, region, opts.fill,
                   max_slope)
    if n_fitted:
        fill_info["values_fitted_to_encoding"] = n_fitted
        log.info(f"    {n_fitted} of {n_written} fill values were outside the range a slice can store and were "
                 "set to the nearest storable value")

    # QC (confidential): raw face mask and the de-faced volume as NIfTI
    qc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mask_path, qc_dir / f"face_mask{qc_suffix}.nii.gz")
    defaced = sitk.GetImageFromArray(safe_cast(target_hu, np.int16))
    defaced.CopyInformation(image)
    sitk.WriteImage(defaced, str(qc_dir / f"CT_defaced{qc_suffix}.nii.gz"))

    return SeriesResult(old_series_uid=files[0].series_uid, new_series_uid=new_series_uid,
                        frame_uid=frame_uid, sop_map=sop_map, grid=Grid.of(image), region=region,
                        face_ml=face_ml, fill_info=fill_info, flags=flags)
