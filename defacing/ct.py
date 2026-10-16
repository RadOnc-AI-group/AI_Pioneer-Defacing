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

from .common import Options, PatientError, safe_cast
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


def _load_sorted(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """HU volume (slices sorted by position along the slice normal) and the sort order."""
    dss = [pydicom.dcmread(p) for p in paths]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    order = np.argsort([float(np.dot(normal, np.array(d.ImagePositionPatient, float))) for d in dss],
                       kind="stable")
    vol = np.stack([dss[k].pixel_array.astype(np.float64) * float(getattr(dss[k], "RescaleSlope", 1))
                    + float(getattr(dss[k], "RescaleIntercept", 0)) for k in order])
    return vol, order


def verify_written(in_paths: list[Path], out_paths: list[Path], image_hu: np.ndarray,
                   target_hu: np.ndarray, region: np.ndarray, fill: str, max_slope: float) -> None:
    """Independent read-back check of the written series (see module docstring)."""
    vin, order_in = _load_sorted(in_paths)
    vout, order_out = _load_sorted(out_paths)
    if vin.shape != image_hu.shape or not np.allclose(vin, image_hu, atol=0.5):
        raise PatientError("verification: the slice order/orientation of the series reader differs from "
                           "the slice positions in the files (gantry tilt? duplicate slices?); "
                           "nothing was written for this patient")
    if not np.array_equal(order_in, order_out):
        raise PatientError("verification: slice positions changed in the written files")
    if not np.array_equal(vout[~region], vin[~region]):
        n = int((vout[~region] != vin[~region]).sum())
        raise PatientError(f"verification: {n} voxels OUTSIDE the face region were changed")
    # stored values are rounded, so the HU value can be off by half a rescale step
    bad = np.abs(vout[region] - target_hu[region]) > 0.5 * abs(max_slope) + 1e-6
    if bad.any():
        what = "-1000 HU" if fill == "constant" else "the fill values"
        raise PatientError(f"verification: {int(bad.sum())} de-faced voxels do not hold {what}; the pixel "
                           "encoding (data type / rescale slope and intercept) cannot represent it")


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
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        max_slope = max(max_slope, abs(slope))
        pixels = ds.pixel_array.copy()
        m = region[i]
        if m.any():
            pixels[m] = safe_cast((target_hu[i][m] - intercept) / slope, pixels.dtype)
        ds.PixelData = pixels.tobytes()
        _mark_defaced(ds, opts, fill_info)
        target = out_dir / renamed(src.rel.name, src.sop_uid, ds.SOPInstanceUID)
        if target.exists():                            # same file name in two input folders
            target = target.with_name(f"{i:05d}_{target.name}")
        ds.save_as(target)
        sop_map[src.sop_uid] = ds.SOPInstanceUID
        out_paths.append(target)

    verify_written([Path(f) for f in file_names], out_paths, image_hu, target_hu, region, opts.fill,
                   max_slope)

    # QC (confidential): raw face mask and the de-faced volume as NIfTI
    qc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mask_path, qc_dir / f"face_mask{qc_suffix}.nii.gz")
    defaced = sitk.GetImageFromArray(safe_cast(target_hu, np.int16))
    defaced.CopyInformation(image)
    sitk.WriteImage(defaced, str(qc_dir / f"CT_defaced{qc_suffix}.nii.gz"))

    return SeriesResult(old_series_uid=files[0].series_uid, new_series_uid=new_series_uid,
                        frame_uid=frame_uid, sop_map=sop_map, grid=Grid.of(image), region=region,
                        face_ml=face_ml, fill_info=fill_info, flags=flags)
