"""Inventory of a DICOM patient folder: what is de-faced, copied or left out."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.filereader import read_partial
from pydicom.multival import MultiValue

from .common import Options, is_hidden
from .pii import IdentifierSummary
from .rt import roi_table

PLAN_MODALITIES = {"RTPLAN", "RTRECORD"}       # treatment records reference plans
# ImageType values of CT localizers/scouts (projection images that show the face profile)
LOCALIZER_TYPES = {"LOCALIZER", "TOPOGRAM", "SCOUT", "SURVEY", "SCANOGRAM"}
# ScanOptions words of localizers (Philips SURVIEW, GE SCOUT MODE, ...)
LOCALIZER_SCAN_OPTIONS = ("TOPOGRAM", "SCOUT", "SURVIEW", "SCANOGRAM", "LOCALIZER")
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
MULTIFRAME_CT = {"1.2.840.10008.5.1.4.1.1.2.1": "Enhanced CT",
                 "1.2.840.10008.5.1.4.1.1.2.2": "Legacy Converted Enhanced CT"}
MIN_SLICES = 3                                 # fewer slices: not a volume TotalSegmentator can segment
ORIENTATION_TOLERANCE = 0.01                   # direction cosines differing by more: another orientation


@dataclass
class DicomFile:
    path: Path          # absolute path in the input
    rel: Path           # path relative to the patient folder
    modality: str
    sop_uid: str
    series_uid: str
    series_desc: str
    frame_uid: str
    rois: dict = field(default_factory=dict)   # RTSTRUCT only: ROINumber -> (name, type)
    orientation: tuple = ()                    # CT only: ImageOrientationPatient
    pixel_spacing: bool = True                 # CT only: PixelSpacing present (voxel size known)


@dataclass
class Inventory:
    ct: list[DicomFile]                        # CT images to de-face
    rt: list[DicomFile]                        # RT objects to copy (UIDs remapped)
    not_copied: list[tuple[Path, str]]         # (relative path, reason)
    ids: IdentifierSummary | None
    rtdose_removed: int = 0
    rtplan_removed: int = 0
    headerless: int = 0                        # DICOM data without the DICOM file header: not read

    @property
    def ct_series(self) -> dict[str, list[DicomFile]]:
        groups: dict[str, list[DicomFile]] = defaultdict(list)
        for f in self.ct:
            groups[f.series_uid].append(f)
        # largest series first (typically the planning CT)
        return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def inventory(patient_dir: Path, opts: Options) -> Inventory:
    """Read the header of every file below patient_dir (recursively) and sort them."""
    inv = Inventory(ct=[], rt=[], not_copied=[], ids=IdentifierSummary() if opts.pii_check else None)
    paths = [patient_dir] if patient_dir.is_file() else sorted(patient_dir.rglob("*"))
    for path in paths:
        rel = Path(path.name) if patient_dir.is_file() else path.relative_to(patient_dir)
        if is_hidden(rel) or not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError:
            if _headerless_dicom(path):
                inv.headerless += 1
                inv.not_copied.append((rel, "DICOM data without the DICOM file header (128-byte preamble and "
                                            "'DICM'), not read: convert it first, e.g. with 'gdcmconv' or 'dcmconv'"))
            else:
                inv.not_copied.append((rel, "not a DICOM file"))
            continue
        except Exception as e:                       # truncated or damaged file
            inv.not_copied.append((rel, f"unreadable DICOM file ({type(e).__name__})"))
            continue
        if inv.ids is not None:
            inv.ids.collect(ds)
        modality = str(ds.get("Modality", "")).upper()
        image_type = ds.get("ImageType") or []
        image_type = {str(v).upper() for v in ([image_type] if isinstance(image_type, str) else image_type)}
        f = DicomFile(
            path=path, rel=rel, modality=modality,
            sop_uid=str(ds.get("SOPInstanceUID", "")),
            series_uid=str(ds.get("SeriesInstanceUID", "")),
            series_desc=str(ds.get("SeriesDescription", "")),
            frame_uid=str(ds.get("FrameOfReferenceUID", "")),
            rois=roi_table(ds) if modality == "RTSTRUCT" else {},
            orientation=_orientation(ds) if modality == "CT" else (),
            pixel_spacing="PixelSpacing" in ds,
        )
        if modality == "CT":
            sop_class = str(ds.get("SOPClassUID", "") or ds.file_meta.get("MediaStorageSOPClassUID", ""))
            scan_options = _as_text(ds.get("ScanOptions")).upper()
            if sop_class in MULTIFRAME_CT:
                inv.not_copied.append((rel, f"{MULTIFRAME_CT[sop_class]} (multi-frame) image: not supported, "
                                            "cannot be de-faced (convert it to single-frame CT images first)"))
            elif sop_class and sop_class != CT_IMAGE_STORAGE:
                inv.not_copied.append((rel, f"CT object that is not a CT image (SOP class {sop_class}), e.g. a "
                                            "screenshot or dose report with burned-in text; cannot be de-faced"))
            elif LOCALIZER_TYPES & image_type or any(w in scan_options for w in LOCALIZER_SCAN_OPTIONS):
                inv.not_copied.append((rel, "CT localizer/scout (shows the face profile, cannot be de-faced)"))
            elif str(ds.get("BurnedInAnnotation", "")).strip().upper() == "YES":
                inv.not_copied.append((rel, "CT image marked BurnedInAnnotation=YES (text such as the name drawn "
                                            "into the pixels, which de-facing does not remove)"))
            else:
                inv.ct.append(f)
        elif modality == "RTSTRUCT":
            inv.rt.append(f)
        elif modality == "RTDOSE":
            if opts.keep_rtdose:
                inv.rt.append(f)
            else:
                inv.rtdose_removed += 1
                inv.not_copied.append((rel, "RTDOSE removed (the dose is zero outside the body, so "
                                            "it outlines the face; --keep-rtdose keeps it)"))
        elif modality in PLAN_MODALITIES:
            if opts.keep_rtplan:
                inv.rt.append(f)
            else:
                inv.rtplan_removed += 1
                inv.not_copied.append((rel, f"{modality} removed (--keep-rtplan keeps it)"))
        else:
            inv.not_copied.append((rel, f"modality '{modality or '?'}' is not de-faced, so not copied"))
    _drop_non_volumes(inv)
    return inv


def _headerless_dicom(path: Path) -> bool:
    """A file without the DICOM file header (e.g. from old archives) that still holds DICOM data.
    Only the first elements are parsed (up to Modality), so large non-DICOM files are not read."""
    try:
        with open(path, "rb") as fh:
            ds = read_partial(fh, stop_when=lambda tag, vr, length: tag > 0x00080060, force=True)
        return str(ds.get("SOPClassUID", "")).startswith("1.2.840.10008.") and bool(ds.get("Modality"))
    except Exception:
        return False


def _as_text(value) -> str:
    if value is None:
        return ""
    return "\\".join(str(v) for v in value) if isinstance(value, (list, tuple, MultiValue)) else str(value)


def _orientation(ds) -> tuple:
    try:
        return tuple(float(v) for v in ds.ImageOrientationPatient)
    except (AttributeError, TypeError, ValueError):
        return ()


def _mixed_orientations(orientations: list[tuple]) -> bool:
    """True if the images of a series are not parallel (ignoring rounding in the headers)."""
    first = orientations[0]
    for o in orientations[1:]:
        if len(o) != len(first) or (o and max(abs(a - b) for a, b in zip(o, first)) > ORIENTATION_TOLERANCE):
            return True
    return False


def _drop_non_volumes(inv: Inventory) -> None:
    """CT series that are no volume (e.g. a localizer without a LOCALIZER ImageType: one or two
    projection images, or AP and lateral views in one series) are not copied, so that they can
    never reach the output unchanged (e.g. with --allow-empty-mask)."""
    for files in inv.ct_series.values():
        if len(files) < MIN_SLICES:
            reason = f"CT series of {len(files)} image(s): no volume (localizer?), cannot be de-faced"
        elif _mixed_orientations([f.orientation for f in files]):
            reason = "CT series with images of different orientations (localizer?), cannot be de-faced"
        else:
            continue
        for f in files:
            inv.ct.remove(f)
            inv.not_copied.append((f.rel, reason))
