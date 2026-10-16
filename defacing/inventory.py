"""Inventory of a DICOM patient folder: what is de-faced, copied or left out."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from .common import Options, is_hidden
from .pii import IdentifierSummary
from .rt import roi_table

PLAN_MODALITIES = {"RTPLAN", "RTRECORD"}       # treatment records reference plans
# ImageType values of CT localizers/scouts (projection images that show the face profile)
LOCALIZER_TYPES = {"LOCALIZER", "TOPOGRAM", "SCOUT", "SURVEY"}


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


@dataclass
class Inventory:
    ct: list[DicomFile]                        # CT images to de-face
    rt: list[DicomFile]                        # RT objects to copy (UIDs remapped)
    not_copied: list[tuple[Path, str]]         # (relative path, reason)
    ids: IdentifierSummary | None
    rtdose_removed: int = 0
    rtplan_removed: int = 0

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
        )
        if modality == "CT":
            if LOCALIZER_TYPES & image_type:
                inv.not_copied.append((rel, "CT localizer/scout (shows the face profile, cannot be de-faced)"))
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
    return inv
