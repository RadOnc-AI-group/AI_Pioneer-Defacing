"""Check of the metadata for patient-identifying data (PII). Nothing is changed."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.multival import MultiValue

from .common import classify_suffix, log, strip_known_suffix

# Person-name (PN) elements are collected everywhere in the file, including inside
# sequences, in addition to these tags.
DIRECT_IDENTIFIERS = [
    "PatientName", "PatientID", "OtherPatientIDs", "OtherPatientNames", "IssuerOfPatientID",
    "PatientBirthDate", "PatientBirthTime", "PatientBirthName", "PatientMotherBirthName",
    "PatientAddress", "PatientTelephoneNumbers", "MedicalRecordLocator", "AccessionNumber",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName", "PhysiciansOfRecord",
    "NameOfPhysiciansReadingStudy", "ReviewerName",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
]
INDIRECT_IDENTIFIERS = [
    "PatientSex", "PatientAge", "PatientSize", "PatientWeight", "EthnicGroup",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "InstanceCreationDate",
    "StructureSetDate", "RTPlanDate", "StudyTime", "StudyID", "StudyDescription",
    "SeriesDescription", "StructureSetLabel", "RTPlanLabel", "RTPlanName",
    "PatientComments", "ImageComments", "StationName", "DeviceSerialNumber",
]
# A patient's name/ID equal to the patient folder name (e.g. TUM_401) counts as a pseudonym.
PSEUDONYM_CHECKED = {"PatientName", "PatientID"}
# Burned-in text (e.g. the patient's name drawn into the pixels): very bright pixels outside
# the body and the devices, at the same position in most slices
TEXT_MIN_HU = 1500.0
TEXT_MIN_PIXELS = 10
# Metadata keys in NRRD headers or JSON sidecars that suggest patient-identifying data
PII_KEY_RE = re.compile(r"patient|birth|institution|physician|operator|accession|0010[|,._ ]?[0-9a-f]{4}",
                        re.I)


def _norm(value: str) -> str:
    return str(value).replace("^", " ").strip().lower()


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, MultiValue)):
        return "\\".join(str(v) for v in value).strip()
    return str(value).strip()


@dataclass
class IdentifierSummary:
    values: dict = field(default_factory=lambda: defaultdict(Counter))  # keyword -> Counter(value)
    kinds: dict = field(default_factory=dict)                            # keyword -> direct|indirect
    private_tags: set = field(default_factory=set)

    def add(self, keyword: str, kind: str, text: str) -> None:
        if text:
            self.values[keyword][text] += 1
            if self.kinds.get(keyword) != "direct":
                self.kinds[keyword] = kind

    def collect(self, ds: pydicom.Dataset) -> None:
        for kind, keywords in (("direct", DIRECT_IDENTIFIERS), ("indirect", INDIRECT_IDENTIFIERS)):
            for kw in keywords:
                if kw in ds and ds[kw].VR != "PN":          # PN handled below, everywhere
                    self.add(kw, kind, _as_text(ds[kw].value))
        for elem in ds.iterall():
            if elem.VR == "PN" and elem.value:
                self.add(elem.keyword or str(elem.tag), "direct", _as_text(elem.value))
            if elem.tag.is_private:
                self.private_tags.add(int(elem.tag))

    def matches_folder(self, keyword: str, folder: str) -> bool:
        if keyword not in PSEUDONYM_CHECKED or not self.values.get(keyword):
            return False
        return all(_norm(v) == _norm(folder) for v in self.values[keyword])

    def flagged(self, folder: str) -> list[str]:
        """Direct identifiers that are populated (a name/ID equal to the folder name excluded)."""
        return sorted(kw for kw, kind in self.kinds.items()
                      if kind == "direct" and self.values[kw] and not self.matches_folder(kw, folder))

    def indirect(self) -> list[str]:
        return sorted(kw for kw, kind in self.kinds.items() if kind == "indirect" and self.values[kw])


def warn_dicom(ids: IdentifierSummary, folder: str) -> list[str]:
    """Log a warning naming the populated identifying tags (never their values)."""
    flagged, indirect = ids.flagged(folder), ids.indirect()
    if flagged:
        log.warning(f"    ! PII: identifying tags populated: {', '.join(flagged)}")
    if indirect:
        log.info(f"    PII: dates/descriptions that identify indirectly: {', '.join(indirect[:8])}"
                 f"{', ...' if len(indirect) > 8 else ''}")
    if ids.private_tags:
        log.info(f"    PII: {len(ids.private_tags)} private (vendor) tags present, not inspected")
    if not flagged:
        log.info("    PII: no direct identifiers populated (name/ID absent or equal to the folder name)")
    return flagged


def header_findings(input_path: Path) -> list[str]:
    """Header fields of a NIfTI/NRRD file, and keys of a JSON sidecar next to it, that may
    hold patient-identifying data."""
    import nibabel as nib
    import SimpleITK as sitk

    found = set()
    if classify_suffix(input_path) == ".nrrd":
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(input_path))
        reader.ReadImageInformation()
        found.update(f"NRRD key '{k}'" for k in reader.GetMetaDataKeys()
                     if PII_KEY_RE.search(k) and reader.GetMetaData(k).strip())
    else:
        header = nib.load(str(input_path)).header
        for name in ("descrip", "aux_file", "db_name"):
            if name in header:
                text = np.asarray(header[name]).tobytes().decode("latin-1").strip("\x00 ")
                if text:
                    found.add(f"NIfTI header '{name}' (free text)")
        # extensions can hold a whole DICOM header (code 2), JSON or other metadata
        for ext in getattr(header, "extensions", []):
            found.add(f"NIfTI header extension (code {ext.get_code()})")
    sidecar = input_path.with_name(strip_known_suffix(input_path) + ".json")
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            found.update(f"{sidecar.name} key '{k}'" for k, v in meta.items()
                         if PII_KEY_RE.search(k) and str(v).strip())
        except (OSError, ValueError):
            pass
    findings = sorted(found)
    if findings:
        log.warning(f"    ! PII: {input_path.name} may contain identifying metadata: {'; '.join(findings)}"
                    " (headers are copied to the output)")
    return findings


def burned_in_text_pixels(hu: np.ndarray, si_axis: int) -> int:
    """
    Number of in-plane positions that are very bright (>= TEXT_MIN_HU) outside every large
    object (body, couch, devices) in at least half of the slices: text or graphics drawn
    into the images (names, dates, rulers) look like this, anatomy and metal inside the
    body do not. Metal outside the body along the head-feet axis (e.g. the posts of a
    stereotactic frame) can also be counted, so the result is a hint for a visual check.
    """
    from scipy.ndimage import binary_dilation, binary_fill_holes, label
    h = np.moveaxis(hu, si_axis, 0)
    count = np.zeros(h.shape[1:], np.int32)
    min_object = max(50, h.shape[1] * h.shape[2] // 500)     # letters are smaller than this
    for s in h:
        bright = s >= TEXT_MIN_HU
        if not bright.any():
            continue
        lab, _ = label(s > -500.0)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        large = binary_fill_holes(np.isin(lab, np.flatnonzero(sizes >= min_object)))
        count += bright & ~binary_dilation(large, iterations=3)
    return int((count >= max(3, 0.5 * len(h))).sum())
