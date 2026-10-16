"""
RT objects: removal of the Body/External and Skin contours, UID remapping, and the
overlap of the remaining ROIs with the de-faced region.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

# ROI names are normalised (lower case, umlauts as ae/oe/ue, ß as ss) before matching.
# A name that is one of these words, optionally followed by a number or a thickness
# ("BODY_1", "Skin 5mm", "Haut_3mm"), is removed. A name that only CONTAINS one of the
# words ("Körperteile", "Body-PTV", "Glaskörper", "Wirbelkörper") is kept and reported.
BODY_WORDS = ["body", "external", "outline", "koerper", "koerperkontur", "aussenkontur",
              "aussenkontour", "patient", "patientenkontur", "patientkontur"]
SKIN_WORDS = ["skin", "haut", "hautkontur"]
# Kept but reported: may follow the face surface (immobilisation mask, bolus, shell)
FACE_SHAPE_WORDS = ["mask", "maske", "bolus", "shell", "schale"]
_SUFFIX = r"[\s_\-.]*(\d+([.,]\d+)?\s*(mm|cm)?)?"
_BODY_RE = re.compile(rf"(?:{'|'.join(BODY_WORDS)}){_SUFFIX}")
_SKIN_RE = re.compile(rf"(?:{'|'.join(SKIN_WORDS)}){_SUFFIX}")
_UNCLEAR_RE = re.compile("|".join(["body", "external", "outline", "koerper", "aussenkont", "skin",
                                   "haut"] + FACE_SHAPE_WORDS))


def normalise(name: str) -> str:
    s = name.strip().lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def classify_roi(name: str, kind: str) -> str:
    """'body', 'skin', 'unclear' or '' for an ROI name and its RTROIInterpretedType."""
    n = normalise(name)
    if kind == "EXTERNAL" or _BODY_RE.fullmatch(n):
        return "body"
    if _SKIN_RE.fullmatch(n):
        return "skin"
    if kind == "BOLUS" or _UNCLEAR_RE.search(n):
        return "unclear"
    return ""


def roi_table(ds: Dataset) -> dict[int, tuple[str, str]]:
    """ROINumber -> (ROIName, RTROIInterpretedType) of an RTSTRUCT."""
    types = {int(o.ReferencedROINumber): str(o.get("RTROIInterpretedType", "")).upper()
             for o in ds.get("RTROIObservationsSequence", []) if "ReferencedROINumber" in o}
    return {int(r.ROINumber): (str(r.get("ROIName", "")).strip(), types.get(int(r.ROINumber), ""))
            for r in ds.get("StructureSetROISequence", []) if "ROINumber" in r}


@dataclass
class RoiPlan:
    remove: dict[int, str] = field(default_factory=dict)    # ROINumber -> ROI name
    unclear: list[str] = field(default_factory=list)        # kept, but may follow the skin/face
    has_body: bool = False


def plan_roi_removal(rois: dict[int, tuple[str, str]], opts) -> RoiPlan:
    plan = RoiPlan()
    for number, (name, kind) in rois.items():
        cls = classify_roi(name, kind)
        plan.has_body |= cls == "body"
        if cls == "body" and not opts.keep_body:
            plan.remove[number] = name
        elif cls == "skin" and not opts.keep_skin:
            plan.remove[number] = name
        elif any(p.fullmatch(name) for p in opts.remove_roi):
            plan.remove[number] = name
        elif cls == "unclear":
            plan.unclear.append(name)
    return plan


def split_rois(ds: Dataset, numbers: set[int]) -> Dataset:
    """Remove the ROIs `numbers` from ds in place; return a copy of ds that holds only them."""
    removed = copy.deepcopy(ds)
    for seq, key in (("StructureSetROISequence", "ROINumber"),
                     ("ROIContourSequence", "ReferencedROINumber"),
                     ("RTROIObservationsSequence", "ReferencedROINumber")):
        if seq not in ds:
            continue
        setattr(ds, seq, Sequence([it for it in ds[seq].value if int(it.get(key, -1)) not in numbers]))
        setattr(removed, seq, Sequence([it for it in removed[seq].value if int(it.get(key, -1)) in numbers]))
    removed.SOPInstanceUID = generate_uid()
    removed.file_meta.MediaStorageSOPInstanceUID = removed.SOPInstanceUID
    removed.SeriesInstanceUID = generate_uid()
    removed.StructureSetLabel = "REMOVED_ROIS"
    removed.SeriesDescription = "ROIs removed by de-facing (QC only, do not share)"
    return removed


# ---------------------------------------------------------------------------
# UIDs
# ---------------------------------------------------------------------------

def renamed(name: str, old_uid: str, new_uid: str) -> str:
    """File names that contain the SOP Instance UID (CT.<uid>.dcm, ...) get the new UID."""
    return name.replace(old_uid, new_uid) if old_uid and old_uid in name else name


def remap_uids(ds: Dataset, uid_map: dict[str, str]) -> int:
    """Replace every UID value (VR UI, anywhere in the dataset) that is a key of uid_map."""
    changed = 0
    for elem in ds.iterall():
        if elem.VR != "UI" or not elem.value:
            continue
        if isinstance(elem.value, (list, MultiValue)):
            old = [str(v) for v in elem.value]
            new = [uid_map.get(v, v) for v in old]
            if new != old:
                elem.value = new
                changed += sum(a != b for a, b in zip(old, new))
        else:
            v = str(elem.value)
            if v in uid_map:
                elem.value = uid_map[v]
                changed += 1
    return changed


def rtstruct_image_refs(ds: Dataset) -> set[str]:
    refs: set[str] = set()
    for roi in ds.get("ROIContourSequence", []):
        for contour in roi.get("ContourSequence", []):
            for img in contour.get("ContourImageSequence", []):
                refs.add(str(img.get("ReferencedSOPInstanceUID", "")))
    for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
        for study in frame.get("RTReferencedStudySequence", []):
            for series in study.get("RTReferencedSeriesSequence", []):
                for img in series.get("ContourImageSequence", []):
                    refs.add(str(img.get("ReferencedSOPInstanceUID", "")))
    refs.discard("")
    return refs


def referenced_series(ds: Dataset) -> set[str]:
    return {str(series.get("SeriesInstanceUID", ""))
            for frame in ds.get("ReferencedFrameOfReferenceSequence", [])
            for study in frame.get("RTReferencedStudySequence", [])
            for series in study.get("RTReferencedSeriesSequence", [])} - {""}


# ---------------------------------------------------------------------------
# Contours -> voxels, and overlap with the de-faced region
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    """Geometry of a CT volume whose numpy axes are (z, y, x) (SimpleITK convention)."""
    origin: np.ndarray
    spacing: np.ndarray        # x, y, z
    direction: np.ndarray      # 3x3, columns = index axes
    size: tuple[int, int, int]  # x, y, z

    @classmethod
    def of(cls, img) -> "Grid":
        return cls(np.array(img.GetOrigin()), np.array(img.GetSpacing()),
                   np.array(img.GetDirection()).reshape(3, 3), tuple(img.GetSize()))

    def to_index(self, points: np.ndarray) -> np.ndarray:
        """(n, 3) world points (mm, LPS) -> (n, 3) continuous indices (x, y, z)."""
        m = self.direction * self.spacing[None, :]
        return np.linalg.solve(m, (points - self.origin).T).T


def fill_polygon(cols: np.ndarray, rows: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixels whose centres lie inside a closed polygon (even-odd rule), by scanlines."""
    out = np.zeros(shape, bool)
    c0, r0 = cols, rows
    c1, r1 = np.roll(cols, -1), np.roll(rows, -1)
    for row in range(max(0, int(np.ceil(rows.min()))), min(shape[0] - 1, int(np.floor(rows.max()))) + 1):
        cross = ((r0 <= row) & (r1 > row)) | ((r1 <= row) & (r0 > row))
        if not cross.any():
            continue
        xs = np.sort(c0[cross] + (row - r0[cross]) * (c1[cross] - c0[cross]) / (r1[cross] - r0[cross]))
        for a, b in zip(xs[0::2], xs[1::2]):
            lo, hi = max(int(np.ceil(a)), 0), min(int(np.floor(b)), shape[1] - 1)
            if hi >= lo:
                out[row, lo:hi + 1] = True
    return out


def roi_slices(ds: Dataset, grid: Grid) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
    """(ROINumber, {slice index: 2D boolean mask}) per ROI, from its closed planar contours.
    Contours on the same slice are combined with XOR, so holes are respected. Only the
    slices with contours are held in memory."""
    nx, ny, nz = grid.size
    for roi in ds.get("ROIContourSequence", []):
        slices: dict[int, np.ndarray] = {}
        for contour in roi.get("ContourSequence", []):
            data = contour.get("ContourData")
            if not data or len(data) < 9 or str(contour.get("ContourGeometricType", "CLOSED_PLANAR")) != "CLOSED_PLANAR":
                continue
            idx = grid.to_index(np.asarray(data, float).reshape(-1, 3))
            z = int(round(float(np.median(idx[:, 2]))))
            if not 0 <= z < nz or np.ptp(idx[:, 2]) > 0.5:
                continue                         # outside the volume, or not in an image plane
            poly = fill_polygon(idx[:, 0], idx[:, 1], (ny, nx))
            slices[z] = slices[z] ^ poly if z in slices else poly
        yield int(roi.get("ReferencedROINumber", -1)), slices


@dataclass
class RegionStats:
    """Overlap of ROIs with the de-faced region and their distance (mm) to it."""
    region: np.ndarray                          # (z, y, x)
    spacing_zyx: tuple
    cap_mm: float = 50.0
    _dist: np.ndarray | None = None             # distance to the region, in a crop around it
    _crop: tuple | None = None

    def _distances(self) -> None:
        from scipy.ndimage import distance_transform_edt
        idx = np.nonzero(self.region)
        pad = [int(np.ceil(self.cap_mm / s)) for s in self.spacing_zyx]
        self._crop = tuple(slice(max(0, i.min() - p), i.max() + p + 1) for i, p in zip(idx, pad))
        self._dist = distance_transform_edt(~self.region[self._crop], sampling=self.spacing_zyx)

    def measure(self, slices: dict[int, np.ndarray]) -> tuple[int, int, float]:
        """(voxels, voxels inside the region, smallest distance to the region capped at cap_mm)."""
        voxels = sum(int(m.sum()) for m in slices.values())
        inside = sum(int((m & self.region[z]).sum()) for z, m in slices.items())
        if inside or not voxels:
            return voxels, inside, 0.0 if inside else self.cap_mm
        if not self.region.any():
            return voxels, 0, self.cap_mm
        if self._dist is None:
            self._distances()
        cz, cy, cx = self._crop
        best = self.cap_mm
        for z, m in slices.items():
            if cz.start <= z < cz.stop:
                part = m[cy, cx]
                if part.any():
                    best = min(best, float(self._dist[z - cz.start][part].min()))
        return voxels, 0, best
