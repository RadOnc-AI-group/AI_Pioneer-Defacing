"""
Synthetic test data: a head phantom defined in patient (LPS) coordinates, written as DICOM
CT series (any orientation, slice order and pixel encoding), RT objects, NIfTI and NRRD,
plus a stand-in for TotalSegmentator that marks a known face region.

The phantom (millimetres, LPS: +x left, +y posterior, +z superior):
  head      ellipsoid, centre 0, radii (70, 90, 80), soft tissue 40 HU, skull shell 800 HU
  nose      box in front of the head (y from -110 to -85), 40 HU
  face      the region the fake segmentation marks: head/nose voxels with y < FACE_Y,
            |x| < 50 and -40 < z < 30
  couch     plate behind the head (y 100..106), 300 HU (or in front of it with prone=True)
  shell     optional thermoplastic immobilisation mask in front of the face (y -122..-118), 150 HU
  air       -1000 HU with noise, clipped at -1024 HU
  padding   optional: -3024 HU outside a circular field of view
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
RTSTRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
RTDOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
RTPLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"
MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"
ENHANCED_CT = "1.2.840.10008.5.1.4.1.1.2.1"

HEAD_R = np.array([70.0, 90.0, 80.0])
NOSE = ((-10.0, 10.0), (-110.0, -85.0), (-20.0, 10.0))      # x, y, z ranges (LPS)
FACE_Y = -60.0
FACE_X, FACE_Z = 50.0, (-40.0, 30.0)
TISSUE_HU, BONE_HU, NOSE_HU, COUCH_HU, AIR_HU, SHELL_HU = 40.0, 800.0, 40.0, 300.0, -1000.0, 150.0
SHELL = ((-40.0, 40.0), (-122.0, -118.0), (-35.0, 25.0))

# Image orientations (row cosine, column cosine) of the tested acquisitions
ORIENTATIONS = {
    "axial_hfs": ([1, 0, 0], [0, 1, 0]),        # supine: rows run towards posterior
    "axial_prone": ([-1, 0, 0], [0, -1, 0]),    # prone: rows run towards anterior
    "axial_ffs": ([-1, 0, 0], [0, 1, 0]),
    "sagittal": ([0, 1, 0], [0, 0, -1]),
    "coronal": ([1, 0, 0], [0, 0, -1]),
    "oblique": ([np.cos(0.2), np.sin(0.2), 0], [-np.sin(0.2), np.cos(0.2), 0]),
}


# ---------------------------------------------------------------------------
# Phantom in world coordinates
# ---------------------------------------------------------------------------

def inside_head(p: np.ndarray) -> np.ndarray:
    return ((p / HEAD_R) ** 2).sum(axis=-1) <= 1.0


def in_box(p: np.ndarray, box) -> np.ndarray:
    return np.all([(p[..., i] >= lo) & (p[..., i] <= hi) for i, (lo, hi) in enumerate(box)], axis=0)


def in_nose(p: np.ndarray) -> np.ndarray:
    return in_box(p, NOSE)


def in_face(p: np.ndarray) -> np.ndarray:
    """The face region the fake TotalSegmentator marks (LPS points)."""
    near = (p[..., 1] < FACE_Y) & (np.abs(p[..., 0]) < FACE_X) & (p[..., 2] > FACE_Z[0]) & (p[..., 2] < FACE_Z[1])
    return near & (inside_head(p) | in_nose(p))


def in_shell(p: np.ndarray) -> np.ndarray:
    return in_box(p, SHELL)


def phantom_hu(p: np.ndarray, rng: np.random.Generator, prone: bool = False,
               fov_radius: float | None = None, noise: float = 15.0, shell: bool = False) -> np.ndarray:
    """HU values at the LPS points p (..., 3)."""
    q = ((p / HEAD_R) ** 2).sum(axis=-1)
    hu = np.full(p.shape[:-1], AIR_HU) + rng.normal(0, noise, p.shape[:-1])
    hu = np.maximum(hu, -1024.0)
    head = q <= 1.0
    hu[head] = TISSUE_HU + rng.normal(0, noise, int(head.sum()))
    skull = (q > 0.72) & (q <= 0.85)
    hu[skull] = BONE_HU
    hu[in_nose(p)] = NOSE_HU
    if shell:
        hu[in_shell(p)] = SHELL_HU
    couch_y = (-112.0, -106.0) if prone else (100.0, 106.0)
    hu[in_box(p, ((-120, 120), couch_y, (-1e9, 1e9)))] = COUCH_HU
    if fov_radius is not None:
        hu[np.hypot(p[..., 0], p[..., 1]) > fov_radius] = -3024.0
    return hu


# ---------------------------------------------------------------------------
# DICOM CT series
# ---------------------------------------------------------------------------

def _file_meta(sop_class: str, sop_uid: str, ts: str = ExplicitVRLittleEndian) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ts
    meta.ImplementationClassUID = generate_uid()
    return meta


def _patient(ds: Dataset, patient: str, study_uid: str, frame_uid: str) -> None:
    ds.PatientName = patient
    ds.PatientID = patient
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.StudyID = "1"
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.FrameOfReferenceUID = frame_uid


class CTSeries:
    """Geometry and files of a written CT series."""

    def __init__(self, paths, sop_uids, series_uid, study_uid, frame_uid, origin, row, col, normal,
                 spacing, shape, hu):
        self.paths, self.sop_uids = paths, sop_uids
        self.series_uid, self.study_uid, self.frame_uid = series_uid, study_uid, frame_uid
        self.origin, self.row, self.col, self.normal = origin, row, col, normal
        self.spacing, self.shape, self.hu = spacing, shape, hu   # (dc, dr, dz), (nz, nr, nc), HU (z, r, c)

    def world(self) -> np.ndarray:
        """LPS position of every voxel, shape (nz, nr, nc, 3), slices in geometric order."""
        nz, nr, nc = self.shape
        k, r, c = np.meshgrid(np.arange(nz), np.arange(nr), np.arange(nc), indexing="ij")
        dc, dr, dz = self.spacing
        return (self.origin + k[..., None] * dz * self.normal + r[..., None] * dr * self.col
                + c[..., None] * dc * self.row)


def write_ct_series(folder: Path, *, orientation: str = "axial_hfs", shape=(36, 56, 56),
                    spacing=(4.5, 4.5, 6.0), signed: bool = True, bits_stored: int = 16,
                    intercept: float = 0.0, slope: float = 1.0, prone: bool = False,
                    fov_radius: float | None = None, descending: bool = False, names: str = "uid",
                    patient: str = "PAT01", study_uid: str | None = None, frame_uid: str | None = None,
                    series_desc: str = "Head CT", compress: str | None = None, seed: int = 0,
                    image_type=("ORIGINAL", "PRIMARY", "AXIAL"), extra=None, shell: bool = False,
                    tilt: float = 0.0, centre_z: float = 0.0, text: bool = False) -> CTSeries:
    """Write a CT series of the phantom; returns its geometry (slices in geometric order).
    tilt: shift (mm) of each slice along the column direction, as with a gantry tilt.
    centre_z: head-feet position of the scan (e.g. -400 for a chest CT without the head).
    text: burned-in "text" (bright strokes in the image corner of every slice)."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    row, col = (np.array(v, float) for v in ORIENTATIONS[orientation])
    normal = np.cross(row, col)
    nz, nr, nc = shape
    dc, dr, dz = spacing
    centre = np.array([0.0, 0.0, centre_z])
    origin = centre - (nc - 1) / 2 * dc * row - (nr - 1) / 2 * dr * col - (nz - 1) / 2 * dz * normal
    series = CTSeries([], [], generate_uid(), study_uid or generate_uid(), frame_uid or generate_uid(),
                      origin, row, col, normal, spacing, shape, None)
    world = series.world()
    if tilt:
        world = world + (np.arange(nz)[:, None, None, None] * tilt) * col
    hu = phantom_hu(world, rng, prone=prone, fov_radius=fov_radius, shell=shell)
    if text:
        hu[:, 2:7, 2:22:3] = 3071.0                 # "letters" in the corner, outside the body
        hu[:, 2, 2:22] = 3071.0
    if not signed and intercept == 0:
        hu = np.maximum(hu, 0)          # cannot be stored; image without air (still a valid file)
    stored = np.round((hu - intercept) / slope)
    info_lo, info_hi = ((-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1) if signed
                        else (0, (1 << bits_stored) - 1))
    stored = np.clip(stored, info_lo, info_hi)
    series.hu = stored * slope + intercept
    order = list(range(nz))[::-1] if descending else list(range(nz))
    for n, k in enumerate(order):
        sop = generate_uid()
        ds = Dataset()
        ds.file_meta = _file_meta(CT_IMAGE_STORAGE, sop)
        ds.SOPClassUID, ds.SOPInstanceUID = CT_IMAGE_STORAGE, sop
        ds.Modality = "CT"
        _patient(ds, patient, series.study_uid, series.frame_uid)
        ds.SeriesInstanceUID = series.series_uid
        ds.SeriesNumber = 2
        ds.SeriesDescription = series_desc
        ds.InstanceNumber = n + 1
        ds.ImageType = list(image_type)
        ds.ImagePositionPatient = [f"{v:.4f}" for v in origin + k * dz * normal + k * tilt * col]
        ds.ImageOrientationPatient = [f"{v:.6f}" for v in np.concatenate([row, col])]
        ds.PixelSpacing = [f"{dr:g}", f"{dc:g}"]
        ds.SliceThickness = f"{dz:g}"
        ds.Rows, ds.Columns = nr, nc
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = bits_stored
        ds.HighBit = bits_stored - 1
        ds.PixelRepresentation = 1 if signed else 0
        ds.RescaleIntercept = f"{intercept:g}"
        ds.RescaleSlope = f"{slope:g}"
        ds.RescaleType = "HU"
        for key, value in (extra or {}).items():
            setattr(ds, key, value)
        ds.PixelData = stored[k].astype(np.int16 if signed else np.uint16).tobytes()
        if compress == "rle":
            from pydicom.uid import RLELossless
            ds.compress(RLELossless)
        name = {"uid": f"CT.{sop}.dcm", "plain": f"IM{n:04d}", "shuffled": f"{rng.integers(1e9):09d}.dcm"}[names]
        path = folder / name
        ds.save_as(path, enforce_file_format=True)
        series.paths.append(path)
        series.sop_uids.append(sop)
    # paths/sop_uids in geometric order (k = 0..nz-1)
    if descending:
        series.paths, series.sop_uids = series.paths[::-1], series.sop_uids[::-1]
    return series


def read_series_hu(folder: Path) -> tuple[np.ndarray, np.ndarray, list[Dataset]]:
    """HU volume and LPS voxel positions of a DICOM series, read with pydicom only."""
    dss = [pydicom.dcmread(p) for p in sorted(folder.iterdir()) if p.is_file()]
    iop = np.array(dss[0].ImageOrientationPatient, float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    dss.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))
    hu = np.stack([d.pixel_array * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in dss])
    dr, dc = (float(v) for v in dss[0].PixelSpacing)
    nr, nc = dss[0].Rows, dss[0].Columns
    r, c = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    world = np.stack([np.array(d.ImagePositionPatient, float) + r[..., None] * dr * col
                      + c[..., None] * dc * row for d in dss])
    return hu, world, dss


# ---------------------------------------------------------------------------
# Other DICOM objects
# ---------------------------------------------------------------------------

def _ellipse(z: float, n: int = 24, scale: float = 1.0) -> list[float] | None:
    s = 1 - (z / HEAD_R[2]) ** 2
    if s <= 0.05:
        return None
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rx, ry = HEAD_R[0] * np.sqrt(s) * scale, HEAD_R[1] * np.sqrt(s) * scale
    return [v for t in a for v in (rx * np.cos(t), ry * np.sin(t), z)]


def _box_contour(x, y, z) -> list[float]:
    (x0, x1), (y0, y1) = x, y
    return [x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]


def write_rtstruct(path: Path, ct: CTSeries, rois: list[tuple[str, str]] | None = None,
                   patient: str = "PAT01") -> Dataset:
    """RTSTRUCT on ct. rois: (name, RTROIInterpretedType); geometry by name:
    names containing 'eye'/'lips' -> box in the face region; 'ptv'/'brain' -> small ellipse in
    the centre; anything else -> the outer head outline."""
    rois = rois or [("BODY", "EXTERNAL"), ("PTV_60", "PTV"), ("Eye_L", "ORGAN")]
    sop = generate_uid()
    ds = Dataset()
    ds.file_meta = _file_meta(RTSTRUCT_STORAGE, sop)
    ds.SOPClassUID, ds.SOPInstanceUID = RTSTRUCT_STORAGE, sop
    ds.Modality = "RTSTRUCT"
    _patient(ds, patient, ct.study_uid, ct.frame_uid)
    del ds.FrameOfReferenceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.StructureSetLabel = "RS"
    ds.StructureSetDate = "20240101"
    zs = [float(np.dot(ct.normal, ct.origin + k * ct.spacing[2] * ct.normal)) for k in range(ct.shape[0])]
    images = Sequence()
    for uid in ct.sop_uids:
        it = Dataset()
        it.ReferencedSOPClassUID, it.ReferencedSOPInstanceUID = CT_IMAGE_STORAGE, uid
        images.append(it)
    series = Dataset()
    series.SeriesInstanceUID = ct.series_uid
    series.ContourImageSequence = images
    study = Dataset()
    study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    study.ReferencedSOPInstanceUID = ct.study_uid
    study.RTReferencedSeriesSequence = Sequence([series])
    frame = Dataset()
    frame.FrameOfReferenceUID = ct.frame_uid
    frame.RTReferencedStudySequence = Sequence([study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([frame])
    ss, contours, obs = Sequence(), Sequence(), Sequence()
    for number, (name, kind) in enumerate(rois, 1):
        item = Dataset()
        item.ROINumber, item.ReferencedFrameOfReferenceUID, item.ROIName = number, ct.frame_uid, name
        item.ROIGenerationAlgorithm = "MANUAL"
        ss.append(item)
        cs = Sequence()
        for k, z in enumerate(zs):
            low = name.lower()
            if "eye" in low or "lips" in low:
                if not (-20 <= z <= 10):
                    continue
                data = _box_contour((-30, -10), (-84, -70), z)
            elif "ptv" in low or "brain" in low:
                data = _ellipse(z, scale=0.3)
            else:
                data = _ellipse(z)
            if data is None:
                continue
            c = Dataset()
            img = Dataset()
            img.ReferencedSOPClassUID, img.ReferencedSOPInstanceUID = CT_IMAGE_STORAGE, ct.sop_uids[k]
            c.ContourImageSequence = Sequence([img])
            c.ContourGeometricType = "CLOSED_PLANAR"
            c.NumberOfContourPoints = len(data) // 3
            c.ContourData = [f"{v:.3f}" for v in data]
            cs.append(c)
        rc = Dataset()
        rc.ReferencedROINumber = number
        rc.ROIDisplayColor = [255, 0, 0]
        rc.ContourSequence = cs
        contours.append(rc)
        o = Dataset()
        o.ObservationNumber, o.ReferencedROINumber = number, number
        o.RTROIInterpretedType, o.ROIInterpreter = kind, ""
        obs.append(o)
    ds.StructureSetROISequence, ds.ROIContourSequence, ds.RTROIObservationsSequence = ss, contours, obs
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return ds


def write_simple(path: Path, modality: str, sop_class: str, ct: CTSeries | None = None,
                 patient: str = "PAT01", **attrs) -> Dataset:
    """A minimal DICOM object (RTDOSE, RTPLAN, MR, ...) without pixel data."""
    sop = generate_uid()
    ds = Dataset()
    ds.file_meta = _file_meta(sop_class, sop)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, sop
    ds.Modality = modality
    _patient(ds, patient, ct.study_uid if ct else generate_uid(), ct.frame_uid if ct else generate_uid())
    ds.SeriesInstanceUID = generate_uid()
    for key, value in attrs.items():
        setattr(ds, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return ds


# ---------------------------------------------------------------------------
# NIfTI / NRRD
# ---------------------------------------------------------------------------

def nifti_affine(orientation: str = "ras", shape=(56, 56, 36), spacing=(4.5, 4.5, 6.0)) -> np.ndarray:
    """RAS affine of an image centred on the origin. 'ras', 'lps' (flipped x and y) or 'las'."""
    signs = {"ras": (1, 1, 1), "lps": (-1, -1, 1), "las": (-1, 1, 1)}[orientation]
    a = np.diag([s * d for s, d in zip(signs, spacing)] + [1.0])
    a[:3, 3] = -a[:3, :3] @ ((np.array(shape) - 1) / 2)
    return a


def nifti_world_lps(affine: np.ndarray, shape) -> np.ndarray:
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape[:3]), indexing="ij")
    ras = np.stack([i, j, k, np.ones_like(i)], axis=-1) @ affine.T
    return ras[..., :3] * np.array([-1.0, -1.0, 1.0])


def write_nifti(path: Path, orientation: str = "ras", dtype=np.int16, shape=(56, 56, 36),
                spacing=(4.5, 4.5, 6.0), seed: int = 0, slope: float | None = None,
                inter: float | None = None, descrip: str = "", centre_z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Phantom as NIfTI; returns (HU data as stored after scaling, affine)."""
    affine = nifti_affine(orientation, shape, spacing)
    affine[2, 3] += centre_z
    hu = phantom_hu(nifti_world_lps(affine, shape), np.random.default_rng(seed))
    img = nib.Nifti1Image(np.zeros(shape, dtype), affine)
    if slope is not None:
        raw = np.round((hu - inter) / slope).astype(dtype)
        img = nib.Nifti1Image(raw, affine)
        img.header.set_slope_inter(slope, inter)
    else:
        img = nib.Nifti1Image(np.round(hu).astype(dtype), affine)
    img.header.set_data_dtype(dtype)
    if descrip:
        img.header["descrip"] = descrip
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    loaded = nib.load(str(path))
    return loaded.get_fdata(), loaded.affine


# ---------------------------------------------------------------------------
# Stand-in for TotalSegmentator
# ---------------------------------------------------------------------------

def fake_face_mask(input_nii: Path, out_path: Path, empty: bool = False) -> None:
    """Write the phantom's face region on the grid of input_nii (like TotalSegmentator)."""
    img = nib.load(str(input_nii))
    mask = np.zeros(img.shape[:3], np.uint8)
    if not empty:
        mask[in_face(nifti_world_lps(img.affine, img.shape))] = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, img.affine), str(out_path))
