#!/usr/bin/env python
"""
CT de-facing tool — removes facial features from CT images.

Single patient:
    python deface.py -i <input.nii / .nii.gz / .nrrd / dicom_dir> -o <output_dir>

Batch (folder of patient sub-folders):
    python deface.py --batch <patients_dir> -o <output_dir> [--ct-glob "*CT*.nrrd"]

See README.md for full setup instructions and options.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


FACE_VALUE_HU = -1000.0  # value written into de-faced voxels (air HU for CT)

# Default glob patterns tried in order when discovering CT files in batch mode
DEFAULT_CT_GLOBS = ["*CT*.nrrd", "*CT*.nii.gz", "*CT*.nii", "*ct*.nrrd", "*ct*.nii.gz"]

# Recognized image file extensions, longest-first so ".nii.gz" wins over ".nii"
KNOWN_SUFFIXES = (".nii.gz", ".nii", ".nrrd")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def classify_suffix(path: Path) -> str:
    """
    Return the recognized extension of path ('.nii.gz', '.nii', '.nrrd') or ''.

    Unlike Path.suffixes, this only looks at the end of the filename, so
    filenames with extra dots (DICOM-UID-style names, versioned filenames,
    etc.) are still classified correctly.
    """
    name = path.name.lower()
    for suffix in KNOWN_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def strip_known_suffix(path: Path) -> str:
    """Return the filename with its recognized image extension removed."""
    suffix = classify_suffix(path)
    if suffix:
        return path.name[: -len(suffix)]
    return path.stem


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def safe_cast(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Round/clip data into dtype's representable range before casting, to avoid wraparound."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        data = np.clip(np.round(data), info.min, info.max)
    return data.astype(dtype)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_nifti(input_path: Path, tmp_dir: Path) -> Path:
    """Return a NIfTI path for TotalSegmentator. Converts nrrd or dcm dir if needed."""
    suffix = classify_suffix(input_path)
    if suffix in (".nii.gz", ".nii"):
        return input_path
    if suffix == ".nrrd":
        out = tmp_dir / (input_path.stem + ".nii.gz")
        img = sitk.ReadImage(str(input_path))
        sitk.WriteImage(img, str(out))
        return out
    if input_path.is_dir():
        result = subprocess.run(
            ["dcm2niix", "-z", "y", "-f", "input", "-o", str(tmp_dir), str(input_path)],
            check=True, capture_output=True, text=True,
        )
        niis = sorted(tmp_dir.glob("input*.nii.gz"))
        if not niis:
            raise FileNotFoundError(f"dcm2niix produced no output for {input_path}")
        if len(niis) > 1:
            print(
                f"WARNING: dcm2niix produced {len(niis)} NIfTI files for {input_path} "
                f"({', '.join(n.name for n in niis)}); using '{niis[0].name}'. This can "
                f"happen with gantry tilt or multiple series in one folder — verify the result.",
                file=sys.stderr,
            )
        return niis[0]
    raise ValueError(f"Unsupported input type: {input_path}")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def run_totalsegmentator(input_nii: Path, seg_dir: Path, device: str) -> Path:
    # Note: the "face" task does not support --fast (TotalSegmentator raises ValueError).
    subprocess.run(
        ["TotalSegmentator", "-i", str(input_nii), "-o", str(seg_dir), "-ta", "face", "-d", device],
        check=True,
    )
    mask_path = seg_dir / "face.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Expected face mask not found at {mask_path}")
    return mask_path


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    """True if two images occupy exactly the same voxel grid (size + physical space)."""
    return (a.GetSize() == b.GetSize()
            and a.GetSpacing() == b.GetSpacing()
            and a.GetOrigin() == b.GetOrigin()
            and a.GetDirection() == b.GetDirection())


def _dilated_mask(mask_sitk: sitk.Image, dilate_vox: int, reference: sitk.Image = None) -> np.ndarray:
    """Return a boolean numpy mask, resampled onto the reference image grid if given.

    The mask and the reference image can reach this function through different
    converters (e.g. dcm2niix vs. SimpleITK's DICOM reader), which may produce
    volumes of identical size but different orientation. Comparing the full grid
    (not just size) and resampling in physical space whenever it differs avoids a
    silent misalignment that would deface the wrong region. Resampling is a no-op
    when the grids already match exactly.
    """
    if reference is not None and not _same_grid(reference, mask_sitk):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask_sitk = resampler.Execute(sitk.Cast(mask_sitk, sitk.sitkUInt8))
    mask = sitk.GetArrayFromImage(mask_sitk) > 0
    if dilate_vox > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilate_vox)
    return mask


def _mask_from_nifti(mask_img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> np.ndarray:
    """Boolean mask array aligned to ref_img's voxel grid, resampling via the affines if needed."""
    if mask_img.shape[:3] == ref_img.shape[:3]:
        return mask_img.get_fdata() > 0
    from scipy.ndimage import affine_transform
    # Maps a ref-image voxel index to the corresponding mask-image voxel index.
    transform = np.linalg.inv(mask_img.affine) @ ref_img.affine
    resampled = affine_transform(
        mask_img.get_fdata(), transform[:3, :3], offset=transform[:3, 3],
        output_shape=ref_img.shape[:3], order=0, mode="constant", cval=0,
    )
    return resampled > 0.5


def deface_nifti(image_path: Path, mask_path: Path, out_path: Path, dilate_vox: int) -> None:
    img = nib.load(str(image_path))
    mask_img = nib.load(str(mask_path))
    orig_dtype = img.get_data_dtype()
    data = img.get_fdata().astype(np.float64)
    mask = _mask_from_nifti(mask_img, img)
    if dilate_vox > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilate_vox)
    data[mask] = FACE_VALUE_HU
    data = safe_cast(data, orig_dtype)
    out_img = nib.Nifti1Image(data, img.affine, img.header)
    out_img.set_data_dtype(orig_dtype)
    nib.save(out_img, str(out_path))


def deface_nrrd(image_path: Path, mask_path: Path, out_path: Path, dilate_vox: int) -> None:
    img = sitk.ReadImage(str(image_path))
    mask_sitk = sitk.ReadImage(str(mask_path))
    mask = _dilated_mask(mask_sitk, dilate_vox, reference=img)
    orig_dtype = sitk.GetArrayViewFromImage(img).dtype
    data = sitk.GetArrayFromImage(img).astype(np.float64)
    data[mask] = FACE_VALUE_HU
    data = safe_cast(data, orig_dtype)
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))


def deface_dicom_dir(dicom_dir: Path, mask_path: Path, out_dir: Path, dilate_vox: int) -> None:
    import pydicom
    from pydicom.uid import generate_uid

    mask_sitk = sitk.ReadImage(str(mask_path))
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    mask = _dilated_mask(mask_sitk, dilate_vox, reference=image)
    out_dir.mkdir(parents=True, exist_ok=True)

    # New SeriesInstanceUID for the whole de-faced series (pixel data differs from the
    # original, so it must not share UIDs with it — avoids PACS collisions/overwrites
    # at the receiving center). StudyInstanceUID is left unchanged: same study/exam.
    new_series_uid = generate_uid()
    for i, fname in enumerate(file_names):
        ds = pydicom.dcmread(fname)

        if ds.file_meta.TransferSyntaxUID.is_compressed:
            # Decompresses PixelData in place, switches TransferSyntaxUID to an
            # uncompressed one, and generates a new SOPInstanceUID. Without this,
            # writing decoded pixel bytes back under the original compressed
            # TransferSyntaxUID produces a DICOM file with corrupt/undecodable pixel data.
            ds.decompress()
        else:
            ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesInstanceUID = new_series_uid

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixels = ds.pixel_array
        hu = pixels.astype(np.float64) * slope + intercept
        hu[mask[i]] = FACE_VALUE_HU
        ds.PixelData = safe_cast((hu - intercept) / slope, pixels.dtype).tobytes()
        if "ImageComments" in ds:
            ds.ImageComments = str(ds.ImageComments) + " | de-faced (TotalSegmentator-face)"
        else:
            ds.ImageComments = "De-faced (TotalSegmentator-face)"
        ds.save_as(str(out_dir / Path(fname).name))


# ---------------------------------------------------------------------------
# Single-patient pipeline
# ---------------------------------------------------------------------------

def deface_one(input_path: Path, out_dir: Path, device: str, dilate_vox: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "face_seg"
    is_dicom = input_path.is_dir()
    suffix = classify_suffix(input_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nii_path = to_nifti(input_path, tmp)
        mask_path = run_totalsegmentator(nii_path, seg_dir, device)

        if is_dicom:
            deface_dicom_dir(input_path, mask_path, out_dir / "dcm", dilate_vox)
            deface_nifti(nii_path, mask_path, out_dir / "deFaced.nii.gz", dilate_vox)
        elif suffix == ".nrrd":
            stem = strip_known_suffix(input_path)
            deface_nrrd(input_path, mask_path, out_dir / f"{stem}_deFaced.nrrd", dilate_vox)
        else:
            stem = strip_known_suffix(input_path)
            deface_nifti(input_path, mask_path, out_dir / f"{stem}_deFaced.nii.gz", dilate_vox)

    print(f"Done: {input_path.name}  ->  {out_dir}")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def find_ct_in_folder(folder: Path, ct_globs: list[str]) -> Path | None:
    """Return the first file matching any of the CT glob patterns.

    Files whose name starts with '.' are ignored — this skips macOS AppleDouble
    sidecars (e.g. '._patient_CT.nrrd'), which otherwise match the glob and, being
    alphabetically first, would be picked ahead of the real CT.
    """
    for pattern in ct_globs:
        matches = sorted(m for m in folder.glob(pattern) if not m.name.startswith("."))
        if matches:
            return matches[0]
    return None


def find_dicom_ct_series(patient_dir: Path) -> Path | None:
    """
    Find the CT series directory for a patient folder. Handles three common layouts:
      1. patient_dir/<dcm files>      — DICOM files flat in the patient folder
      2. patient_dir/<series>/<dcm>   — one series subfolder
      3. patient_dir/<CT>/,
         patient_dir/<MR>/...         — multiple series; picks CT by DICOM Modality tag,
                                        and if still ambiguous, the series with the most files
    DICOM files are detected by content (via GDCM), so a '.dcm' file extension is
    not required — many scanners export DICOM with no extension. Returns the
    directory containing the CT DICOM files, or None if no DICOM series is found.
    """
    import pydicom

    reader = sitk.ImageSeriesReader()

    def series_ids(folder: Path) -> list[str]:
        try:
            return list(reader.GetGDCMSeriesIDs(str(folder)))
        except Exception:
            return []

    def num_dicom_files(folder: Path) -> int:
        return sum(len(reader.GetGDCMSeriesFileNames(str(folder), sid)) for sid in series_ids(folder))

    def modality(folder: Path) -> str:
        ids = series_ids(folder)
        if not ids:
            return ""
        files = reader.GetGDCMSeriesFileNames(str(folder), ids[0])
        if not files:
            return ""
        try:
            return pydicom.dcmread(files[0], stop_before_pixels=True).get("Modality", "")
        except Exception:
            return ""

    # Layout 1: DICOM files directly in patient_dir
    if series_ids(patient_dir):
        return patient_dir

    # Layout 2 / 3: one level of series subfolders
    series_dirs = [d for d in sorted(patient_dir.iterdir()) if d.is_dir() and series_ids(d)]
    if not series_dirs:
        return None
    if len(series_dirs) == 1:
        return series_dirs[0]

    # Multiple series — prefer CT modality, then largest series
    ct_series = [d for d in series_dirs if modality(d) == "CT"]
    candidates = ct_series if ct_series else series_dirs
    return max(candidates, key=num_dicom_files)


def discover_patients(patients_dir: Path, ct_globs: list[str]) -> list[tuple[Path, str]]:
    """
    Return (input_path, patient_name) pairs for all patients found under patients_dir.
    Detection order per subfolder:
      1. NIfTI / NRRD file matching ct_globs
      2. DICOM series (flat or nested one level, CT modality preferred)
    Also handles NIfTI/NRRD files placed directly in patients_dir (no subfolder).
    """
    entries = []
    for item in sorted(patients_dir.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            ct_file = find_ct_in_folder(item, ct_globs)
            if ct_file:
                entries.append((ct_file, item.name))
            else:
                dcm_dir = find_dicom_ct_series(item)
                if dcm_dir:
                    entries.append((dcm_dir, item.name))
        else:
            if classify_suffix(item):
                entries.append((item, strip_known_suffix(item)))
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Single input: .nii, .nii.gz, .nrrd, or DICOM directory")
    group.add_argument("--batch", metavar="PATIENTS_DIR",
                       help="Batch mode: folder containing one sub-folder (or file) per patient")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("-d", "--device", default="cpu", help="cpu or gpu (default: cpu)")
    p.add_argument("--dilate", type=int, default=2,
                   help="Voxels to dilate the face mask by for safety margin (default: 2)")
    p.add_argument("--ct-glob", default=None,
                   help=f"Glob pattern(s) to find the CT file inside each patient folder in batch mode "
                        f"(comma-separated, default: {','.join(DEFAULT_CT_GLOBS)})")
    args = p.parse_args()

    out_dir = Path(args.output)
    ct_globs = [g.strip() for g in args.ct_glob.split(",")] if args.ct_glob else DEFAULT_CT_GLOBS

    if args.input:
        deface_one(Path(args.input), out_dir, args.device, args.dilate)
    else:
        patients_dir = Path(args.batch)
        patients = discover_patients(patients_dir, ct_globs)
        if not patients:
            print(f"ERROR: No CT images found in {patients_dir} using patterns: {ct_globs}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(patients)} patient(s) to process.")
        failed = []
        for i, (input_path, patient_name) in enumerate(patients, 1):
            print(f"\n[{i}/{len(patients)}] {patient_name}: {input_path.name}")
            try:
                deface_one(input_path, out_dir / patient_name, args.device, args.dilate)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                failed.append(patient_name)
        if failed:
            print(f"\nFailed ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
            sys.exit(1)
        print(f"\nAll done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
