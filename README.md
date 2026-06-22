# CT De-facing Tool

Removes facial features from CT images by replacing the face region with air (-1000 HU). Supports single patients and batch processing of entire cohorts.

**Supported input formats:** NIfTI (`.nii`, `.nii.gz`), NRRD (`.nrrd`), DICOM (folder of DICOM files — a `.dcm` extension is not required)  
**Supported operating systems:** Windows, Linux  
**GPU required:** No — runs on CPU out of the box, -d gpu possible

---

## Setup (do this once per computer)

### Step 1 — Install Miniconda

If you do not have conda installed, download and install Miniconda from:  
https://docs.conda.io/en/latest/miniconda.html

Accept all defaults during installation. Then open a terminal:
- **Windows:** search for "Anaconda Prompt" in the Start menu and open it
- **Linux:** open a regular terminal

### Step 2 — Create the software environment

Navigate to this folder and run:

```
conda env create -f environment.yml
```

This downloads all required packages automatically. It only needs to be done once.

### Step 3 — Obtain a license key

This tool requires a free academic license from TotalSegmentator.

1. Go to: https://backend.totalsegmentator.com/license-academic/
2. Fill in your name and institution and submit
3. You will receive a license key by email (e.g. `aca_XXXXXXXXXXXXXXXX`)

### Step 4 — Activate the license

```
conda activate totalseg-deface
totalseg_set_license -l <YOUR_LICENSE_KEY>
```

Replace `<YOUR_LICENSE_KEY>` with the key you received by email. This only needs to be done once per computer.

---

## Usage

Activate the environment before every session:

```
conda activate totalseg-deface
```

### Process a single patient

```
python deface.py -i <input> -o <output_folder>
```

`<input>` can be:
- a `.nii`, `.nii.gz`, or `.nrrd` file
- a folder containing DICOM files for a single CT series

**Examples:**

```bash
# NIfTI
python deface.py -i patient01_CT.nii.gz -o results/patient01

# NRRD
python deface.py -i patient01_IMG_CT.nrrd -o results/patient01

# DICOM folder
python deface.py -i /data/patient01/CT_dicoms/ -o results/patient01
```

### Process multiple patients at once (batch mode)

Organize your data as one sub-folder per patient and run:

```
python deface.py --batch patients/ -o results/
```

Results are saved to `results/<patient_name>/`. If one patient fails, processing continues for the rest.

#### Supported folder layouts

The tool automatically detects the input format in each patient folder. All three layouts below are supported:

**NIfTI or NRRD — CT file inside patient sub-folder:**
```
patients/
├── patient01/
│   └── patient01_IMG_CT.nrrd
├── patient02/
│   └── patient02_IMG_CT.nii.gz
```

**DICOM — flat (all .dcm files directly in the patient folder):**
```
patients/
├── patient01/
│   ├── 000001.dcm
│   ├── 000002.dcm
│   └── ...
├── patient02/
│   └── ...
```

**DICOM — series sub-folder (one or more series per patient):**
```
patients/
├── patient01/
│   ├── CT_20230101/
│   │   ├── 000001.dcm
│   │   └── ...
│   └── MR_20230101/       ← automatically skipped (non-CT modality)
│       └── ...
├── patient02/
│   └── CT/
│       └── ...
```

When a patient folder contains multiple DICOM series, the tool reads the DICOM Modality tag and selects the CT series automatically. If multiple CT series are present, the one with the most files (typically the planning CT) is used. DICOM files are detected by their content, so the `.dcm` file extension is optional — exports that use no extension or DICOM-UID-style filenames are handled as well.

By default the tool looks for NIfTI/NRRD files matching `*CT*.nrrd`, `*CT*.nii.gz`, or `*CT*.nii`. If your files use a different naming convention, specify the pattern with `--ct-glob`:

```
python deface.py --batch patients/ -o results/ --ct-glob "*_CT_*.nii.gz"
```

---

## Output files

| Input type | Output |
|---|---|
| `.nii` / `.nii.gz` | `<name>_deFaced.nii.gz` in the output folder |
| `.nrrd` | `<name>_deFaced.nrrd` in the output folder |
| DICOM folder | `dcm/` subfolder with de-faced DICOM files + `deFaced.nii.gz` for visual QC |

The `face_seg/` subfolder in each output directory contains the intermediate face segmentation mask, which can be used for quality control.

---

## Options

| Option | Default | Description |
|---|---|---|
| `-d cpu` or `-d gpu` | `cpu` | Use GPU if available for faster processing |
| `--dilate N` | `2` | Expands the face mask by N voxels in all directions |
| `--ct-glob "<pattern>"` | see above | File search pattern for batch mode |


---

## Important notes

- **DICOM metadata is not anonymized.** This tool only removes facial features from the image pixel data. Patient name, date of birth, scan date, and other DICOM tags are left unchanged. If full anonymization is required, run the DICOM files through your institution's DICOM de-identification software before or after de-facing.
- **Visual QC is strongly recommended.** After processing, open the output in a viewer (e.g. 3D Slicer) to confirm the face region has been correctly removed, especially for unusual scan geometries or positioning.
- Processed voxels are replaced with -1000 HU (air).
- Output pixel/voxel data type matches the input (e.g. `int16` stays `int16`); only the de-faced region's values change.
- **DICOM output gets new instance UIDs.** Because the pixel data is modified, each de-faced DICOM file is written with a new `SOPInstanceUID`, and the whole de-faced series gets a new shared `SeriesInstanceUID`. `StudyInstanceUID` is left unchanged. This prevents the de-faced series from colliding with (or silently overwriting) the original series if both end up in the same PACS/viewer at the receiving center.
- Input DICOM series using a compressed transfer syntax (e.g. JPEG2000) are automatically decompressed before de-facing; the output is always written uncompressed.

---

## Troubleshooting

**"License number has not been set"**  
Run `totalseg_set_license -l <YOUR_LICENSE_KEY>` with the key from your registration email.

**"No CT images found"** (batch mode)  
Your files may use a different naming pattern. Use `--ct-glob` to specify the correct pattern, e.g. `--ct-glob "*_ct_*.nii.gz"`. For DICOM input this option is not needed — the tool detects DICOM series automatically.

**Wrong series selected** (DICOM batch with multiple series)  
If the tool picks the wrong series, move the unwanted series to a separate folder outside the patient directory, or restructure so each patient folder contains only the CT series you want to de-face.