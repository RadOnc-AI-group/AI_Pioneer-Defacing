# CT De-facing Tool

Removes facial features from CT images, and from the radiotherapy (RT) objects that come with them, so that CT data can be shared without showing the patient's face. The face is found with TotalSegmentator (`face` task). Single patients and whole cohorts (batch mode, with resume after interruptions) are supported.

**Supported input formats:** DICOM (a patient folder with CT series and RT objects, searched recursively; a `.dcm` extension is not required), NIfTI (`.nii`, `.nii.gz`), NRRD (`.nrrd`)  
**Supported operating systems:** Windows, Linux, macOS  
**GPU required:** No. It runs on the CPU by default; `-d gpu` (CUDA) or `-d mps` (Apple) are possible.

> **This tool de-faces images. It does not anonymise metadata.** Patient names, IDs, dates etc. stay in the DICOM headers; the tool warns about them (see [Metadata check](#metadata-check-for-patient-identifying-data)). Run a DICOM de-identification tool before sharing.

---

## Setup (do this once per computer)

### Step 1: Install Miniconda

If you do not have conda installed, download and install Miniconda from:  
https://docs.conda.io/en/latest/miniconda.html

Accept all defaults during installation. Then open a terminal:
- **Windows:** search for "Anaconda Prompt" in the Start menu and open it
- **Linux / macOS:** open a regular terminal

### Step 2: Create the software environment

Navigate to this folder and run:

```
conda env create -f environment.yml
```

This downloads all required packages automatically. It only needs to be done once.

*Without conda:* create a Python 3.10 virtual environment and run `pip install -r requirements.txt`.

### Step 3: Obtain a license key

This tool requires a free academic license from TotalSegmentator.

1. Go to: https://backend.totalsegmentator.com/license-academic/
2. Fill in your name and institution and submit
3. You will receive a license key by email (e.g. `aca_XXXXXXXXXXXXXXXX`)

### Step 4: Activate the license

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
- a DICOM patient folder: all CT series and RT objects below it are found by their content, also in sub-folders
- a `.nii`, `.nii.gz`, or `.nrrd` file

```bash
python deface.py -i /data/patient01/ -o results/patient01          # DICOM (CT + RTSTRUCT, ...)
python deface.py -i patient01_CT.nii.gz -o results/patient01        # NIfTI
python deface.py -i patient01_IMG_CT.nrrd -o results/patient01      # NRRD
```

The output folder must not exist yet (or be empty). Confidential QC files go to `results/patient01_qc/` (see [Output](#output)).

### Process multiple patients at once (batch mode)

```
python deface.py --batch patients/ -o results/
```

Results are saved to `results/<patient>/`, QC files and reports to `results_qc/`. If one patient fails, processing continues with the rest. If the patients are grouped in sub-folders (e.g. `patients/<diagnosis>/<patient>/`), use `--depth 2`.

**Resume.** Each patient is written to `<patient>.partial` and renamed when complete. An interrupted run (Ctrl+C, sleep, unplugged drive) can simply be started again: finished patients are skipped, unfinished ones are redone. A patient folder in the output without a manifest in the QC folder (e.g. from an earlier version of this tool) is reported and must be deleted before it is redone. Patients finished earlier with other settings are reported as well.

**Try it first:** `--dry-run` shows what would be done (which series are de-faced, which files and contours are removed, the metadata check) without writing any data; `--only TEXT` restricts the run to patients whose path contains `TEXT`.

#### Supported folder layouts

```
patients/
├── patient01/                     DICOM, any structure below the patient folder
│   ├── CT/ ...                    ← every CT series is de-faced
│   ├── CT_replanning/ ...
│   ├── RS.1.2.3.dcm               ← RTSTRUCT: copied, Body/Skin removed
│   ├── RD.1.2.4.dcm               ← RTDOSE: not copied (default)
│   └── MR/ ...                    ← not copied (cannot be de-faced)
├── patient02/
│   └── patient02_IMG_CT.nrrd      NIfTI/NRRD matching --ct-glob
└── patient03_CT.nii.gz            image file directly in the batch folder
```

For NIfTI/NRRD, the tool looks for files matching `*CT*.nrrd`, `*CT*.nii.gz`, `*CT*.nii` (and lower-case `ct`). Use `--ct-glob` for other names, e.g. `--ct-glob "*_CT_*.nii.gz"`.

---

## What is done by default

| Content | Default | Change with |
|---|---|---|
| CT series (all, except localizers/scouts) | de-faced: face mask dilated, extended to the anterior image edge, filled with air sampled from the same scan; new SOP Instance and Series UIDs | `--dilate`, `--no-extend-anterior`, `--fill` |
| RTSTRUCT | copied; UIDs remapped so it still refers to the de-faced CT | |
| ↳ Body/External contour | **removed** (it traces the skin, i.e. the face) | `--keep-body` |
| ↳ Skin/Haut contour | **removed** | `--keep-skin` |
| ↳ other contours (targets, organs at risk) | kept unchanged; names that might follow the skin are reported | `--remove-roi REGEX` |
| RTDOSE | **not copied** (the dose is zero outside the body, so it outlines the face) | `--keep-rtdose` |
| RTPLAN, RTRECORD | **not copied** | `--keep-rtplan` |
| CT localizers/scouts, MR, PET, REG, RT images, non-DICOM files | not copied (cannot be de-faced); listed in the log | |
| De-faced DICOM | read back and verified (see [Verification](#verification-of-the-written-dicom)) | |
| Metadata | checked for patient-identifying data and reported; **not changed** | `--no-pii-check` |

### Body and Skin contours

ROI names are compared case-insensitively, with umlauts written out (`ö` = `oe`, `ß` = `ss`).

- **Removed:** ROIs of type `EXTERNAL`, and ROIs whose *whole* name is one of `Body`, `External`, `Outline`, `Körper`/`Koerper`, `Körperkontur`, `Außenkontur`/`Aussenkontur`, `Patient`, `Patientenkontur` (Body), or `Skin`, `Haut`, `Hautkontur` (Skin), optionally followed by a number or thickness, e.g. `BODY_1`, `Skin 5mm`, `Haut_3mm`.
- **Kept and reported:** ROIs whose name only *contains* one of these words, or `mask`/`Maske`, `bolus`, `shell`. Examples: `Körperteile`, `Body-PTV`, `z_Skin_ring`, `Maske`. Many such names are real anatomy and must not be removed automatically, e.g. `Glaskörper` (vitreous body), `Wirbelkörper`, `Vertebral_Body`. Check the reported names; remove those that follow the skin or face with `--remove-roi`, e.g. `--remove-roi "Body-PTV" --remove-roi "z_Skin.*"`.
- A structure set without a recognisable Body contour is reported, so that a skin contour with an unusual name is not overlooked.

The removed contours are saved in the QC folder as a separate RTSTRUCT that refers to the de-faced CT, so they can be checked in a viewer. Plans and records that refer to removed objects keep a dangling reference; this does not matter when they are not shared.

---

## Options

| Option | Default | Description |
|---|---|---|
| `-d cpu` / `gpu` / `mps` | `cpu` | Device for TotalSegmentator |
| `--dilate N` | `2` | Expands the face mask by N voxels in all directions |
| `--no-extend-anterior` | off (extension on) | Removes only the dilated face mask, not everything in front of it (see below) |
| `--fill air` / `soft-tissue` / `constant` | `air` | Value of the de-faced voxels (see below) |
| `--seed N` | from the new Series UID | Random seed of the `air` and `soft-tissue` fills |
| `--keep-body`, `--keep-skin` | off | Keep the Body/External or Skin/Haut contours |
| `--keep-rtdose`, `--keep-rtplan` | off | Copy RTDOSE, or RTPLAN and RTRECORD files |
| `--remove-roi REGEX` | | Also remove ROIs whose whole name matches (case-insensitive, repeatable) |
| `--qc-dir DIR` | `<output>_qc` | Confidential folder for face masks, removed contours, reports and logs |
| `--no-pii-check` | off (check runs) | Skips the metadata check |
| `--allow-empty-mask` | off | Accepts images in which no face was found instead of failing them |
| `--dry-run` | off | Shows what would be done; writes no data |
| `--depth N` (batch) | `1` | Folder depth of the patient folders below the batch folder |
| `--only TEXT` (batch) | | Only patients whose relative path contains TEXT (repeatable) |
| `--ct-glob "<pattern>"` (batch) | see above | File pattern of NIfTI/NRRD CTs in patient folders |

### Extension to the anterior image edge (default)

Objects lying *in front of* the face are not part of the face mask, and some of them are moulded to the face, typically the **thermoplastic immobilisation mask** used in head and neck radiotherapy, or bolus material. A 3D rendering of such an object shows the patient's face even after de-facing.

Therefore the (dilated) face mask is extended along the patient's anterior direction up to the image edge: every voxel in front of the most anterior face-mask voxel of each line is replaced as well; nothing behind the face changes. The anterior direction comes from the image orientation, so supine and prone scans are both handled (for prone scans, the couch lies in front of the patient and is removed as well). `--no-extend-anterior` turns this off and replaces only the dilated face mask.

### Fill of the de-faced region

- **`air`** (default): air sampled from the same scan. Noise with the in-plane texture (noise power spectrum) of the scan's air is mapped onto the scan's own distribution of air values, including any clipping (e.g. at −1024 HU). The air is measured outside the body, within 5 cm of the de-faced slices (or in the whole scan, if there is too little air there).
- **`soft-tissue`**: meant to reduce the difference between de-faced training data and clinical CTs (which still show a face) for segmentation models. In every axial slice, the convex hull of the head that remains after de-facing forms a smooth new front of the head without nose, lips or eyes. De-faced voxels inside the hull get soft tissue (median HU of the nearby soft tissue, with the noise level and texture of the scan, and a 2 mm skin layer); voxels outside the hull get air as above. Cannot be combined with `--no-extend-anterior` (concave parts of the face, e.g. around the eyes, would otherwise leave a face-shaped tissue/air boundary inside the hull).
- **`constant`**: −1000 HU everywhere in the region (the behaviour of earlier versions). **Not recommended for shared data:** real air is noisy, so selecting the voxels that are exactly −1000 HU shows the outline of the de-faced region, i.e. the shape of the face mask including the nose; filling that selection with tissue values makes the face profile visible again.

For `air` and `soft-tissue`, everything the fill uses is measured *outside* the de-faced region, plus a random seed, so the fill cannot add information about the face, and the de-faced region cannot be found by selecting a value. Voxels outside the reconstructed field of view (padding below −1100 HU, e.g. −3024 HU) keep their value. The seed and the measured values are recorded in the manifest, and the seed also in the DICOM header (`DeidentificationMethod`), so the result is reproducible.

Image content of structures inside the de-faced region (e.g. lips, eyes, lenses) is replaced by every fill. Use the [ROI overlap report](#roi-overlap-report) to exclude those structures from training or evaluation.

### Verification of the written DICOM

After writing, the de-faced series is read back with a second, independent reader (slices sorted by their position) and compared with the input. Every voxel outside the de-faced region must be unchanged, and every voxel inside it must hold the intended fill value. If not, the patient fails and no output is kept. This catches slice-order or orientation errors and pixel encodings that cannot store the fill value (e.g. unsigned pixel data without a rescale intercept cannot store air at −1000 HU).

The volume that TotalSegmentator segments is the same volume that is written back (read once with SimpleITK), so the face mask always lies on exactly the grid of the output.

### Metadata check for patient-identifying data

By default every patient is checked, and a warning is shown when the metadata may identify the patient:

- **DICOM:** populated identifying tags such as `PatientName`, `PatientID`, `PatientBirthDate`, `AccessionNumber`, institution and physician names, and every person-name element (also inside sequences). Dates, descriptions and other indirectly identifying tags, and private (vendor) tags, which are not inspected, are reported as well. A patient name/ID equal to the patient folder name (a pseudonym) is not flagged.
- **NIfTI:** free-text header fields (`descrip`, `aux_file`) and identifying keys in a JSON sidecar with the same name.
- **NRRD:** header keys that look identifying (e.g. stored DICOM patient tags).

The console and the log show tag *names* only. The report `identifying_tags_*.csv` in the QC folder also lists the values, so it is as confidential as the data. Nothing is changed; `--no-pii-check` skips the check.

### ROI overlap report

For every kept ROI, `roi_overlap*.csv` in the QC folder lists the percentage of its volume inside the de-faced region and its smallest distance to that region (capped at 50 mm). The final summary lists the ROIs that overlap the region and how many targets (CTV/GTV/PTV) lie within 10 mm of it.

---

## Output

```
results/patient01/                    shareable (after de-identifying the metadata)
├── dcm/                              de-faced CT (several CT series: dcm/series_1/, dcm/series_2/, ...)
└── rt/                               RTSTRUCT without Body/Skin (+ RTPLAN/RTDOSE if kept)

results/patient01_qc/                 CONFIDENTIAL, never share together with the output
├── face_mask.nii.gz                  raw TotalSegmentator face mask (it shows the face)
├── CT_defaced.nii.gz                 de-faced CT as NIfTI, for viewing
├── removed_rois/REMOVED_RS....dcm    removed contours, linked to the de-faced CT
├── roi_overlap.csv, manifest.json, tool.log
└── patients_*.csv, identifying_tags_*.csv, roi_overlap_*.csv   reports
```

In batch mode, the QC folder holds one such folder per patient plus `_reports/` with the reports and the run log of all patients. For NIfTI/NRRD input, the output folder contains `<name>_deFaced.nii.gz` or `<name>_deFaced.nrrd`.

File names that contain the old SOP Instance UID (e.g. `CT.<uid>.dcm`) are renamed to the new UID.

---

## Important notes

- **DICOM metadata is not anonymised.** Run the output through your institution's DICOM de-identification software before sharing. Use a tool that remaps UIDs consistently across all files of a patient, otherwise the links between CT and structures break.
- **Do not share the QC folder.** The face mask and the removed Body/Skin contours show the face.
- **Visual QC is strongly recommended**, especially at the start, for unusual scan geometries, and with immobilisation masks: open the de-faced DICOM in a viewer (e.g. 3D Slicer) and look at a volume rendering (*Volume Rendering* module, CT preset), including low thresholds.
- **The skull remains.** De-facing removes facial soft tissue; bony structures are kept.
- **DICOM output gets new UIDs.** Each de-faced file gets a new `SOPInstanceUID`, each de-faced series a new `SeriesInstanceUID`, and copied RT objects new UIDs as well, with every reference between them updated. `StudyInstanceUID` and `FrameOfReferenceUID` are left unchanged.
- The de-faced files record the de-facing in `DeidentificationMethod`, `DeidentificationMethodCodeSequence` (DICOM code 113102, *Clean Recognizable Visual Features Option*) and `ImageComments`.
- Output pixel/voxel data types match the input (e.g. `int16` stays `int16`). Compressed DICOM input (e.g. JPEG2000) is decompressed; the output is always uncompressed, and can therefore be larger than the input.

### Changes compared with earlier versions

- The de-faced region is filled with air sampled from the scan instead of a constant −1000 HU (`--fill constant` restores the old behaviour), and it is extended to the anterior image edge by default (`--no-extend-anterior` restores the old behaviour).
- RT objects are handled: Body/External and Skin contours are removed from RTSTRUCTs; RTDOSE and RTPLAN/RTRECORD are no longer copied by default.
- `face_seg/` and `deFaced.nii.gz` are no longer written into the output folder; they are in the QC folder (`face_mask.nii.gz`, `CT_defaced.nii.gz`).
- A DICOM input folder is searched recursively, and *every* CT series is de-faced (previously only the largest one). Localizers/scouts are recognised by `ImageType` `LOCALIZER`, `TOPOGRAM`, `SCOUT` or `SURVEY` and are not copied.
- The written DICOM is verified; dcm2niix is no longer needed.
- Batch mode resumes interrupted runs and writes reports; the single-patient mode refuses to write into a non-empty output folder. Outputs made by an earlier version are detected when resuming.

---

## Troubleshooting

**"License number has not been set"**  
Run `totalseg_set_license -l <YOUR_LICENSE_KEY>` with the key from your registration email.

**"No patients found"** (batch mode)  
Check `--depth` (patients grouped in sub-folders need `--depth 2`) and `--only`. For NIfTI/NRRD files with other names, use `--ct-glob`.

**"the pixel encoding cannot store the fill values"** / **"verification: ... cannot store it"**  
The pixel encoding of the input cannot store air (e.g. unsigned data without a negative rescale intercept). Convert the series to signed pixel data or a suitable rescale intercept first. The message lists the encoding (BitsStored, sign, slope, intercept) and examples of intended vs. written values. A few extreme fill values that a slice cannot store (e.g. when slices use different rescale intercepts) are set to the nearest storable value automatically; only if more than 5% are affected does the patient fail.

**"face mask is EMPTY"**  
TotalSegmentator found no face, e.g. in a CT of the neck or chest only. If the CT really shows no face, use `--allow-empty-mask`.

**"output exists but has no manifest"** (batch mode)  
The patient's output was written by an earlier version of this tool or with another `--qc-dir`. Delete that output folder and run again.
