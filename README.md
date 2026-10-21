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

This downloads all required packages automatically, in the exact versions of `requirements-lock.txt`, so that every site runs the same software (and gets the same face masks). It only needs to be done once.

*Without conda:* create a Python 3.10 virtual environment and run `pip install -r requirements-lock.txt`.

New to the tool at your site? Follow the [onboarding guide](ONBOARDING.md).

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

**Try it first:** `--dry-run` shows what would be done (which series are de-faced, which files and contours are removed, the metadata check) without writing any image data (only the reports, in `<qc>/_reports/`); `--only TEXT` restricts the run to patients whose path contains `TEXT`.

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

For NIfTI/NRRD, the tool looks for files matching `*CT*.nrrd`, `*CT*.nii.gz`, `*CT*.nii` (and lower-case `ct`). Use `--ct-glob` for other names, e.g. `--ct-glob "*_CT_*.nii.gz"`. A folder that holds a matching file is treated as a NIfTI/NRRD patient: only the first matching file is de-faced, and a warning names the other files in that folder (e.g. DICOM files, or a `*CTV*.nii.gz` mask that also matches `*CT*`).

---

## What is done by default

| Content | Default | Change with |
|---|---|---|
| CT series (all, except localizers/scouts) | de-faced: face mask dilated by 3 mm, extended to the anterior image edge, filled with air sampled from the same scan; new SOP Instance and Series UIDs | `--dilate-mm`, `--no-extend-anterior`, `--fill` |
| RTSTRUCT | copied; UIDs remapped so it still refers to the de-faced CT | |
| ↳ Body/External contour | **removed** (it traces the skin, i.e. the face) | `--keep-body` |
| ↳ Skin/Haut contour | **removed** | `--keep-skin` |
| ↳ other contours (targets, organs at risk) | kept unchanged; names that might follow the skin are reported | `--remove-roi REGEX` |
| RTDOSE | **not copied** (the dose is zero outside the body, so it outlines the face) | `--keep-rtdose` |
| RTPLAN, RTRECORD | **not copied** | `--keep-rtplan` |
| CT localizers/scouts, MR, PET, REG, RT images, non-DICOM files | not copied (cannot be de-faced); listed in the log | |
| CT objects that are no single-frame CT image volume: Enhanced (multi-frame) CT, secondary captures (screenshots, dose reports), series with fewer than 3 images or with images of different orientations | not copied (cannot be de-faced); listed in the log | |
| CT images marked `BurnedInAnnotation = YES` (text such as the name drawn into the pixels) | not copied; listed in the log | |
| DICOM data without the DICOM file header (128-byte preamble and `DICM`, e.g. from old archives) | not read; a warning asks to convert them first (e.g. `gdcmconv`, `dcmconv`) | |
| De-faced DICOM | read back and verified (see [Verification](#verification-of-the-written-dicom)) | |
| Metadata | checked for patient-identifying data and reported; **not changed** | `--no-pii-check` |

### Body and Skin contours

ROI names are compared case-insensitively, with umlauts written out (`ö` = `oe`, `ß` = `ss`).

- **Removed:** ROIs of type `EXTERNAL`, and ROIs whose *whole* name is one of `Body`, `External`, `Outline`, `Körper`/`Koerper`, `Körperkontur`, `Außenkontur`/`Aussenkontur`, `Patient`, `Patientenkontur` (Body), or `Skin`, `Haut`, `Hautkontur` (Skin), optionally followed by a number or thickness, e.g. `BODY_1`, `BODY (1)`, `Skin 5mm`, `Haut_3mm`.
- **Kept and reported:** ROIs of type `BOLUS` or `FIXATION` (e.g. the thermoplastic mask), and ROIs whose name only *contains* one of these words, or `mask`/`Maske`, `bolus`, `shell`, `thermo`, `fixation`, `immobili…`. Examples: `Körperteile`, `Body-PTV`, `z_Skin_ring`, `Maske`. Many such names are real anatomy and must not be removed automatically, e.g. `Glaskörper` (vitreous body), `Wirbelkörper`, `Vertebral_Body`. Check the reported names; remove those that follow the skin or face with `--remove-roi`, e.g. `--remove-roi "Body-PTV" --remove-roi "z_Skin.*"`.
- A structure set without a recognisable Body contour is reported, so that a skin contour with an unusual name is not overlooked.

The removed contours are saved in the QC folder as a separate RTSTRUCT that refers to the de-faced CT, so they can be checked in a viewer. Plans and records that refer to removed objects keep a dangling reference; this does not matter when they are not shared.

---

## Options

| Option | Default | Description |
|---|---|---|
| `-d cpu` / `gpu` / `mps` | `cpu` | Device for TotalSegmentator |
| `--dilate-mm MM` | `3` | Expands the face mask by MM millimetres in all directions (every voxel whose centre lies within MM of the mask; with slices thicker than MM there is no margin across slices) |
| `--dilate N` | `2` | The margin in voxels, used only for images whose voxel size is unknown (DICOM without `PixelSpacing`, NIfTI without orientation and with 1 mm voxels, NRRD without `space`); reported as a warning |
| `--no-extend-anterior` | off (extension on) | Removes only the dilated face mask, not everything in front of it (see below) |
| `--fill air` / `soft-tissue` / `constant` | `air` | Value of the de-faced voxels (see below) |
| `--seed N` | from the new Series UID | Random seed of the `air` and `soft-tissue` fills |
| `--keep-body`, `--keep-skin` | off | Keep the Body/External or Skin/Haut contours |
| `--keep-rtdose`, `--keep-rtplan` | off | Copy RTDOSE, or RTPLAN and RTRECORD files |
| `--remove-roi REGEX` | | Also remove ROIs whose whole name matches (case-insensitive, repeatable) |
| `--qc-dir DIR` | `<output>_qc` | Confidential folder for face masks, QC pictures, removed contours, reports and logs |
| `--no-pii-check` | off (check runs) | Skips the metadata check |
| `--allow-empty-mask` | off | Accepts CT series in which no face was found, but only if TotalSegmentator's `total` model (fastest, 6 mm, no license needed) finds no head (skull, brain, together at least 20 ml) in them either, e.g. a chest CT next to the head CT of the same patient. A series that includes the head but has no face mask always fails the patient |
| `--dry-run` | off | Shows what would be done; writes no image data, only the reports |
| `--depth N` (batch) | `1` | Folder depth of the patient folders below the batch folder |
| `--only TEXT` (batch) | | Only patients whose relative path contains TEXT (repeatable) |
| `--ct-glob "<pattern>"` (batch) | see above | File pattern of NIfTI/NRRD CTs in patient folders |

### Extension to the anterior image edge (default)

Objects lying *in front of* the face are not part of the face mask, and some of them are moulded to the face, typically the **thermoplastic immobilisation mask** used in head and neck radiotherapy, or bolus material. A 3D rendering of such an object shows the patient's face even after de-facing.

Therefore the (dilated) face mask is extended along the patient's anterior direction up to the image edge: every voxel in front of the most anterior face-mask voxel of each line is replaced as well; nothing behind the face changes. The anterior direction comes from the image orientation, so supine and prone scans are both handled (for prone scans, the couch lies in front of the patient and is removed as well). `--no-extend-anterior` turns this off and replaces only the dilated face mask.

Before extending, the tool checks that the face mask lies at the front of the head: on the lines through the mask there must be more tissue behind it than in front of it. If not, the orientation stored in the header is wrong (e.g. a wrong `PatientPosition`, or a NIfTI/NRRD file with a flipped or missing orientation), and the extension would erase the head instead of what lies in front of the face; the patient then fails and nothing is written.

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
- **NIfTI:** free-text header fields (`descrip`, `aux_file`), header extensions (they can hold a whole DICOM header), and identifying keys in a JSON sidecar with the same name.
- **NRRD:** header keys that look identifying (e.g. stored DICOM patient tags).

A name/ID equal to the folder name only shows that both agree: if the folders are named by the real name or patient ID (e.g. a PACS export by ID), they are *not* pseudonyms, and the output folder names identify the patients as well. The final summary therefore lists how many patients were counted as pseudonyms this way.

The console and the log show tag *names* only. The report `identifying_tags_*.csv` in the QC folder also lists the values, so it is as confidential as the data. Nothing is changed; `--no-pii-check` skips the check.

### QC picture and automatic checks

For every de-faced CT series, `QC_defacing*.png` in the QC folder shows the input (top row) and the de-faced output (bottom row), each as a side view (silhouette of the soft tissue, face to the left) and a front view (the skin surface, shaded). A nose, lips or eyes in the bottom row means the de-facing failed for that series. Scrolling through these pictures is the quickest visual QC of a cohort. They are **confidential** (the top row shows the face) and are never written to the output folder.

These checks are reported as `CHECK` lines in the final summary and (except the face mask size, which needs the whole run) in the notes of `patients_*.csv`; the output is written anyway:
- the face mask has a separate part of at least 1 ml more than 10 mm away from the face (a false detection: everything in front of it is removed as well);
- the face mask is smaller than half the median of the run (from 5 patients on);
- possible burned-in text: very bright pixels outside the body at the same position in most slices of the de-faced output (text such as a name drawn into the images; metal outside the body, e.g. a stereotactic frame, can also cause it);
- the voxel size is unknown, so the margin was given in voxels (`--dilate`).

### ROI overlap report

For every kept ROI, `roi_overlap*.csv` in the QC folder lists:

| Column | Meaning |
|---|---|
| `voxels` | size of the ROI (voxels of the CT) |
| `pct_in_defaced_region` | percentage of the ROI's volume inside the de-faced region (its image content is replaced) |
| `ml_in_defaced_region` | the same as a volume in ml (5 % of a large PTV can be more than 100 % of a lens) |
| `pct_in_face_mask` | percentage inside the face mask itself, i.e. without the margin and the extension to the anterior image edge |
| `min_distance_mm` | smallest distance to the de-faced region (`>=50` for 50 mm or more) |

The final summary lists the ROIs that overlap the region and how many targets (CTV/GTV/PTV) lie within 10 mm of it.

For every structure set with ROIs that overlap the de-faced region or lie within 10 mm of it, `QC_roi_overlap*.png` shows the same side and front views as `QC_defacing*.png` (of the input), the de-faced region in blue, and each of these ROIs in its own colour: the part inside the region filled, the outline of the whole ROI drawn. The legend in the picture names each ROI with its percentage and volume inside the region, or its distance; the most affected ROIs come first (at most 12, the rest are in the CSV). Note that a side or front view is a projection: an outline that crosses the blue area does not mean overlap, a filled part does. Like all QC files, the picture is confidential.

---

## Output

```
results/patient01/                    shareable (after de-identifying the metadata)
├── dcm/                              de-faced CT (several CT series: dcm/series_1/, dcm/series_2/, ...)
└── rt/                               RTSTRUCT without Body/Skin (+ RTPLAN/RTDOSE if kept)

results/patient01_qc/                 CONFIDENTIAL, never share together with the output
├── face_mask.nii.gz                  raw TotalSegmentator face mask (it shows the face)
├── CT_defaced.nii.gz                 de-faced CT as NIfTI, for viewing
├── QC_defacing.png                   QC picture: input and de-faced output, side and front view
├── QC_roi_overlap.png                ROIs in or near the de-faced region (if there are any)
├── removed_rois/REMOVED_RS....dcm    removed contours, linked to the de-faced CT
├── roi_overlap.csv, manifest.json, tool.log
└── patients_*.csv, identifying_tags_*.csv, roi_overlap_*.csv   reports
```

In batch mode, the QC folder holds one such folder per patient plus `_reports/` with the reports and the run log of all patients. For NIfTI/NRRD input, the output folder contains `<name>_deFaced.nii.gz` or `<name>_deFaced.nrrd`.

File and folder names are kept, so every output can be traced back to the patient at the hospital: the output folders are named like the input patient folders, the DICOM files like the input files (file names that contain the old SOP Instance UID, e.g. `CT.<uid>.dcm`, get the new UID), and NIfTI/NRRD outputs like the input files. Names that identify the patient (name, patient ID) must therefore be replaced by the hospital before sharing, like the metadata.

---

## Important notes

- **DICOM metadata is not anonymised.** Run the output through your institution's DICOM de-identification software before sharing. Use a tool that remaps UIDs consistently across all files of a patient, otherwise the links between CT and structures break.
- **Do not share the QC folder.** The face mask, the QC pictures and the removed Body/Skin contours show the face; the reports contain identifying data.
- **Visual QC is strongly recommended**, especially at the start, for unusual scan geometries, and with immobilisation masks: look at every `QC_defacing*.png`, and for a sample of patients open the de-faced DICOM in a viewer (e.g. 3D Slicer) and look at a volume rendering (*Volume Rendering* module, CT preset), including low thresholds.
- **File and folder names are not changed** (see [Output](#output)); replace identifying names when de-identifying the metadata.
- **The skull remains.** De-facing removes facial soft tissue; bony structures are kept.
- **DICOM output gets new UIDs.** Each de-faced file gets a new `SOPInstanceUID`, each de-faced series a new `SeriesInstanceUID`, and copied RT objects new UIDs as well, with every reference between them updated. `StudyInstanceUID` and `FrameOfReferenceUID` are left unchanged.
- Icon images (`IconImageSequence`, a thumbnail of the original slice that would show the face) are removed from the de-faced files.
- The de-faced files record the de-facing in `DeidentificationMethod`, `DeidentificationMethodCodeSequence` (DICOM code 113102, *Clean Recognizable Visual Features Option*) and `ImageComments`.
- Output pixel/voxel data types match the input (e.g. `int16` stays `int16`), and so does the NIfTI scaling (`scl_slope`/`scl_inter`, e.g. `uint16` with `scl_inter = -1024` as written by dcm2niix). Compressed DICOM input (e.g. JPEG2000) is decompressed; the output is always uncompressed, and can therefore be larger than the input.

### Changes compared with earlier versions

- The de-faced region is filled with air sampled from the scan instead of a constant −1000 HU (`--fill constant` restores the old behaviour), and it is extended to the anterior image edge by default (`--no-extend-anterior` restores the old behaviour).
- RT objects are handled: Body/External and Skin contours are removed from RTSTRUCTs; RTDOSE and RTPLAN/RTRECORD are no longer copied by default.
- `face_seg/` and `deFaced.nii.gz` are no longer written into the output folder; they are in the QC folder (`face_mask.nii.gz`, `CT_defaced.nii.gz`).
- A DICOM input folder is searched recursively, and *every* CT series is de-faced (previously only the largest one). Localizers/scouts are recognised by `ImageType` `LOCALIZER`, `TOPOGRAM`, `SCOUT`, `SURVEY` or `SCANOGRAM`, or by `ScanOptions` (e.g. `SURVIEW`, `SCOUT MODE`), and are not copied.
- The written DICOM is verified; dcm2niix is no longer needed.
- Batch mode resumes interrupted runs and writes reports; the single-patient mode refuses to write into a non-empty output folder. Outputs made by an earlier version are detected when resuming.
- Tool version 5: the margin is given in millimetres (`--dilate-mm`, default 3 mm; before: 2 voxels); `--allow-empty-mask` accepts an empty face mask only in series without a head; QC pictures and automatic checks; icon images are removed; secondary captures, Enhanced CT, images with `BurnedInAnnotation = YES` and CT series that are no volume are not copied; DICOM files without file header are reported; a face mask that is not at the front of the head (wrong orientation in the header) fails the patient; NIfTI scaling is kept; the ROI overlap report gives the replaced volume and the share inside the face mask, uses the CT series the RTSTRUCT was drawn on, and comes with `QC_roi_overlap*.png`; exact package versions in `requirements-lock.txt`.

---

## Tests

The automated tests (`tests/`) run the complete tool on synthetic head phantoms (DICOM in six orientations, several slice orders and pixel encodings, compressed files, RT objects, NIfTI and NRRD), with TotalSegmentator replaced by a stand-in that marks the phantom's face. They check, in patient coordinates, that the face and everything in front of it is replaced and nothing else changes, and they cover the RT handling, the metadata check, batch/resume and the command line. They run on every push (GitHub Actions, Linux, Windows and macOS); locally:

```
pip install -r requirements-test.txt
python -m pytest
```

They do not test TotalSegmentator itself. `tests/test_totalsegmentator.py` does, on a real head CT, and is skipped unless one is given; use it as a site check for every new scanner or protocol (see the [onboarding guide](ONBOARDING.md)):

```
DEFACE_TEST_CT=/data/patient01 python -m pytest tests/test_totalsegmentator.py -v -s
```

On GitHub, the workflow `totalsegmentator` runs it on the CPU (weekly, by hand, and when the packages change) once the repository has the secret `TOTALSEG_LICENSE` and the variable `DEFACE_TEST_CT_URL` (a public head CT); see the comments in `.github/workflows/totalsegmentator.yml`.

---

## Troubleshooting

**"License number has not been set"**  
Run `totalseg_set_license -l <YOUR_LICENSE_KEY>` with the key from your registration email.

**"No patients found"** (batch mode)  
Check `--depth` (patients grouped in sub-folders need `--depth 2`) and `--only`. For NIfTI/NRRD files with other names, use `--ct-glob`.

**"the pixel encoding cannot store the fill values"** / **"verification: ... cannot store it"**  
The pixel encoding of the input cannot store air (e.g. unsigned data without a negative rescale intercept). Convert the series to signed pixel data or a suitable rescale intercept first. The message lists the encoding (BitsStored, sign, slope, intercept) and examples of intended vs. written values. A few extreme fill values that a slice cannot store (e.g. when slices use different rescale intercepts) are set to the nearest storable value automatically; only if more than 5% are affected does the patient fail.

**"the face mask does not lie at the anterior side of the head"**  
The orientation in the image header does not match the image: check `PatientPosition` (DICOM) or the orientation of the NIfTI/NRRD file (e.g. in 3D Slicer, the face must point to *Anterior*). Fix the header and run again; `--no-extend-anterior` would only replace the face mask itself.

**"face mask is EMPTY"**  
TotalSegmentator found no face, e.g. in a CT of the chest only. If the CT really shows no face, use `--allow-empty-mask`.

**"face mask is EMPTY ... although the scan includes the head"**  
With `--allow-empty-mask`, TotalSegmentator found skull or brain but no face. Look at the scan: the face may be cut off by the field of view, or the image is unusual (e.g. a cone-beam CT). Such a series is never passed on unchanged.

**"... DICOM data without the DICOM file header ..."**  
Some files (e.g. from old archives) hold DICOM data without the 128-byte preamble and `DICM` marker. Convert them into standard DICOM files, e.g. `gdcmconv input output` or `dcmconv input output` (DCMTK), and run again.

**"output exists but has no manifest"** (batch mode)  
The patient's output was written by an earlier version of this tool or with another `--qc-dir`. Delete that output folder and run again.
