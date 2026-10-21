# Onboarding a partner site

A checklist for the first de-facing run at a new hospital. The details are in the [README](README.md); this page gives the order. Plan about half a day, most of it for visual checks.

## 1. Install (once per computer)

Follow [Setup](README.md#setup-do-this-once-per-computer): conda environment from `environment.yml` (exact package versions from `requirements-lock.txt`), then the TotalSegmentator license. Do not install other package versions: the face masks should be the same at every site.

If the computer has no internet access during the runs, run step 3 once while it has: TotalSegmentator downloads its models (`face`, and `total` with `--fastest` for `--allow-empty-mask`) on first use.

## 2. Prepare the data

- One folder per patient (DICOM: any structure below it; or one NIfTI/NRRD file). See [Supported folder layouts](README.md#supported-folder-layouts).
- **Folder and file names are kept in the output** so that each case can be traced back and discussed with you. If they contain names or patient IDs, replace them when you de-identify the metadata (step 6), or use pseudonyms (e.g. `SITE_001`) from the start. If `PatientName`/`PatientID` equal the folder name, the tool counts them as pseudonyms and says so in the summary.
- Keep the input read-only; the tool never changes it.

## 3. Check one real scan (site validation)

Run the test on one typical head CT of your site (a DICOM folder or NIfTI/NRRD file):

```
conda activate totalseg-deface
pip install pytest
DEFACE_TEST_CT=/data/SITE_001 python -m pytest tests/test_totalsegmentator.py -v -s
```

(Windows: `set DEFACE_TEST_CT=D:\data\SITE_001` first, then `python -m pytest tests/test_totalsegmentator.py -v -s`.)

All three tests must pass. Open the QC picture whose folder the test prints (`QC_defacing.png`): the bottom row must show no nose, lips or eyes. Repeat for every scanner or protocol (e.g. head-and-neck planning CT with thermoplastic mask, diagnostic head CT, cone-beam CT if you share it).

## 4. Dry run of the cohort

```
python deface.py --batch /data/patients -o /data/defaced --dry-run
```

Writes no image data. Check the log and `/data/defaced_qc/_reports/`:
- the number of patients and CT series found (`--depth 2` if patients are grouped in sub-folders);
- files that are not copied, and why (localizers, MR, RTDOSE, screenshots, files without DICOM header, ...);
- contours that would be removed, and **kept contours whose names suggest the skin, a mask or a bolus**: remove those that follow the face with `--remove-roi NAME`;
- the identifying tags (`identifying_tags_*.csv` is confidential).

## 5. Run, then check

```
python deface.py --batch /data/patients -o /data/defaced
```

An interrupted run can simply be started again. Afterwards:
- every `FAILED` patient: see [Troubleshooting](README.md#troubleshooting);
- every `CHECK` line in the summary (see [QC picture and automatic checks](README.md#qc-picture-and-automatic-checks));
- **look at all `QC_defacing*.png`** in `/data/defaced_qc/` (an image viewer that shows a folder of pictures is enough), and for a few patients at a 3D volume rendering of the de-faced DICOM;
- `QC_roi_overlap*.png` and `roi_overlap*.csv` show which contours lost image content (e.g. eyes, lenses, lips): useful when selecting data for training or evaluation.

## 6. De-identify and share

- Run `/data/defaced` through your DICOM de-identification software (it must keep the links between CT and structures by remapping UIDs consistently), and replace identifying folder/file names.
- Share **only** the de-identified output. **Never share the QC folder** `/data/defaced_qc`: it holds the face masks, the removed contours, the QC pictures (with the face) and the identifying tags.
- Keep the QC folder at your site: it lets both of us trace any shared case back to the patient.

## 7. Tell us

Send the coordinating centre the tool version (`tool_commit` in any `manifest.json`), the options you used, and the number of patients done/failed/flagged. Report any QC picture in which the face is still visible.
