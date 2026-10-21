#!/usr/bin/env python
"""
CT de-facing tool: removes facial features from CT images, and from the RT objects that
come with them.

Single patient (DICOM folder, searched recursively, or one .nii/.nii.gz/.nrrd image):
    python deface.py -i <input> -o <output_dir> [--qc-dir <qc_dir>]

Batch (one patient per folder, optionally nested: --depth 2 for <group>/<patient>):
    python deface.py --batch <patients_dir> -o <output_dir> [--depth N] [--ct-glob "*CT*.nrrd"]

What is done by default (see README.md for details):
  * every CT series is de-faced: the TotalSegmentator "face" mask, dilated by 3 mm and extended
    to the anterior image edge, is filled with air sampled from the same scan; new UIDs
  * RTSTRUCTs are copied and linked to the de-faced CT; the Body/External and Skin/Haut
    contours are removed (--keep-body / --keep-skin keep them); unclear names are reported
  * RTDOSE and RTPLAN/RTRECORD are not copied (--keep-rtdose / --keep-rtplan)
  * scouts, MR, PET and other files are not copied
  * the written DICOM is read back and verified against the face mask
  * the metadata is checked for patient-identifying data (--no-pii-check skips it); it is
    NOT anonymised
  * face masks, QC pictures, removed contours and reports go to the QC folder (default <output>_qc),
    which is confidential and must not be shared with the output
"""
import argparse
import re
import sys
from pathlib import Path

from defacing.common import FILL_MODES, Options
from defacing.run import DEFAULT_CT_GLOBS, run_batch, run_single


def roi_pattern(text: str) -> re.Pattern:
    try:
        return re.compile(text, re.I)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"invalid regular expression {text!r}: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", type=Path,
                       help="Single input: DICOM folder (CT + RT files, searched recursively), "
                            ".nii, .nii.gz or .nrrd")
    group.add_argument("--batch", metavar="PATIENTS_DIR", type=Path,
                       help="Batch mode: folder containing one sub-folder (or image file) per patient")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output directory (shareable files only)")
    p.add_argument("--qc-dir", type=Path,
                   help="Confidential QC/report folder (default: <output>_qc next to the output)")

    seg = p.add_argument_group("de-facing")
    seg.add_argument("-d", "--device", default="cpu", help="cpu, gpu or mps (default: cpu)")
    seg.add_argument("--dilate-mm", type=float, default=3.0, metavar="MM",
                     help="Safety margin around the face mask in millimetres (default: 3)")
    seg.add_argument("--dilate", type=int, default=2, metavar="N",
                     help="The margin in voxels, used only if the voxel size of an image is unknown "
                          "(a warning is shown; default: 2)")
    ext = seg.add_mutually_exclusive_group()
    ext.add_argument("--extend-anterior", dest="extend_anterior", action="store_true", default=True,
                     help="Also remove everything between the face mask and the anterior image edge, e.g. "
                          "a thermoplastic immobilisation mask or bolus in front of the face (default)")
    ext.add_argument("--no-extend-anterior", dest="extend_anterior", action="store_false",
                     help="Remove only the (dilated) face mask")
    seg.add_argument("--fill", choices=FILL_MODES, default="air",
                     help="Value of the de-faced voxels: 'air' (default: air sampled from the same scan), "
                          "'soft-tissue' (face-independent smooth front of the head with scan-matched "
                          "tissue and air) or 'constant' (-1000 HU; reveals the outline of the face mask)")
    seg.add_argument("--seed", type=int, help="Random seed of the air/soft-tissue fill (default: from the new UID)")
    seg.add_argument("--allow-empty-mask", action="store_true",
                     help="Accept CT series where no face was found instead of failing them, but only if "
                          "TotalSegmentator finds no head (skull, brain) in them either")

    rt = p.add_argument_group("RT objects")
    rt.add_argument("--keep-body", action="store_true", help="Keep Body/External contours (they show the face)")
    rt.add_argument("--keep-skin", action="store_true", help="Keep Skin/Haut contours (they show the face)")
    rt.add_argument("--keep-rtplan", action="store_true", help="Copy RTPLAN and RTRECORD files")
    rt.add_argument("--keep-rtdose", action="store_true",
                    help="Copy RTDOSE files (the dose distribution outlines the face)")
    rt.add_argument("--remove-roi", action="append", default=[], type=roi_pattern, metavar="REGEX",
                    help="Also remove ROIs whose whole name matches REGEX (case-insensitive, repeatable)")

    other = p.add_argument_group("other")
    other.add_argument("--no-pii-check", action="store_true",
                       help="Skip the metadata check for patient-identifying data")
    other.add_argument("--dry-run", action="store_true",
                       help="Only show what would be done (inventory, removals, metadata check)")
    other.add_argument("--depth", type=int, default=1,
                       help="Batch: folder depth of the patients below PATIENTS_DIR (default: 1)")
    other.add_argument("--only", action="append", default=[], metavar="TEXT",
                       help="Batch: only patients whose relative path contains TEXT (repeatable)")
    other.add_argument("--ct-glob", default=None,
                       help=f"Batch: glob pattern(s) to find a NIfTI/NRRD CT inside each patient folder "
                            f"(comma-separated, default: {','.join(DEFAULT_CT_GLOBS)})")
    args = p.parse_args()
    if args.fill == "soft-tissue" and not args.extend_anterior:
        p.error("--fill soft-tissue needs the region to reach the anterior image edge; "
                "remove --no-extend-anterior")
    if args.depth < 1:
        p.error("--depth must be 1 or more")
    if args.dilate < 0:
        p.error("--dilate must be 0 or more")
    if not 0 <= args.dilate_mm < 1000:
        p.error("--dilate-mm must be 0 or more (and finite)")

    opts = Options(device=args.device, dilate_mm=args.dilate_mm, dilate=args.dilate,
                   extend_anterior=args.extend_anterior, fill=args.fill, seed=args.seed,
                   keep_body=args.keep_body, keep_skin=args.keep_skin, keep_rtplan=args.keep_rtplan,
                   keep_rtdose=args.keep_rtdose, remove_roi=args.remove_roi, pii_check=not args.no_pii_check, allow_empty_mask=args.allow_empty_mask)
    out = args.output.expanduser().resolve()
    qc = (args.qc_dir or out.with_name(out.name + "_qc")).expanduser().resolve()
    if qc == out or out in qc.parents:
        sys.exit("The QC folder must not lie inside the output folder (it holds the face masks).")

    if args.input:
        return run_single(args.input.expanduser().resolve(), out, qc, opts, args.dry_run)
    ct_globs = [g.strip() for g in args.ct_glob.split(",")] if args.ct_glob else DEFAULT_CT_GLOBS
    return run_batch(args.batch.expanduser().resolve(), out, qc, opts, args.depth, args.only, ct_globs,
                     args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
