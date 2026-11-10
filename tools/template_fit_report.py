#!/usr/bin/env python3
"""
EXPERIMENTAL: how often can a template be fitted? Nothing is written to any image.

Give each case as a CT and the face mask of an earlier run (the face_mask.nii.gz in its QC
folder). The output holds numbers only, and cases are numbered in the order given, so it
can be shared. skin_p10/p50/p90_mm compare the fitted front with the real skin (positive =
in front of it): a validation of the fit that is never used by it.

    python tools/template_fit_report.py --template flame_template.npz \\
        --case CT_FOLDER_OR_FILE QC_FOLDER/face_mask.nii.gz --case ... --json fit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from defacing.masks import applied_mask_sitk, sitk_axes  # noqa: E402
from defacing.template import fit_template, front_minus_skin, load_template  # noqa: E402


def read_image(path: Path) -> sitk.Image:
    """A NIfTI/NRRD file, or the largest CT series of a DICOM folder."""
    if path.is_file():
        return sitk.ReadImage(str(path))
    reader = sitk.ImageSeriesReader()
    series = [reader.GetGDCMSeriesFileNames(str(path), s) for s in reader.GetGDCMSeriesIDs(str(path))]
    if not series:
        raise ValueError("no DICOM series")
    reader.SetFileNames(max(series, key=len))
    return reader.Execute()


def one_case(template, ct: Path, mask_path: Path, dilate_mm: float) -> dict:
    image = read_image(ct)
    hu = sitk.GetArrayFromImage(image).astype(np.float64)
    mask = sitk.ReadImage(str(mask_path))
    face = applied_mask_sitk(mask, image, 0, False)
    region = applied_mask_sitk(mask, image, 2, True, hu=hu, dilate_by_mm=dilate_mm)
    axes = sitk_axes(image)
    fit = fit_template(template, hu, face, region, axes)
    return {**fit.numbers(), **(front_minus_skin(fit, hu, face, region, axes) or {})}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--case", nargs=2, action="append", required=True, type=Path, metavar=("CT", "FACE_MASK"))
    p.add_argument("--dilate-mm", type=float, default=3.0)
    p.add_argument("--json", type=Path, help="also write the results to this file")
    args = p.parse_args(argv)
    template = load_template(args.template)
    rows = []
    for number, (ct, mask) in enumerate(args.case, 1):
        try:
            row = one_case(template, ct, mask, args.dilate_mm)
        except Exception as e:                      # the message could name a file, so only its type is shown
            row = {"accepted": False, "reason": f"error: {type(e).__name__}"}
        rows.append({"case": number, **row})
        print(json.dumps(rows[-1]), flush=True)
    accepted = sum(r["accepted"] for r in rows)
    print(f"\naccepted {accepted} of {len(rows)}")
    for reason, count in Counter(r["reason"] for r in rows if not r["accepted"]).most_common():
        print(f"  {count} x {reason}")
    if args.json:
        args.json.write_text(json.dumps({"template_sha256": template.sha256, "cases": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
