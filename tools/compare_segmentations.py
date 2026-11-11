#!/usr/bin/env python3
"""
How much does a downstream segmentation model change when it sees de-faced CTs?

Run the model (e.g. a lymph node CTV model) on the original CT and on each de-faced
version, then compare. The main endpoint needs no ground truth: the prediction on a
de-faced CT against the prediction on the original CT (consistency; 1 = defacing changes
nothing). With clinical contours, both are also compared with the ground truth.

Input: a CSV with one row per patient and the columns
  case        a pseudonym (copied to the output)
  reference   the model's prediction on the ORIGINAL CT (NIfTI/NRRD)
  gt          optional: ground truth mask (e.g. the clinical CTVn, as NIfTI/NRRD)
  mask        optional: face mask of the run (face_mask*.nii.gz); the de-faced region is
              rebuilt from it (3 mm margin and extension to the anterior image edge, as the
              defaults), to report how much of the change lies far from the region
  <any other> one column per de-faced version, e.g. air, soft-tissue, template: the
              model's prediction on that CT
All masks must describe the same patient position (predictions on the CT_defaced*.nii.gz
of the runs share the grid of the original). Other grids are resampled (nearest neighbour).
Relative paths are relative to the CSV file.

    python tools/compare_segmentations.py preds.csv --out results/seg [--label 1]
    -> seg_cases.csv, seg_summary.json, printed table

Metrics: Dice, 95th percentile Hausdorff distance (HD95, mm), surface Dice at 2 mm,
volume change; with `mask`, the changed volume and the share of it more than 10 mm from the
de-faced region (a model that changes far from the face reacts to the context, not to
removed tissue). Statistics: median [IQR], Wilcoxon signed-rank test of Dice to the ground
truth (each version against the original), Friedman test across versions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from defacing.masks import applied_mask_sitk  # noqa: E402

RESERVED = {"case", "reference", "gt", "mask"}
TOLERANCE_MM = 2.0
FAR_MM = 10.0
MARGIN_MM = 3.0


def read_mask(path: str, reference: sitk.Image | None, label: int | None) -> tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(path)
    if reference is not None and (img.GetSize() != reference.GetSize()
                                  or not np.allclose(img.GetSpacing(), reference.GetSpacing())
                                  or not np.allclose(img.GetOrigin(), reference.GetOrigin(), atol=1e-3)
                                  or not np.allclose(img.GetDirection(), reference.GetDirection(), atol=1e-4)):
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(reference)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        img = r.Execute(img)
    a = sitk.GetArrayFromImage(img)
    return (a == label if label is not None else a > 0), img


def _surface(m: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return m & ~binary_erosion(m)


def surface_distances(a: np.ndarray, b: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from the surface of a to the surface of b, and back."""
    from scipy.ndimage import distance_transform_edt
    sa, sb = _surface(a), _surface(b)
    return (distance_transform_edt(~sb, sampling=spacing)[sa], distance_transform_edt(~sa, sampling=spacing)[sb])


def metrics(pred: np.ndarray, ref: np.ndarray, spacing) -> dict:
    voxel_ml = float(np.prod(spacing)) / 1000.0
    vp, vr = int(pred.sum()), int(ref.sum())
    out = {"dice": None, "hd95_mm": None, "surface_dice": None,
           "volume_ml": round(vp * voxel_ml, 2),
           "volume_change_pct": round(100.0 * (vp - vr) / vr, 1) if vr else None}
    if vp == 0 and vr == 0:
        out.update(dice=1.0, surface_dice=1.0, hd95_mm=0.0)
        return out
    out["dice"] = round(2.0 * float((pred & ref).sum()) / (vp + vr), 4)
    if vp == 0 or vr == 0:
        out["surface_dice"] = 0.0
        return out
    d_ab, d_ba = surface_distances(pred, ref, spacing)
    out["hd95_mm"] = round(float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))), 2)
    out["surface_dice"] = round(float(((d_ab <= TOLERANCE_MM).sum() + (d_ba <= TOLERANCE_MM).sum())
                                      / (d_ab.size + d_ba.size)), 4)
    return out


def _median_iqr(values: list) -> list | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    q1, m, q3 = np.percentile(v, [25, 50, 75])
    return [round(float(m), 4), round(float(q1), 4), round(float(q3), 4)]


def main(argv: list[str] | None = None) -> int:
    from scipy.ndimage import distance_transform_edt
    from scipy.stats import friedmanchisquare, wilcoxon

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output prefix, e.g. results/seg")
    ap.add_argument("--label", type=int, help="label value to compare (default: every value > 0)")
    args = ap.parse_args(argv)

    with open(args.cases_csv, newline="", encoding="utf-8-sig") as fh:
        table = list(csv.DictReader(fh))
    if not table or not {"case", "reference"} <= set(table[0]):
        print("the CSV needs the columns case, reference and at least one version column")
        return 2
    base = args.cases_csv.resolve().parent
    for r in table:
        for k, v in r.items():
            if k != "case" and v and not Path(v).is_absolute():
                r[k] = str(base / v)
    versions = [k for k in table[0] if k not in RESERVED]
    has_gt = "gt" in table[0]
    if not versions:
        print("no version column (e.g. air, soft-tissue, template)")
        return 2

    rows = []
    for k, r in enumerate(table, 1):
        print(f"[{k}/{len(table)}] {r['case']}")
        ref, ref_img = read_mask(r["reference"], None, args.label)
        spacing = ref_img.GetSpacing()[::-1]
        voxel_ml = float(np.prod(spacing)) / 1000.0
        gt = read_mask(r["gt"], ref_img, args.label)[0] if has_gt and r.get("gt") else None
        far = None
        if r.get("mask"):
            region = applied_mask_sitk(sitk.ReadImage(r["mask"]), ref_img, 0, True, dilate_by_mm=MARGIN_MM)
            far = distance_transform_edt(~region, sampling=spacing) > FAR_MM
        base = {"case": r["case"]}
        if gt is not None:
            base["original_dice_gt"] = metrics(ref, gt, spacing)["dice"]
        for v in versions:
            if not r.get(v):
                continue
            pred = read_mask(r[v], ref_img, args.label)[0]
            m = metrics(pred, ref, spacing)
            row = {**base, "version": v, **{f"{key}_vs_original": val for key, val in m.items()
                                           if key not in ("volume_ml",)}, "volume_ml": m["volume_ml"]}
            if gt is not None:
                row["dice_gt"] = metrics(pred, gt, spacing)["dice"]
                row["delta_dice_gt"] = (None if row["dice_gt"] is None or base["original_dice_gt"] is None
                                        else round(row["dice_gt"] - base["original_dice_gt"], 4))
            if far is not None:
                changed = pred ^ ref
                row["changed_ml"] = round(float(changed.sum()) * voxel_ml, 2)
                row["changed_far_pct"] = (round(100.0 * float((changed & far).sum()) / float(changed.sum()), 1)
                                          if changed.any() else None)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(f"{args.out}_cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary: dict = {}
    keys = ["dice_vs_original", "hd95_mm_vs_original", "surface_dice_vs_original", "volume_change_pct_vs_original",
            "dice_gt", "delta_dice_gt", "changed_ml", "changed_far_pct"]
    for v in versions:
        rv = [row for row in rows if row["version"] == v]
        s = {"cases": len(rv), **{k: _median_iqr([row.get(k) for row in rv]) for k in keys if any(k in row for row in rv)}}
        if has_gt:
            pairs = [(row["dice_gt"], row["original_dice_gt"]) for row in rv
                     if row.get("dice_gt") is not None and row.get("original_dice_gt") is not None]
            diffs = [a - b for a, b in pairs]
            if len(pairs) >= 6 and any(d != 0 for d in diffs):
                s["wilcoxon_p_dice_gt_vs_original"] = round(float(wilcoxon([a for a, _ in pairs],
                                                                           [b for _, b in pairs]).pvalue), 4)
        summary[v] = s
    if has_gt:
        summary["_original"] = {"dice_gt": _median_iqr([row.get("original_dice_gt") for row in rows
                                                        if row["version"] == versions[0]])}
    complete = [c for c in {row["case"] for row in rows}
                if all(any(row["case"] == c and row["version"] == v and row.get("dice_vs_original") is not None
                           for row in rows) for v in versions)]
    if len(versions) >= 3 and len(complete) >= 6:
        series = [[next(row["dice_vs_original"] for row in rows if row["case"] == c and row["version"] == v)
                   for c in sorted(complete)] for v in versions]
        summary["_friedman_p_dice_vs_original"] = round(float(friedmanchisquare(*series).pvalue), 4)
    Path(f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))

    def fmt(x):
        return "-" if not x else f"{x[0]:.3g} [{x[1]:.3g}-{x[2]:.3g}]"
    print(f"\n{'version':<14}{'n':>4}  {'Dice vs original':<24}{'HD95 mm':<22}{'surface Dice 2mm':<24}"
          + ("Dice to GT (original " + fmt(summary.get('_original', {}).get('dice_gt')) + ")" if has_gt else ""))
    for v in versions:
        s = summary[v]
        print(f"{v:<14}{s['cases']:>4}  {fmt(s.get('dice_vs_original')):<24}{fmt(s.get('hd95_mm_vs_original')):<22}"
              f"{fmt(s.get('surface_dice_vs_original')):<24}"
              + (f"{fmt(s.get('dice_gt'))}  p={s.get('wilcoxon_p_dice_gt_vs_original', '-')}" if has_gt else ""))
    if "_friedman_p_dice_vs_original" in summary:
        print(f"Friedman test across versions (Dice vs original): p={summary['_friedman_p_dice_vs_original']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
