#!/usr/bin/env python3
"""
Cohort-level face identification: can a de-faced CT be assigned to the right patient?

--face-check compares each patient only with themselves. This script asks the cohort
question: every probe (a de-faced CT) is compared with the gallery CT of EVERY patient, and
the probe counts as identified if its own patient's gallery CT is the most similar one
(rank 1). Chance level is 1/N.

Renders and models are those of --face-check (defacing/reid.py): surface renders from the
front and turned 30 degrees to each side, RetinaFace + FaceNet512. The similarity of two
CTs is the highest cosine similarity over all pairs of views in which a face was found (the
attacker tries every view). A probe without any detected face cannot be matched: it counts
as not identified, and its similarities are 0 for the AUC.

--gallery-variant says what the attacker's image looks like:
  same       the gallery CT is rendered like the probes. With the original of the probe CT as
             gallery this is the attacker holding the very same scan: an upper bound.
  perturbed  the gallery CT is resampled to 1 mm, rotated (pitch and yaw 4-8 degrees, roll
             0-3 degrees, random signs, seeded by the case pseudonym), blurred (sigma 0.7 mm)
             and rendered with a surface threshold 100 HU higher from other angles (-45, -15,
             15, 45 degrees), a stand-in for another scan of the patient. Probes never change.

Input: a CSV with one row per patient and the columns
  case          a pseudonym (it is copied to the output; do not use names or patient IDs)
  gallery       the CT the attacker holds: ideally ANOTHER scan of the patient (e.g. the
                original replanning CT); else the original of the probe CT (worst case:
                the attacker has the very same scan, which overestimates the risk)
  gallery_mask  optional: face mask of the gallery CT (its face_mask*.nii.gz); without it,
                the top --crop-mm of the scan are rendered
  mask          face mask of the probe CTs (face_mask*.nii.gz of the run)
  <any other>   one column per probe set, e.g. original, air, soft-tissue, template: the
                CT_defaced*.nii.gz of each run (or the original CT, as a positive control
                when the gallery is a different scan)
Paths are NIfTI/NRRD files or DICOM folders holding one series (the largest is used);
relative paths are relative to the CSV file.

    python tools/face_identification.py cases.csv --out results/face_id
    -> face_id_cases.csv (one row per case and probe set), face_id_summary.json (rank-1 and
       rank-5 rate, mean reciprocal rank, CMC curve, AUC, exact McNemar test against the probe
       set "air"), face_id_scores.json (genuine and impostor similarities, sorted, without
       case names), printed table

The outputs hold scores only. --cache keeps the face embeddings for re-runs: embeddings are
biometric data, so keep the cache in the confidential QC area and delete it afterwards.
Needs the optional packages: pip install -r requirements-reid.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from defacing.masks import applied_mask_sitk, sitk_axes  # noqa: E402

RESERVED = {"case", "gallery", "gallery_mask", "mask"}
MATCH_THRESHOLD = 0.30        # FaceNet512 cosine distance of DeepFace, as --face-check-threshold
REFERENCE_PROBE = "air"       # the McNemar tests compare every other probe set with this one
GALLERY_VARIANTS = ("same", "perturbed")
PERTURB = {"spacing_mm": 1.0, "pitch_yaw_deg": (4.0, 8.0), "roll_deg": (0.0, 3.0), "blur_sigma_mm": 0.7,
           "skin_hu_offset": 100.0, "yaws": (-45, -15, 15, 45)}


def read_image(path: Path) -> sitk.Image:
    """A NIfTI/NRRD file, or the largest series of a DICOM folder."""
    if path.is_file():
        return sitk.ReadImage(str(path))
    reader = sitk.ImageSeriesReader()
    series = [reader.GetGDCMSeriesFileNames(str(path), s) for s in reader.GetGDCMSeriesIDs(str(path))]
    if not series:
        raise ValueError(f"no image file or DICOM series: {path}")
    reader.SetFileNames(max(series, key=len))
    return reader.Execute()


def top_region(hu: np.ndarray, axes, crop_mm: float) -> np.ndarray:
    """Without a face mask: mark the top crop_mm of the body, so that only the head is rendered."""
    si = axes.si_axis
    other = tuple(i for i in range(3) if i != si)
    has_body = np.flatnonzero((hu > -300).any(axis=other))
    region = np.zeros(hu.shape, bool)
    if has_body.size == 0:
        return region
    n = int(math.ceil(crop_mm / axes.spacing[si]))
    top = has_body[-1] if axes.superior else has_body[0]
    lo, hi = (max(0, top - n + 1), top + 1) if axes.superior else (top, min(hu.shape[si], top + n))
    index = [slice(None)] * 3
    index[si] = slice(lo, hi)
    region[tuple(index)] = True
    return region


def perturbation_angles(case: str) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a case; the same pseudonym always gives the same angles."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(case.encode("utf-8")).digest()[:8], "big"))
    pitch, yaw = rng.uniform(*PERTURB["pitch_yaw_deg"], 2) * rng.choice((-1.0, 1.0), 2)
    roll = rng.uniform(*PERTURB["roll_deg"]) * rng.choice((-1.0, 1.0))
    return float(pitch), float(yaw), float(roll)


def perturbed_gallery(img: sitk.Image, region: np.ndarray, case: str):
    """(hu, region, axes) of the gallery CT resampled to 1 mm, rotated about the centre of the
    face region (so that the head stays in the image) with air outside, and blurred."""
    from scipy.ndimage import gaussian_filter
    pitch, yaw, roll = perturbation_angles(case)
    centre = np.argwhere(region).mean(axis=0) if region.any() else (np.array(region.shape) - 1) / 2.0
    rotation = sitk.Euler3DTransform()
    rotation.SetCenter(img.TransformContinuousIndexToPhysicalPoint([float(v) for v in centre[::-1]]))
    rotation.SetRotation(math.radians(pitch), math.radians(roll), math.radians(yaw))    # about L-R, A-P, S-I (LPS)
    step = PERTURB["spacing_mm"]
    size = [int(math.ceil((n - 1) * s / step)) + 1 for n, s in zip(img.GetSize(), img.GetSpacing())]
    mask = sitk.GetImageFromArray(region.astype(np.uint8))
    mask.CopyInformation(img)
    grid = (size, rotation)
    ct = sitk.Resample(img, *grid, sitk.sitkLinear, img.GetOrigin(), (step,) * 3, img.GetDirection(), -1000.0,
                       sitk.sitkFloat32)
    moved = sitk.Resample(mask, *grid, sitk.sitkNearestNeighbor, img.GetOrigin(), (step,) * 3, img.GetDirection(),
                          0, sitk.sitkUInt8)
    hu = gaussian_filter(sitk.GetArrayFromImage(ct), PERTURB["blur_sigma_mm"] / step, mode="nearest")
    return hu, sitk.GetArrayFromImage(moved).astype(bool), sitk_axes(ct)


def embeddings(image_path: Path, mask_path: Path | None, crop_mm: float, gallery_variant: str = "same",
               case: str = "") -> list[np.ndarray]:
    """Normalised face embeddings of the views in which a face was found (may be empty)."""
    from defacing.qc import SKIN_HU
    from defacing.reid import detect_face, render_views
    img = read_image(image_path)
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    axes = sitk_axes(img)
    if mask_path is not None:
        region = applied_mask_sitk(sitk.ReadImage(str(mask_path)), img, 0, False)
    else:
        region = top_region(hu, axes, crop_mm)
    if gallery_variant == "perturbed":
        hu, region, axes = perturbed_gallery(img, region, case)
        views = render_views(hu, region, axes, skin_hu=SKIN_HU + PERTURB["skin_hu_offset"], yaws=PERTURB["yaws"])
    else:
        views = render_views(hu, region, axes)
    out = []
    for view in views.values():
        face = detect_face(view)
        if face is not None:
            e = np.asarray(face["embedding"], float)
            out.append(e / np.linalg.norm(e))
    return out


def similarity(a: list[np.ndarray], b: list[np.ndarray]) -> float | None:
    """Highest cosine similarity over all view pairs; None if either has no face."""
    if not a or not b:
        return None
    return float(max(float(x @ y) for x in a for y in b))


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    """95% Wilson interval of a proportion."""
    if n == 0:
        return None, None
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def auc(genuine: list[float], impostor: list[float]) -> float | None:
    """Probability that a genuine pair scores higher than an impostor pair (ties count 1/2)."""
    if not genuine or not impostor:
        return None
    g, i = np.asarray(genuine)[:, None], np.asarray(impostor)[None, :]
    return round(float((g > i).mean() + 0.5 * (g == i).mean()), 3)


def mcnemar_exact(reference: dict[str, bool], other: dict[str, bool]) -> dict:
    """Exact two-sided McNemar test of paired yes/no results (same cases in both)."""
    only_reference = sum(1 for c in reference if reference[c] and not other[c])
    only_other = sum(1 for c in reference if other[c] and not reference[c])
    n = only_reference + only_other
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(only_reference, only_other) + 1)) / 2 ** n) if n else 1.0
    return {"only_reference": only_reference, "only_this": only_other, "p": round(p, 6)}


def cache_key(path: str, mask: str | None, crop_mm: float, variant: str = "same", case: str = "") -> str:
    """Embeddings depend on the files, the crop and, for a perturbed gallery, the case and all parameters."""
    how = "same" if variant == "same" else f"{variant}|{case}|{json.dumps(PERTURB, sort_keys=True)}"
    return f"{Path(path).resolve()}|{Path(mask).resolve() if mask else ''}|{crop_mm}|{how}"


def evaluate(gallery: dict[str, list], probes: dict[str, list], threshold: float) -> tuple[list[dict], dict, dict]:
    """Per-case rows, the summary and the similarities (sorted, without case names) of one probe
    set. Cases whose gallery shows no face are left out."""
    cases = [c for c in gallery if gallery[c]]
    n = len(cases)
    rows, genuine_all, impostor_all, ranks = [], [], [], []
    hits1 = hits5 = detected = matched = false_matches = impostor_pairs = 0
    sim_to = 1.0 - threshold
    for c in cases:
        sims = {g: similarity(probes.get(c, []), gallery[g]) for g in cases}
        own = sims[c]
        others = [s for g, s in sims.items() if g != c]
        detected += bool(probes.get(c))
        rank = None
        if own is not None:
            rank = 1 + sum(1 for s in others if s is not None and s >= own)
            hits1 += rank == 1
            hits5 += rank <= 5
            matched += own >= sim_to
        false_matches += sum(1 for s in others if s is not None and s >= sim_to)
        impostor_pairs += len(others)
        ranks.append(rank)
        genuine_all.append(own if own is not None else 0.0)
        impostor_all += [s if s is not None else 0.0 for s in others]
        best_other = max((s for s in others if s is not None), default=None)
        rows.append({"case": c, "probe_views_with_face": len(probes.get(c, [])),
                     "gallery_views_with_face": len(gallery[c]),
                     "genuine_similarity": None if own is None else round(own, 3),
                     "best_impostor_similarity": None if best_other is None else round(best_other, 3),
                     "rank": rank})
    summary = {
        "cases": n, "chance_rank1": round(1 / n, 3) if n else None,
        "probe_face_rate": round(detected / n, 3) if n else None,
        "rank1_rate": round(hits1 / n, 3) if n else None, "rank1_ci95": wilson(hits1, n),
        "rank5_rate": round(hits5 / n, 3) if n else None,
        "rank1_hits": hits1,
        "mrr": round(sum(1 / r for r in ranks if r is not None) / n, 3) if n else None,
        "cmc": [round(sum(r is not None and r <= k for r in ranks) / n, 4) for k in range(1, n + 1)],
        "auc": auc(genuine_all, impostor_all),
        "match_rate": round(matched / n, 3) if n else None, "match_ci95": wilson(matched, n),
        "false_match_rate": round(false_matches / impostor_pairs, 3) if impostor_pairs else None,
        "median_genuine_similarity": round(float(np.median(genuine_all)), 3) if genuine_all else None,
        "median_impostor_similarity": round(float(np.median(impostor_all)), 3) if impostor_all else None,
        "threshold_distance": threshold,
    }
    scores = {"genuine": sorted(round(float(s), 3) for s in genuine_all),
              "impostor": sorted(round(float(s), 3) for s in impostor_all)}
    return rows, summary, scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output prefix, e.g. results/face_id")
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                    help=f"cosine distance below which two faces count as the same (default {MATCH_THRESHOLD})")
    ap.add_argument("--crop-mm", type=float, default=260.0,
                    help="without gallery_mask: render the top MM of the body (default 260)")
    ap.add_argument("--cache", type=Path, help="JSON file to keep embeddings between runs (confidential)")
    ap.add_argument("--gallery-variant", choices=GALLERY_VARIANTS, default="same",
                    help="same: render the gallery like the probes (upper bound if it is the original of the probe); "
                         "perturbed: resample, rotate, blur and render it differently (default same)")
    args = ap.parse_args(argv)

    from defacing.reid import check_face_check
    problem = check_face_check()
    if problem:
        print(problem)
        return 1

    with open(args.cases_csv, newline="", encoding="utf-8-sig") as fh:
        table = list(csv.DictReader(fh))
    if not table or not {"case", "gallery", "mask"} <= set(table[0]):
        print("the CSV needs the columns case, gallery, mask and at least one probe column")
        return 2
    probe_sets = [k for k in table[0] if k not in RESERVED]
    if not probe_sets:
        print("no probe column (e.g. air, soft-tissue, template)")
        return 2
    if len({r["case"] for r in table}) != len(table):
        print("case pseudonyms must be unique")
        return 2

    base = args.cases_csv.resolve().parent
    for r in table:
        for k, v in r.items():
            if k != "case" and v and not Path(v).is_absolute():
                r[k] = str(base / v)
    cache: dict[str, list] = {}
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache and args.cache.exists():
        cache = json.loads(args.cache.read_text())

    def emb(path: str, mask: str | None, case: str, variant: str = "same") -> list[np.ndarray]:
        key = cache_key(path, mask, args.crop_mm, variant, case)
        if key not in cache:
            try:
                cache[key] = [e.tolist() for e in embeddings(Path(path), Path(mask) if mask else None, args.crop_mm,
                                                             variant, case)]
            except Exception as e:                                   # one bad case must not stop the cohort
                print(f"  ! {case}: {type(e).__name__}: {e}")
                cache[key] = []
            if args.cache:
                args.cache.write_text(json.dumps(cache))
        return [np.asarray(v) for v in cache[key]]

    gallery, probes = {}, {p: {} for p in probe_sets}
    for k, r in enumerate(table, 1):
        print(f"[{k}/{len(table)}] {r['case']}")
        gallery[r["case"]] = emb(r["gallery"], r.get("gallery_mask") or None, r["case"], args.gallery_variant)
        for p in probe_sets:
            if r.get(p):
                probes[p][r["case"]] = emb(r[p], r["mask"], r["case"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, scores, identified = [], {}, {}, {}
    for p in probe_sets:
        rows, summaries[p], scores[p] = evaluate(gallery, probes[p], args.threshold)
        all_rows += [{"probe_set": p, **row} for row in rows]
        identified[p] = {row["case"]: row["rank"] == 1 for row in rows}
    if REFERENCE_PROBE in identified:
        for p in probe_sets:
            if p != REFERENCE_PROBE:
                summaries[p]["mcnemar_vs_air"] = mcnemar_exact(identified[REFERENCE_PROBE], identified[p])
    summaries["_gallery"] = {"cases": len(gallery), "variant": args.gallery_variant,
                             "perturbation": PERTURB if args.gallery_variant == "perturbed" else None,
                             "gallery_face_rate": round(sum(bool(v) for v in gallery.values()) / len(gallery), 3)}
    with open(f"{args.out}_cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]) if all_rows else ["probe_set"])
        w.writeheader()
        w.writerows(all_rows)
    Path(f"{args.out}_summary.json").write_text(json.dumps(summaries, indent=2))
    Path(f"{args.out}_scores.json").write_text(json.dumps(scores))

    print(f"\ngallery: a face in {summaries['_gallery']['gallery_face_rate']:.0%} of {len(gallery)} cases "
          "(cases without are left out)")
    print(f"{'probe set':<16}{'n':>4}{'face':>7}{'rank-1':>8}{'95% CI':>14}{'chance':>8}{'MRR':>7}{'AUC':>7}{'match':>7}"
          f"{'FMR':>7}{'p vs air':>10}")
    for p in probe_sets:
        s = summaries[p]
        if not s["cases"]:
            print(f"{p:<16}{0:>4}")
            continue
        ci = f"{s['rank1_ci95'][0]:.2f}-{s['rank1_ci95'][1]:.2f}"
        print(f"{p:<16}{s['cases']:>4}{s['probe_face_rate']:>7.0%}{s['rank1_rate']:>8.0%}{ci:>14}"
              f"{s['chance_rank1']:>8.0%}{s['mrr']:>7.3f}{(s['auc'] or 0):>7.2f}{s['match_rate']:>7.0%}"
              f"{(s['false_match_rate'] or 0):>7.1%}{s.get('mcnemar_vs_air', {}).get('p', ''):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
