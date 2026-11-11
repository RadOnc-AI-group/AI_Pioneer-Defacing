#!/usr/bin/env python3
"""
Numbers for a report or abstract from the batch reports of one or more runs.

Give each run as NAME=REPORTS_FOLDER (the `_reports` folder in the QC folder of a batch
run, e.g. air=results_air_qc/_reports). Of every report kind, all files in the folder are
read (resumed runs write several); a patient counts once (its last row).

    python tools/summarize_runs.py air=results_air_qc/_reports soft-tissue=results_st_qc/_reports \\
        --out results/runs

Writes <out>_oar_overlap.csv (structure groups, from the first run: the de-faced region is the
same for every fill) and <out>_summary.json, and prints:
  - status and runtime per run (median [IQR] of the pipeline, without the optional checks)
  - face mask volume, kept target volume and exposed target area
  - per structure group: patients with the structure, how many overlap the de-faced region,
    median [IQR] and maximum of the percentage of its volume inside the region
  - face check results, if the runs used --face-check
Structure groups are matched by name, case-insensitively (English and German names); edit
GROUPS for your site. Unmatched names are listed so that none is missed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GROUPS = [   # (group, regex searched in the ROI name); the first match wins
    ("Lips", r"lip|lippe"),
    ("Lens", r"lens|linse"),
    ("Eye / globe", r"eye|globe|bulbus|auge|orbit"),
    ("Optic nerve", r"optic|opticus|n_?ii\b"),
    ("Chiasm", r"chiasm"),
    ("Lacrimal gland", r"lacrim|tr(ae|ä)nen"),
    ("Oral cavity", r"oral|mundh|cavum|cavity_oral"),
    ("Mandible", r"mandib|unterkiefer"),
    ("Parotid", r"parot"),
    ("Submandibular gland", r"submand"),
    ("Brain / brainstem", r"brain|hirn"),
    ("Spinal cord", r"cord|myelon|rueckenmark|rückenmark|medulla"),
    ("Cochlea", r"cochlea"),
    ("Larynx / pharynx", r"laryn|pharyn|constrictor|glottis"),
    ("Teeth", r"teeth|tooth|z(ae|ä)hne|dental"),
    ("Nasal cavity / sinus", r"nas|sinus|nase"),
]
TARGET = re.compile(r"^(ptv|ctv|gtv|itv|law|zv)", re.I)


def read_rows(folder: Path, prefix: str) -> list[dict]:
    rows = []
    for f in sorted(folder.glob(f"{prefix}_*.csv")):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def last_per_patient(rows: list[dict]) -> list[dict]:
    seen = {}
    for r in rows:
        seen[r.get("patient")] = r
    return list(seen.values())


def num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def miqr(values) -> list | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    q1, m, q3 = np.percentile(v, [25, 50, 75])
    return [round(float(m), 2), round(float(q1), 2), round(float(q3), 2)]


def group_of(name: str, rtype: str, is_target: str) -> str | None:
    if str(is_target).lower() in ("1", "true", "yes") or TARGET.match(name) or rtype.upper() in ("PTV", "CTV", "GTV"):
        m = TARGET.match(name)
        return f"Target {m.group(1).upper()}" if m else f"Target {rtype.upper()}"
    for g, rx in GROUPS:
        if re.search(rx, name, re.I):
            return g
    return None


def fmt(x) -> str:
    return "-" if not x else f"{x[0]:g} [{x[1]:g}-{x[2]:g}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="NAME=REPORTS_FOLDER")
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    args = ap.parse_args(argv)
    runs = []
    for item in args.runs:
        name, _, folder = item.partition("=")
        if not folder:
            ap.error(f"give runs as NAME=FOLDER, not {item}")
        runs.append((name, Path(folder)))

    summary: dict = {}
    print(f"{'run':<14}{'done':>6}{'failed':>8}  {'runtime s (median [IQR])':<28}{'face mask ml':<22}"
          f"{'target kept ml':<20}{'exposed cm2'}")
    for name, folder in runs:
        patients = last_per_patient(read_rows(folder, "patients"))
        timing = [r for r in last_per_patient(read_rows(folder, "timing")) if r.get("status") == "done"]
        status = Counter(r.get("status") for r in patients)
        optional = [k for k in (timing[0] if timing else {}) if k in ("anatomy_check_s", "face_check_s")]
        pipeline = [num(r.get("total_s")) - sum(num(r.get(k)) or 0 for k in optional)
                    for r in timing if num(r.get("total_s")) is not None]
        s = {"patients": len(patients), "status": dict(status),
             "runtime_s": miqr(pipeline), "devices": dict(Counter(r.get("device") for r in timing)),
             "face_mask_ml": miqr(num(r.get("face_mask_ml")) for r in patients),
             "target_kept_ml": miqr(num(r.get("target_kept_ml")) for r in patients),
             "target_exposed_cm2": miqr(num(r.get("target_exposed_cm2")) for r in patients),
             "fill_used": dict(Counter(r.get("fill_used") for r in patients))}
        fc = Counter(r.get("face_check") for r in patients if r.get("face_check"))
        if fc:
            s["face_check"] = dict(fc)
            s["face_check_score"] = miqr(num(r.get("face_check_score")) for r in patients)
        summary[name] = s
        print(f"{name:<14}{status.get('done', 0):>6}{status.get('failed', 0):>8}  {fmt(s['runtime_s']):<28}"
              f"{fmt(s['face_mask_ml']):<22}{fmt(s['target_kept_ml']):<20}{fmt(s['target_exposed_cm2'])}")
        if fc:
            print(f"{'':<14}face check: " + ", ".join(f"{k} {v}" for k, v in sorted(fc.items())))

    name, folder = runs[0]
    overlap = read_rows(folder, "roi_overlap")
    per_patient: dict[str, dict[str, float]] = defaultdict(dict)   # group -> patient -> highest pct
    ml: dict[str, dict[str, float]] = defaultdict(dict)
    unmatched = Counter()
    for r in overlap:
        g = group_of(r.get("roi", ""), r.get("type", ""), r.get("is_target", ""))
        if g is None:
            unmatched[r.get("roi", "")] += 1
            continue
        p, pct, vol = r.get("patient"), num(r.get("pct_in_defaced_region")) or 0.0, num(r.get("ml_in_defaced_region")) or 0.0
        if pct >= per_patient[g].get(p, -1):
            per_patient[g][p], ml[g][p] = pct, vol
    table = []
    for g in [g for g, _ in GROUPS] + sorted(k for k in per_patient if k.startswith("Target")):
        if g not in per_patient:
            continue
        pcts = list(per_patient[g].values())
        table.append({"group": g, "patients": len(pcts),
                      "patients_overlapping": sum(1 for x in pcts if x > 0),
                      "pct_in_region_median_iqr": miqr(pcts), "pct_in_region_max": round(max(pcts), 1),
                      "ml_in_region_median_iqr": miqr(ml[g].values())})
    with open(f"{args.out}_oar_overlap.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "patients", "patients_overlapping", "pct_median", "pct_q1", "pct_q3", "pct_max",
                    "ml_median", "ml_q1", "ml_q3"])
        for t in table:
            p, m = t["pct_in_region_median_iqr"] or [None] * 3, t["ml_in_region_median_iqr"] or [None] * 3
            w.writerow([t["group"], t["patients"], t["patients_overlapping"], *p, t["pct_in_region_max"], *m])
    summary["_overlap"] = {"run": name, "groups": table, "unmatched_names": dict(unmatched.most_common())}
    Path(f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nStructures and the de-faced region (run {name}; per patient the most affected structure of a group, e.g. of left and right eye)")
    print(f"{'group':<22}{'patients':>9}{'overlap':>9}  {'% in region, median [IQR]':<28}{'max %':>7}")
    for t in table:
        print(f"{t['group']:<22}{t['patients']:>9}{t['patients_overlapping']:>9}  "
              f"{fmt(t['pct_in_region_median_iqr']):<28}{t['pct_in_region_max']:>7}")
    if unmatched:
        print("\nNot in any group (add to GROUPS if relevant): " + ", ".join(list(unmatched)[:30])
              + (" ..." if len(unmatched) > 30 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
