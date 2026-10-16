"""Single and batch runs: discovery, resume, reports, final summary."""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .common import Options, PatientError, classify_suffix, log, strip_known_suffix
from .patient import (OVERLAP_COLS, SCRIPT_VERSION, TARGET_TYPES, PatientResult, deface_dicom_patient,
                      deface_image_file)
from .segment import check_totalsegmentator

DEFAULT_CT_GLOBS = ["*CT*.nrrd", "*CT*.nii.gz", "*CT*.nii", "*ct*.nrrd", "*ct*.nii.gz"]
REPORTS_DIRNAME = "_reports"


def setup_logging(log_file: Path | None = None) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(log_file, encoding="utf-8")
        to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(to_file)


# ---------------------------------------------------------------------------
# Reports (confidential: they contain identifier values and ROI names)
# ---------------------------------------------------------------------------

class Reports:
    PATIENT_COLS = ["patient", "status", "seconds", "ct_series", "ct_files", "rt_files", "files_not_copied",
                    "rtdose_removed", "rtplan_removed", "rois_removed", "rois_unclear_kept", "face_mask_ml",
                    "fill", "rtstruct_unresolved_refs", "identifying_tags_populated", "notes"]
    ID_COLS = ["patient", "class", "keyword", "pseudonym_matches_folder", "n_distinct_values",
               "values (max 5 shown)"]

    def __init__(self, report_dir: Path, stamp: str, pii_check: bool):
        self.dir, self.stamp, self.pii_check = report_dir, stamp, pii_check
        self.patients_csv = report_dir / f"patients_{stamp}.csv"
        self.ids_csv = report_dir / f"identifying_tags_{stamp}.csv" if pii_check else None
        self.overlap_csv = report_dir / f"roi_overlap_{stamp}.csv"
        self._files: dict[str, tuple] = {}

    def _writer(self, path: Path, header: list[str]):
        if str(path) not in self._files:            # opened on first use
            self.dir.mkdir(parents=True, exist_ok=True)
            fh = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(fh)
            w.writerow(header)
            self._files[str(path)] = (fh, w)
        return self._files[str(path)][1]

    def write(self, r: PatientResult) -> None:
        folder = Path(r.rel).name
        flagged = r.ids.flagged(folder) if r.ids else r.pii
        fills = ";".join(f.get("fill", "") for f in r.fill)
        self._writer(self.patients_csv, self.PATIENT_COLS).writerow([
            r.rel, r.status, f"{r.seconds:.0f}", r.ct_series, r.ct_files, r.rt_files, r.not_copied,
            r.rtdose_removed, r.rtplan_removed, ";".join(r.rois_removed), ";".join(r.rois_unclear),
            ";".join(f"{v:.1f}" for v in r.face_mask_ml), fills, r.unresolved_refs, ";".join(flagged),
            " | ".join(([r.error] if r.error else []) + r.flags),
        ])
        if r.ids and self.ids_csv:
            w = self._writer(self.ids_csv, self.ID_COLS)
            for kw in sorted(r.ids.values, key=lambda k: (r.ids.kinds[k] != "direct", k)):
                values = r.ids.values[kw]
                shown = " | ".join(v[:80] for v, _ in values.most_common(5))
                w.writerow([r.rel, r.ids.kinds[kw], kw, "yes" if r.ids.matches_folder(kw, folder) else "",
                            len(values), shown])
            if r.ids.private_tags:
                w.writerow([r.rel, "private", "(private tags)", "", len(r.ids.private_tags),
                            "vendor-specific tags, not inspected"])
        if r.overlap:
            w = self._writer(self.overlap_csv, OVERLAP_COLS)
            for row in r.overlap:
                w.writerow([row[c] for c in OVERLAP_COLS])
        for fh, _ in self._files.values():
            fh.flush()

    def close(self) -> None:
        for fh, _ in self._files.values():
            fh.close()

    def written(self) -> list[Path]:
        return [Path(p) for p in self._files]


def _is_target(row: dict) -> bool:
    return row["type"] in TARGET_TYPES or row["roi"].upper().startswith(TARGET_TYPES)


def final_summary(results: list[PatientResult], reports: Reports, out_root: Path, qc_root: Path,
                  dry_run: bool, interrupted: bool, opts: Options) -> str:
    status = Counter(r.status for r in results)
    bar = "=" * 78
    title = "DRY RUN FINISHED" if dry_run else ("RUN INTERRUPTED" if interrupted else "DE-FACING FINISHED")
    log.info("")
    log.info(bar)
    log.info(f"{title}: " + ", ".join(f"{n} {s}" for s, n in status.items()))
    for r in results:
        if r.status == "failed":
            log.error(f"  FAILED  {r.rel}: {r.error}")
    legacy = [r for r in results if r.status == "legacy output"]
    if legacy:
        log.error(f"  {len(legacy)} output folder(s) have no manifest in the QC folder: made by an earlier")
        log.error("  version (they may contain face masks, Body contours and RTDOSE) or with another")
        log.error(f"  --qc-dir. Delete them from {out_root} and rerun:")
        for r in legacy:
            log.error(f"    {r.rel}")
    for r in results:
        for flag in r.flags:
            log.warning(f"  CHECK   {r.rel}: {flag}")
    volumes = [v for r in results if r.status == "done" for v in r.face_mask_ml]
    if volumes:
        log.info(f"  Face mask volume (ml): min {min(volumes):.1f}, median {statistics.median(volumes):.1f}, "
                 f"max {max(volumes):.1f}. Check unusually small values in a viewer.")

    processed = [r for r in results if r.status in ("done", "dry run")]
    verb = "would be" if dry_run else "were"
    lines = []
    n_dose = sum(r.rtdose_removed for r in processed)
    if n_dose:
        lines.append(f"!!! RTDOSE: {n_dose} file(s) of {sum(1 for r in processed if r.rtdose_removed)} "
                     f"patient(s) {verb} NOT copied (the dose outlines the face; --keep-rtdose keeps them)")
    n_plan = sum(r.rtplan_removed for r in processed)
    if n_plan:
        lines.append(f"!!! RTPLAN/RTRECORD: {n_plan} file(s) {verb} NOT copied (--keep-rtplan keeps them)")
    removed = Counter(name for r in processed for name in r.rois_removed)
    if removed:
        lines.append(f"!!! CONTOURS {verb} removed from the RTSTRUCTs of "
                     f"{sum(1 for r in processed if r.rois_removed)} patient(s):")
        lines += [f"    {name:<32} {n} structure set(s)" for name, n in removed.most_common()]
        if not dry_run:
            lines.append(f"    Copies linked to the de-faced CT: {qc_root}/.../removed_rois (do not share)")
    unclear = Counter(name for r in processed for name in r.rois_unclear)
    if unclear:
        lines.append("!!! KEPT ROIs whose name suggests they may follow the skin or face (check them;")
        lines.append("    remove with --remove-roi NAME if they do):")
        lines += [f"    {name:<32} {n} structure set(s)" for name, n in unclear.most_common()]
    rows = [row for r in processed for row in r.overlap]
    inside = [row for row in rows if row["pct_in_defaced_region"] > 0]
    if inside:
        by_roi: dict[str, list[float]] = {}
        for row in inside:
            by_roi.setdefault(row["roi"], []).append(row["pct_in_defaced_region"])
        lines.append("ROIs overlapping the de-faced region (image content there is replaced; consider")
        lines.append("excluding them from training/evaluation). ROI: structure sets, mean % inside:")
        lines += [f"    {name:<32} {len(v):>4}  {statistics.mean(v):5.1f}%"
                  for name, v in sorted(by_roi.items(), key=lambda kv: -len(kv[1]))[:20]]
    targets = [row for row in rows if _is_target(row)]
    if targets:
        close = [row for row in targets if not str(row["min_distance_mm"]).startswith(">=")
                 and float(row["min_distance_mm"]) < 10]
        lines.append(f"Targets (CTV/GTV/PTV): {len(close)} of {len(targets)} lie within 10 mm of the "
                     "de-faced region (see the roi_overlap report).")
    if lines:
        log.info(bar)
        for ln in lines:
            (log.warning if ln.startswith("!!!") else log.info)(ln)

    log.info(bar)
    with_ids = [r for r in results if (r.ids and r.ids.flagged(Path(r.rel).name)) or (not r.ids and r.pii)]
    if not opts.pii_check:
        log.warning("Metadata check DISABLED (--no-pii-check). Metadata was NOT anonymised.")
    elif with_ids:
        per_tag = Counter(kw for r in with_ids for kw in (r.ids.flagged(Path(r.rel).name) if r.ids else r.pii))
        log.warning("!!! ATTENTION: METADATA WAS NOT ANONYMISED !!!")
        log.warning(f"{len(with_ids)} of {len(results)} patient(s) have populated identifying tags:")
        for kw, n in per_tag.most_common():
            log.warning(f"    {kw:<32} {n} patient(s)")
        log.warning("They are copied unchanged. De-identify the data before sharing it.")
    else:
        log.info("No populated direct identifiers found (name/ID absent or equal to the folder name).")
        log.info("Dates and other tags are listed in the report; review before sharing.")
    for path in reports.written():
        log.info(f"Report : {path}")
    if not dry_run:
        log.info(f"QC     : {qc_root}  (face masks and removed contours: confidential, do not share)")
    log.info(bar)
    msg = f"{title.title()}: {status.get('done', 0)} done, {status.get('failed', 0)} failed."
    if with_ids or not opts.pii_check:
        msg += " Metadata NOT anonymised."
    return msg


def notify(msg: str) -> None:
    print("\a", end="", flush=True)
    if sys.platform == "darwin":
        try:
            subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "CT de-facing"'],
                           check=False, capture_output=True, timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _process(kind: str, path: Path, rel: str, out_dir: Path, qc_dir: Path, opts: Options,
             dry_run: bool) -> PatientResult:
    t0 = time.time()
    fn = deface_dicom_patient if kind == "dicom" else deface_image_file
    try:
        res = fn(path, rel, out_dir, qc_dir, opts, dry_run=dry_run)
    except PatientError as e:
        res = PatientResult(rel=rel, error=str(e))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:                                   # unexpected: log it and keep going
        log.exception("    unexpected error")
        res = PatientResult(rel=rel, error=f"{type(e).__name__}: {e}")
    res.seconds = time.time() - t0
    if res.status == "failed":
        log.error(f"    FAILED: {res.error}")
    return res


def run_single(input_path: Path, out_dir: Path, qc_dir: Path, opts: Options, dry_run: bool) -> int:
    setup_logging()
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        return 2
    if not dry_run:
        if out_dir.exists() and any(out_dir.iterdir()):
            log.error(f"Output folder exists and is not empty: {out_dir}. Delete it or choose another -o.")
            return 2
        problem = check_totalsegmentator()
        if problem:
            log.error(problem)
            return 2
        if out_dir.exists():
            out_dir.rmdir()                                  # empty; recreated by the rename
    kind = "dicom" if input_path.is_dir() or not classify_suffix(input_path) else "image"
    rel = input_path.name if kind == "dicom" else strip_known_suffix(input_path)
    log.info(f"{input_path}  ->  {out_dir}   (QC: {qc_dir})")
    res = _process(kind, input_path, rel, out_dir, qc_dir, opts, dry_run)
    # written after processing, which starts by clearing the QC folder
    reports = Reports(qc_dir, dt.datetime.now().strftime("%Y%m%d_%H%M%S"), opts.pii_check)
    reports.write(res)
    reports.close()
    final_summary([res], reports, out_dir, qc_dir, dry_run, False, opts)
    return 0 if res.status in ("done", "dry run") else 1


def discover(root: Path, depth: int, ct_globs: list[str]) -> list[tuple[str, Path, str]]:
    """(kind, path, rel) of every patient: a folder at `depth` below root, or an image file
    at that depth. A folder holding a NIfTI/NRRD file matching ct_globs is an image patient."""
    entries, level = [], [root]
    for d in range(depth):
        nxt = []
        for folder in level:
            for c in sorted(folder.iterdir()):
                if c.name.startswith("."):
                    continue
                if c.is_dir():
                    nxt.append(c)
                elif d == depth - 1 and classify_suffix(c):
                    rel = c.relative_to(root)
                    entries.append(("image", c, (rel.parent / strip_known_suffix(c)).as_posix()))
        level = nxt
    for folder in level:
        rel = folder.relative_to(root).as_posix()
        image = next((m for g in ct_globs for m in sorted(folder.glob(g)) if not m.name.startswith(".")), None)
        entries.append(("image", image, rel) if image else ("dicom", folder, rel))
    return sorted(entries, key=lambda e: e[2])


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_batch(in_root: Path, out_root: Path, qc_root: Path, opts: Options, depth: int, only: list[str],
              ct_globs: list[str], dry_run: bool) -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = qc_root / REPORTS_DIRNAME
    setup_logging(None if dry_run else report_dir / f"run_{stamp}.log")
    if not in_root.is_dir():
        log.error(f"Input folder not found: {in_root}")
        return 2
    if out_root == in_root or in_root in out_root.parents:
        log.error("The output folder must not be the input folder or lie inside it.")
        return 2
    if qc_root == out_root or out_root in qc_root.parents:
        log.error("The QC folder must not lie inside the output folder (it holds the face masks).")
        return 2
    if not dry_run:
        problem = check_totalsegmentator()
        if problem:
            log.error(problem)
            return 2

    patients = discover(in_root, depth, ct_globs)
    if only:
        patients = [p for p in patients if any(t in p[2] for t in only)]
    if not patients:
        log.error("No patients found (check the input folder, --depth and --only).")
        if any(c.is_file() and not c.name.startswith(".") for c in in_root.iterdir()):
            log.error(f"  {in_root} itself contains files, so it looks like ONE patient folder. --batch "
                      "expects the folder that contains the patient folders (e.g. its parent, with "
                      f"--depth 1); for a single patient use: -i {in_root}")
        else:
            log.error(f"  Patient folders are looked for {depth} folder level(s) below {in_root} (--depth {depth}).")
        return 2
    todo = [p for p in patients if not (out_root / p[2]).exists()]
    log.info(f"Input   : {in_root}")
    log.info(f"Output  : {out_root}{'  (dry run, nothing written)' if dry_run else ''}")
    log.info(f"QC      : {qc_root}")
    log.info(f"Patients: {len(patients)} found, {len(patients) - len(todo)} already in the output, "
             f"{len(todo)} to process | " + ", ".join(f"{k} {v}" for k, v in opts.settings().items()))
    if not dry_run and todo:
        needed = sum(_size(p[1]) for p in todo)
        out_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(out_root).free
        log.info(f"Space   : about {needed / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")
        if free < needed * 1.5:          # de-faced output is uncompressed
            log.error("Not enough free space on the output drive.")
            return 2

    reports = Reports(report_dir, stamp, opts.pii_check)
    results: list[PatientResult] = []
    durations: list[float] = []
    interrupted, n_todo_done = False, 0
    try:
        for i, (kind, path, rel) in enumerate(patients, 1):
            log.info(f"[{i:>3}/{len(patients)}] {rel}")
            out_dir, qc_dir = out_root / rel, qc_root / rel
            if not dry_run and (out_dir.exists()):
                manifest = qc_dir / "manifest.json"
                if not manifest.is_file():
                    log.error("    output exists but has no manifest (earlier version?): delete it and rerun")
                    res = PatientResult(rel=rel, status="legacy output")
                else:
                    log.info("    already done, skipping (delete its output folder to redo it)")
                    res = PatientResult(rel=rel, status="already done")
                    done_with = json.loads(manifest.read_text(encoding="utf-8"))
                    diff = [f"{k}={done_with.get(k)}" for k, v in opts.settings().items() if done_with.get(k) != v]
                    if done_with.get("script_version") != SCRIPT_VERSION:
                        diff.insert(0, f"tool version {done_with.get('script_version')} (now {SCRIPT_VERSION})")
                    if diff:
                        res.flags.append(f"done earlier with other settings: {', '.join(diff)}")
            else:
                res = _process(kind, path, rel, out_dir, qc_dir, opts, dry_run)
                if not dry_run:
                    n_todo_done += 1
                    if res.status == "done":
                        durations.append(res.seconds)
                        remaining = len(todo) - n_todo_done
                        eta = (f", about {_fmt(statistics.mean(durations) * remaining)} left" if remaining else "")
                        log.info(f"    done in {_fmt(res.seconds)} (face mask "
                                 f"{', '.join(f'{v:.0f} ml' for v in res.face_mask_ml)}){eta}")
            results.append(res)
            reports.write(res)

            if not dry_run and (not in_root.exists() or not out_root.exists()):
                log.error("The input or output folder disappeared (drive unplugged?); stopping.")
                break
            if (not dry_run and n_todo_done >= 3 and not durations
                    and all(r.status == "failed" for r in results if r.status not in ("already done", "legacy output"))):
                log.error("The first 3 patients all failed; stopping. Check the errors above.")
                break
    except KeyboardInterrupt:
        interrupted = True
        log.warning("\nInterrupted. The unfinished patient was cleaned up; rerun to continue.")
    finally:
        reports.close()

    msg = final_summary(results, reports, out_root, qc_root, dry_run, interrupted, opts)
    notify(msg)
    return 1 if any(r.status in ("failed", "legacy output") for r in results) or interrupted else 0


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
