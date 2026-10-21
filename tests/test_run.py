"""Single and batch runs: discovery, safety checks, resume, reports, and the command line."""
import csv
import json
import logging
import sys
from pathlib import Path

import pytest

import phantom as ph
from defacing.common import Options
from defacing.patient import PatientResult
from defacing.run import DEFAULT_CT_GLOBS, Reports, discover, final_summary, run_batch, run_single

SMALL = dict(shape=(30, 48, 48), spacing=(5.0, 5.0, 7.0))
REPO = Path(__file__).resolve().parent.parent


def dicom_patient(folder: Path, name: str, rt: bool = True):
    ct = ph.write_ct_series(folder / "CT", patient=name, **SMALL)
    if rt:
        ph.write_rtstruct(folder / "RS.dcm", ct, patient=name)
    return ct


@pytest.fixture
def cohort(tmp_path) -> Path:
    root = tmp_path / "patients"
    dicom_patient(root / "TUM_001", "TUM_001")
    dicom_patient(root / "TUM_002", "Mustermann^Max")
    ph.write_nifti(root / "TUM_003" / "TUM_003_CT.nii.gz")
    ph.write_nifti(root / "TUM_004_CT.nii.gz")
    (root / ".hidden").mkdir()
    return root


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_discover_depth_1(cohort):
    found = discover(cohort, 1, DEFAULT_CT_GLOBS)
    assert [(k, rel) for k, _, rel in found] == [("dicom", "TUM_001"), ("dicom", "TUM_002"),
                                                 ("image", "TUM_003"), ("image", "TUM_004_CT")]
    assert found[2][1].name == "TUM_003_CT.nii.gz"


def test_discover_depth_2_and_globs(tmp_path):
    root = tmp_path / "p"
    (root / "HNC" / "P1").mkdir(parents=True)
    (root / "HNC" / "P1" / "scan.nrrd").write_bytes(b"")
    (root / "HNC" / "P2").mkdir()
    (root / "Lung" / "P3").mkdir(parents=True)
    (root / "loose.nii.gz").write_bytes(b"")                    # not at depth 2: ignored
    found = discover(root, 2, ["*.nrrd"])
    assert [(k, rel) for k, _, rel in found] == [("image", "HNC/P1"), ("dicom", "HNC/P2"), ("dicom", "Lung/P3")]


def test_discover_warns_about_ignored_files(tmp_path, caplog):
    folder = tmp_path / "p" / "P1"
    ph.write_ct_series(folder / "dicom", **SMALL)
    (folder / "P1_CT.nii.gz").write_bytes(b"")
    (folder / "P1_CTV.nii.gz").write_bytes(b"")
    (folder / "P1_CT.json").write_text("{}")
    with caplog.at_level(logging.WARNING, logger="deface"):
        found = discover(tmp_path / "p", 1, DEFAULT_CT_GLOBS)
    assert found[0][1].name == "P1_CT.nii.gz"
    assert "only P1_CT.nii.gz is de-faced" in caplog.text and "2 other" in caplog.text


# ---------------------------------------------------------------------------
# single patient
# ---------------------------------------------------------------------------

def test_run_single_dicom(cohort, tmp_path, fake_ts, ts_on_path):
    out, qc = tmp_path / "res" / "TUM_002", tmp_path / "res" / "TUM_002_qc"
    assert run_single(cohort / "TUM_002", out, qc, Options(), False) == 0
    assert (out / "dcm").is_dir() and (out / "rt").is_dir()
    reports = {p.name.rsplit("_", 2)[0] if p.name[-19:-4].replace("_", "").isdigit() else p.name
               for p in qc.glob("*.csv")}
    assert reports == {"patients", "identifying_tags", "roi_overlap", "roi_overlap.csv"}
    rows = list(csv.DictReader(open(next(qc.glob("patients_*.csv")), encoding="utf-8")))
    assert rows[0]["status"] == "done" and "PatientName" in rows[0]["identifying_tags_populated"]


def test_run_single_refuses_non_empty_output_and_missing_input(cohort, tmp_path, fake_ts, ts_on_path):
    out = tmp_path / "res"
    out.mkdir()
    (out / "old.txt").write_text("x")
    assert run_single(cohort / "TUM_001", out, tmp_path / "qc", Options(), False) == 2
    assert (out / "old.txt").exists() and not fake_ts.calls
    assert run_single(tmp_path / "nope", tmp_path / "res2", tmp_path / "qc2", Options(), False) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_single(cohort / "TUM_001", empty, tmp_path / "qc3", Options(), False) == 0


def test_run_single_needs_totalsegmentator(cohort, tmp_path, monkeypatch):
    import defacing.run
    monkeypatch.setattr(defacing.run, "check_totalsegmentator", lambda: "not found")
    assert run_single(cohort / "TUM_001", tmp_path / "res", tmp_path / "qc", Options(), False) == 2


def test_run_single_failure_leaves_no_output(cohort, tmp_path, fake_ts, ts_on_path):
    fake_ts.mode = "empty"
    out = tmp_path / "res"
    assert run_single(cohort / "TUM_001", out, tmp_path / "qc", Options(), False) == 1
    assert not out.exists() and not (tmp_path / "res.partial").exists()


def test_run_single_image(cohort, tmp_path, fake_ts, ts_on_path):
    out = tmp_path / "res"
    assert run_single(cohort / "TUM_004_CT.nii.gz", out, tmp_path / "qc", Options(), False) == 0
    assert (out / "TUM_004_CT_deFaced.nii.gz").exists()


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def batch(cohort, tmp_path, opts=None, **kw):
    args = dict(depth=1, only=[], ct_globs=DEFAULT_CT_GLOBS, dry_run=False)
    args.update(kw)
    return run_batch(cohort, tmp_path / "results", tmp_path / "results_qc", opts or Options(), **args)


def test_batch_processes_resumes_and_reports(cohort, tmp_path, fake_ts, ts_on_path, caplog):
    with caplog.at_level(logging.INFO, logger="deface"):
        assert batch(cohort, tmp_path) == 0
    res, qc = tmp_path / "results", tmp_path / "results_qc"
    assert sorted(p.name for p in res.iterdir()) == ["TUM_001", "TUM_002", "TUM_003", "TUM_004_CT"]
    assert all((qc / p / "manifest.json").is_file() for p in ("TUM_001", "TUM_002", "TUM_003", "TUM_004_CT"))
    reports = qc / "_reports"
    rows = list(csv.DictReader(open(next(reports.glob("patients_*.csv")), encoding="utf-8")))
    assert [r["status"] for r in rows] == ["done"] * 4
    assert next(reports.glob("run_*.log")).stat().st_size > 0
    ids = list(csv.DictReader(open(next(reports.glob("identifying_tags_*.csv")), encoding="utf-8")))
    assert any(r["keyword"] == "PatientName" and r["pseudonym_matches_folder"] == "yes" for r in ids
               if r["patient"] == "TUM_001")
    assert "counted as pseudonyms" in caplog.text            # TUM_001: name equals the folder name
    assert "Mustermann" not in caplog.text                    # values only in the confidential report
    # resume: nothing is redone
    n_calls = len(fake_ts.calls)
    assert batch(cohort, tmp_path) == 0
    assert len(fake_ts.calls) == n_calls


def test_batch_flags_other_settings_and_legacy_output(cohort, tmp_path, fake_ts, ts_on_path, caplog):
    batch(cohort, tmp_path, only=["TUM_001"])
    (tmp_path / "results" / "TUM_002").mkdir()                  # output without manifest
    with caplog.at_level(logging.INFO, logger="deface"):
        code = batch(cohort, tmp_path, Options(dilate=5), only=["TUM_001", "TUM_002"])
    assert code == 1
    assert "done earlier with other settings: dilate=2" in caplog.text
    assert "no manifest" in caplog.text


def test_batch_continues_after_a_failed_patient(cohort, tmp_path, fake_ts, ts_on_path):
    (cohort / "TUM_000").mkdir()
    (cohort / "TUM_000" / "notes.txt").write_text("no CT here")
    assert batch(cohort, tmp_path) == 1
    res = tmp_path / "results"
    assert not (res / "TUM_000").exists() and (res / "TUM_001").exists() and (res / "TUM_004_CT").exists()


def test_batch_stops_when_the_first_three_fail(cohort, tmp_path, fake_ts, ts_on_path):
    fake_ts.mode = "empty"
    assert batch(cohort, tmp_path) == 1
    assert len(fake_ts.calls) == 3


def test_batch_dry_run_writes_only_the_reports(cohort, tmp_path, fake_ts):
    assert batch(cohort, tmp_path, dry_run=True) == 0
    assert not (tmp_path / "results").exists() and not fake_ts.calls
    written = sorted(p.relative_to(tmp_path / "results_qc").parts[0] for p in (tmp_path / "results_qc").rglob("*")
                     if p.is_file())
    assert set(written) == {"_reports"}
    assert not list((tmp_path / "results_qc").rglob("run_*.log"))


@pytest.mark.parametrize("out, qc", [("patients", "qc"), ("patients/sub", "qc"), ("res", "res/qc"), ("res", "res")])
def test_batch_refuses_dangerous_folders(cohort, tmp_path, fake_ts, ts_on_path, out, qc):
    root = cohort.parent
    assert run_batch(cohort, root / out, root / qc, Options(), 1, [], DEFAULT_CT_GLOBS, False) == 2
    assert not fake_ts.calls


def test_batch_without_patients(tmp_path, ts_on_path):
    (tmp_path / "empty").mkdir()
    assert batch(tmp_path / "empty", tmp_path) == 2


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def cli(monkeypatch, *args):
    import deface
    monkeypatch.setattr(sys, "argv", ["deface.py", *args])
    return deface.main()


@pytest.mark.parametrize("args, message", [
    (["--fill", "soft-tissue", "--no-extend-anterior"], "soft-tissue"),
    (["--depth", "0"], "--depth"),
    (["--dilate", "-1"], "--dilate"),
    (["--dilate-mm", "-1"], "--dilate-mm"),
    (["--dilate-mm", "nan"], "--dilate-mm"),
    (["--dilate-mm", "inf"], "--dilate-mm"),
    (["--remove-roi", "("], "invalid regular expression"),
    (["--fill", "zero"], "invalid choice"),
])
def test_cli_rejects_invalid_arguments(tmp_path, monkeypatch, capsys, args, message):
    with pytest.raises(SystemExit) as e:
        cli(monkeypatch, "-i", str(tmp_path), "-o", str(tmp_path / "o"), *args)
    assert e.value.code == 2 and message in capsys.readouterr().err


def test_cli_needs_exactly_one_input(tmp_path, monkeypatch):
    for args in ([], ["-i", "a", "--batch", "b"]):
        with pytest.raises(SystemExit):
            cli(monkeypatch, "-o", str(tmp_path / "o"), *args)


def test_cli_refuses_qc_inside_output(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="QC folder"):
        cli(monkeypatch, "-i", str(tmp_path), "-o", str(tmp_path / "o"), "--qc-dir", str(tmp_path / "o" / "qc"))


def test_cli_passes_the_options_on(cohort, tmp_path, monkeypatch):
    import deface
    seen = {}
    monkeypatch.setattr(deface, "run_batch", lambda *a: seen.setdefault("args", a) and 0)
    cli(monkeypatch, "--batch", str(cohort), "-o", str(tmp_path / "o"), "--dilate", "4", "--dilate-mm", "2.5",
        "--fill", "constant",
        "--no-extend-anterior", "--keep-body", "--remove-roi", "z_.*", "--ct-glob", "*a*.nrrd, *b*.nii",
        "--depth", "2", "--only", "TUM", "--no-pii-check")
    in_root, out, qc, opts, depth, only, globs, dry = seen["args"]
    assert out.name == "o" and qc.name == "o_qc"
    assert (opts.dilate, opts.dilate_mm, opts.fill, opts.extend_anterior, opts.keep_body, opts.pii_check) == (
        4, 2.5, "constant", False, True, False)
    assert opts.remove_roi[0].fullmatch("Z_RING") and depth == 2 and only == ["TUM"]
    assert globs == ["*a*.nrrd", "*b*.nii"] and dry is False


def test_manifest_records_the_settings(cohort, tmp_path, fake_ts, ts_on_path):
    run_single(cohort / "TUM_001", tmp_path / "o", tmp_path / "qc", Options(dilate=3, fill="constant"), False)
    m = json.loads((tmp_path / "qc" / "manifest.json").read_text())
    assert m["dilate"] == 3 and m["fill"] == "constant" and m["script_version"] >= 4
    assert m["rois_removed"] == ["BODY"]


def test_summary_flags_unusually_small_face_masks(tmp_path, caplog):
    results = [PatientResult(rel=f"P{i}", status="done", face_mask_ml=[v]) for i, v in enumerate([300, 320, 310,
                                                                                                  290, 330, 90])]
    reports = Reports(tmp_path, "x", True)
    with caplog.at_level(logging.WARNING, logger="deface"):
        final_summary(results, reports, tmp_path, tmp_path, False, False, Options())
    assert "P5: face mask unusually small (90 ml" in caplog.text
    assert "P4: face mask unusually small" not in caplog.text
    few = [PatientResult(rel="Q", status="done", face_mask_ml=[10]), PatientResult(rel="R", status="done",
                                                                                     face_mask_ml=[300])]
    caplog.clear()
    final_summary(few, reports, tmp_path, tmp_path, False, False, Options())
    assert "face mask unusually small" not in caplog.text  # too few patients for a median
