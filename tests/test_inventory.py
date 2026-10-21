"""Inventory of a patient folder: what is de-faced, copied, or left out (and why)."""
from pathlib import Path

import numpy as np
import pytest

import phantom as ph
from defacing.common import Options
from defacing.inventory import inventory

SMALL = dict(shape=(4, 8, 8), spacing=(30.0, 30.0, 40.0))


@pytest.fixture
def patient(tmp_path) -> Path:
    """One folder with every kind of object a clinical export may contain."""
    d = tmp_path / "PAT01"
    ct = ph.write_ct_series(d / "CT" / "planning", **SMALL)
    ph.write_ct_series(d / "CT" / "replan", study_uid=ct.study_uid, **SMALL)
    ph.write_ct_series(d / "scouts" / "loc", image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"), **SMALL)
    ph.write_ct_series(d / "scouts" / "topo", image_type=("ORIGINAL", "PRIMARY"), extra={"ScanOptions": "SURVIEW"},
                       **SMALL)
    ph.write_ct_series(d / "scouts" / "single", shape=(2, 8, 8), spacing=(30.0, 30.0, 40.0))
    ph.write_simple(d / "dose_report.dcm", "CT", ph.SECONDARY_CAPTURE, ct, BurnedInAnnotation="YES")
    ph.write_simple(d / "enhanced.dcm", "CT", ph.ENHANCED_CT, ct)
    ph.write_rtstruct(d / "RS.dcm", ct)
    ph.write_simple(d / "RD.dcm", "RTDOSE", ph.RTDOSE_STORAGE, ct)
    ph.write_simple(d / "RP.dcm", "RTPLAN", ph.RTPLAN_STORAGE, ct)
    ph.write_simple(d / "RR.dcm", "RTRECORD", "1.2.840.10008.5.1.4.1.1.481.4", ct)
    ph.write_simple(d / "MR" / "mr1", "MR", ph.MR_IMAGE_STORAGE)
    (d / "notes.txt").write_text("not DICOM")
    (d / "._RS.dcm").write_bytes(b"\x00\x05\x16\x07")               # macOS AppleDouble sidecar
    (d / ".DS_Store").write_bytes(b"x")
    raw = (d / "RS.dcm").read_bytes()
    (d / "broken.dcm").write_bytes(raw[:200])                        # truncated
    return d


def reasons(inv) -> dict[str, str]:
    return {Path(rel).as_posix(): why for rel, why in inv.not_copied}


def test_ct_series_and_rt_objects_are_found(patient):
    inv = inventory(patient, Options())
    series = inv.ct_series
    assert len(series) == 2 and all(len(v) == 4 for v in series.values())
    assert {f.rel.parts[1] for f in inv.ct} == {"planning", "replan"}
    assert [f.modality for f in inv.rt] == ["RTSTRUCT"]
    assert inv.rt[0].rois[1] == ("BODY", "EXTERNAL")
    assert inv.rtdose_removed == 1 and inv.rtplan_removed == 2


def test_objects_that_cannot_be_defaced_are_not_copied(patient):
    why = reasons(inventory(patient, Options()))
    assert all("localizer" in why[k] for k in why if k.startswith("scouts/loc/") or k.startswith("scouts/topo/"))
    assert all("no volume" in why[k] for k in why if k.startswith("scouts/single/"))
    assert "not a CT image" in why["dose_report.dcm"]
    assert "Enhanced CT" in why["enhanced.dcm"]
    assert "RTDOSE removed" in why["RD.dcm"]
    assert "RTPLAN removed" in why["RP.dcm"] and "RTRECORD removed" in why["RR.dcm"]
    assert "modality 'MR'" in why["MR/mr1"]
    assert why["notes.txt"] == "not a DICOM file"
    assert "broken.dcm" in why                                      # truncated: never copied
    assert "._RS.dcm" not in why and ".DS_Store" not in why        # hidden files are skipped silently


def test_keep_options_copy_rtdose_and_rtplan(patient):
    inv = inventory(patient, Options(keep_rtdose=True, keep_rtplan=True))
    assert sorted(f.modality for f in inv.rt) == ["RTDOSE", "RTPLAN", "RTRECORD", "RTSTRUCT"]
    assert inv.rtdose_removed == inv.rtplan_removed == 0


def test_series_with_mixed_orientations_is_not_copied(tmp_path):
    d = tmp_path / "P"
    ap = ph.write_ct_series(d / "topo", orientation="coronal", **SMALL)
    ph.write_ct_series(d / "topo2", orientation="sagittal", **SMALL)
    for p in sorted((d / "topo2").iterdir()):                       # same series as the coronal views
        import pydicom
        ds = pydicom.dcmread(p)
        ds.SeriesInstanceUID = ap.series_uid
        ds.save_as(p)
    inv = inventory(d, Options())
    assert not inv.ct and all("different orientations" in why for _, why in inv.not_copied)


def test_metadata_is_collected_unless_disabled(patient):
    assert "PatientName" in inventory(patient, Options()).ids.values
    assert inventory(patient, Options(pii_check=False)).ids is None


def test_single_file_input(tmp_path):
    ct = ph.write_ct_series(tmp_path / "CT", **SMALL)
    inv = inventory(ct.paths[0], Options())
    assert len(inv.ct) == 0 and "no volume" in inv.not_copied[0][1]


def test_burned_in_annotation_tag(tmp_path):
    ph.write_ct_series(tmp_path / "P" / "CT", extra={"BurnedInAnnotation": "YES"}, **SMALL)
    inv = inventory(tmp_path / "P", Options())
    assert not inv.ct and all("BurnedInAnnotation" in why for _, why in inv.not_copied)
    ph.write_ct_series(tmp_path / "Q" / "CT", extra={"BurnedInAnnotation": "NO"}, **SMALL)
    assert len(inventory(tmp_path / "Q", Options()).ct) == 4


def test_dicom_without_file_header_is_reported(tmp_path):
    import pydicom
    from pydicom.filebase import DicomFileLike
    from pydicom.filewriter import write_dataset
    ct = ph.write_ct_series(tmp_path / "P" / "CT", **SMALL)
    for p in ct.paths:                                   # dataset only: no preamble, no 'DICM', no meta
        ds = pydicom.dcmread(p)
        with open(p, "wb") as fh:
            fp = DicomFileLike(fh)
            fp.is_little_endian, fp.is_implicit_VR = True, True
            write_dataset(fp, ds)
    (tmp_path / "P" / "notes.txt").write_text("hello")
    inv = inventory(tmp_path / "P", Options())
    assert inv.headerless == 4 and not inv.ct
    why = reasons(inv)
    assert all("without the DICOM file header" in why[f"CT/{p.name}"] for p in ct.paths)
    assert why["notes.txt"] == "not a DICOM file"


def test_pixel_spacing_is_recorded(tmp_path):
    ct = ph.write_ct_series(tmp_path / "P" / "CT", **SMALL)
    import pydicom
    ds = pydicom.dcmread(ct.paths[0])
    del ds.PixelSpacing
    ds.save_as(ct.paths[0])
    flags = {f.rel.name: f.pixel_spacing for f in inventory(tmp_path / "P", Options()).ct}
    assert flags[ct.paths[0].name] is False and sum(flags.values()) == 3


def test_rounding_noise_in_the_orientation_is_not_a_mixed_series(tmp_path):
    import pydicom
    ct = ph.write_ct_series(tmp_path / "P" / "CT", **SMALL)
    for k, p in enumerate(ct.paths):                       # 0.00049 / 0.00051: round to 0.0 / 0.001
        ds = pydicom.dcmread(p)
        ds.ImageOrientationPatient = ["1", f"{0.00049 if k % 2 else 0.00051:.5f}", "0", "0", "1", "0"]
        ds.save_as(p)
    assert len(inventory(tmp_path / "P", Options()).ct) == 4


def test_large_non_dicom_files_are_not_read_completely(tmp_path):
    import time
    d = tmp_path / "P"
    d.mkdir()
    (d / "video.bin").write_bytes(np.random.default_rng(0).bytes(50_000_000))
    t0 = time.time()
    why = reasons(inventory(d, Options()))
    assert why["video.bin"] == "not a DICOM file" and time.time() - t0 < 5
