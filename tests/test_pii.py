"""Metadata check: which identifying data is found and reported (values never in the log)."""
import json
import logging

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import phantom as ph
from defacing.pii import IdentifierSummary, burned_in_text_pixels, header_findings, warn_dicom


def dataset(name="Mustermann^Max", pid="123456"):
    ds = Dataset()
    ds.PatientName, ds.PatientID, ds.PatientBirthDate = name, pid, "19700101"
    ds.InstitutionName = "Klinikum Musterstadt"
    ds.StudyDate, ds.SeriesDescription = "20240101", "Head"
    item = Dataset()
    item.add_new(0x00081050, "PN", "Doe^Jane")                   # PerformingPhysicianName in a sequence
    ds.add_new(0x00081111, "SQ", Sequence([item]))
    ds.add_new(0x00091010, "LO", "vendor data")                  # private tag
    return ds


def test_collect_direct_indirect_nested_names_and_private_tags():
    ids = IdentifierSummary()
    ids.collect(dataset())
    assert ids.flagged("PAT01") == ["InstitutionName", "PatientBirthDate", "PatientID", "PatientName",
                                    "PerformingPhysicianName"]
    assert {"StudyDate", "SeriesDescription"} <= set(ids.indirect())
    assert 0x00091010 in ids.private_tags


def test_pseudonym_equal_to_the_folder_is_not_flagged():
    ids = IdentifierSummary()
    ds = Dataset()
    ds.PatientName, ds.PatientID = "TUM_401", "tum_401"
    ids.collect(ds)
    assert ids.flagged("TUM_401") == [] and ids.matches_folder("PatientName", "TUM_401")
    ids.collect(dataset(name="TUM_401", pid="other"))            # one file with another ID
    assert "PatientID" in ids.flagged("TUM_401")


def test_warning_names_tags_but_never_values(caplog):
    ids = IdentifierSummary()
    ids.collect(dataset())
    with caplog.at_level(logging.INFO, logger="deface"):
        flagged = warn_dicom(ids, "PAT01")
    assert "PatientName" in flagged
    text = caplog.text
    assert "PatientName" in text
    for value in ("Mustermann", "123456", "19700101", "Klinikum", "Doe"):
        assert value not in text


def test_nifti_free_text_extension_and_json_sidecar(tmp_path):
    p = tmp_path / "p01_CT.nii.gz"
    ph.write_nifti(p, descrip="Mustermann Max")
    img = nib.load(str(p))
    img.header.extensions.append(nib.nifti1.Nifti1Extension(6, b'{"PatientName": "Mustermann"}'))
    nib.save(img, str(p))
    (tmp_path / "p01_CT.json").write_text(json.dumps({"PatientName": "x", "PatientBirthDate": "",
                                                      "SliceThickness": 2}))
    found = header_findings(p)
    assert "NIfTI header 'descrip' (free text)" in found
    assert "NIfTI header extension (code 6)" in found
    assert "p01_CT.json key 'PatientName'" in found
    assert not any("PatientBirthDate" in f or "SliceThickness" in f for f in found)   # empty / harmless


def test_clean_nifti_has_no_findings(tmp_path):
    p = tmp_path / "ct.nii"
    ph.write_nifti(p)
    assert header_findings(p) == []


def test_nrrd_keys(tmp_path):
    img = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    img.SetMetaData("PatientName", "Mustermann")
    img.SetMetaData("0010|0020", "123")
    img.SetMetaData("Modality", "CT")
    sitk.WriteImage(img, str(tmp_path / "ct.nrrd"))
    found = header_findings(tmp_path / "ct.nrrd")
    assert "NRRD key 'PatientName'" in found and "NRRD key '0010|0020'" in found
    assert not any("Modality" in f for f in found)


def test_collect_reads_real_files(tmp_path):
    ct = ph.write_ct_series(tmp_path / "CT", shape=(3, 8, 8), patient="Mustermann^Max")
    ids = IdentifierSummary()
    for p in ct.paths:
        ids.collect(pydicom.dcmread(p, stop_before_pixels=True))
    assert ids.values["PatientName"]["Mustermann^Max"] == 3


def head_slices(n=20, size=128):
    rng = np.random.default_rng(0)
    hu = np.full((n, size, size), -1000.0) + rng.normal(0, 10, (n, size, size))
    yy, xx = np.mgrid[:size, :size]
    head = (yy - 64) ** 2 / 50 ** 2 + (xx - 64) ** 2 / 40 ** 2 <= 1
    hu[:, head] = 40.0
    hu[:, (yy - 64) ** 2 / 50 ** 2 + (xx - 64) ** 2 / 40 ** 2 <= 0.3] = 1800.0   # dense bone inside
    hu[:, 120:124, :] = 300.0                                                    # couch
    hu[:, 121, 10:20] = 2500.0                                                   # metal in the couch
    return hu


def test_burned_in_text_is_found_and_anatomy_is_not():
    hu = head_slices()
    assert burned_in_text_pixels(hu, 0) == 0
    hu[:, 3:9, 4:30:3] = 3071.0                         # "text" in the corner of every slice
    n = burned_in_text_pixels(hu, 0)
    assert n == 6 * 9
    assert burned_in_text_pixels(np.moveaxis(hu, 0, 2), 2) == n
    hu[:5, 3:9, 4:30:3] = -1000.0                       # in most slices is still enough ...
    assert burned_in_text_pixels(hu, 0) == n
    hu[:15, 3:9, 4:30:3] = -1000.0                      # ... a few slices are not
    assert burned_in_text_pixels(hu, 0) == 0
