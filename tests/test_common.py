"""Small helpers: file name suffixes, safe casting, hidden files."""
from pathlib import Path

import numpy as np
import pytest

from defacing.common import Options, classify_suffix, is_hidden, safe_cast, strip_known_suffix


@pytest.mark.parametrize("name, suffix", [
    ("ct.nii.gz", ".nii.gz"), ("CT.NII.GZ", ".nii.gz"), ("ct.nii", ".nii"), ("ct.nrrd", ".nrrd"),
    ("scan.v2.final.nii.gz", ".nii.gz"), ("CT.1.2.840.113619.2.55.dcm", ""), ("IM0001", ""),
    ("ct.nii.gz.bak", ""), ("mask.seg.nrrd", ".nrrd"),
])
def test_classify_suffix(name, suffix):
    assert classify_suffix(Path(name)) == suffix


@pytest.mark.parametrize("name, stem", [
    ("p01_CT.nii.gz", "p01_CT"), ("a.b.c.nii", "a.b.c"), ("x.nrrd", "x"), ("CT.1.2.3.dcm", "CT.1.2.3"),
])
def test_strip_known_suffix(name, stem):
    assert strip_known_suffix(Path(name)) == stem


def test_safe_cast_clips_and_rounds_instead_of_wrapping():
    data = np.array([-40000.0, -1.6, 1.5, 2.4, 40000.0])
    assert safe_cast(data, np.int16).tolist() == [-32768, -2, 2, 2, 32767]
    assert safe_cast(np.array([-1000.0, 70000.0]), np.uint16).tolist() == [0, 65535]
    floats = safe_cast(np.array([1.25]), np.float32)
    assert floats.dtype == np.float32 and floats[0] == 1.25


@pytest.mark.parametrize("rel, hidden", [
    ("._CT001.dcm", True), (".DS_Store", True), ("a/.git/x.dcm", True), ("CT/IM0001", False),
    ("pat.01/ct.dcm", False),
])
def test_is_hidden(rel, hidden):
    assert is_hidden(Path(rel)) is hidden


def test_settings_cover_every_option_that_changes_the_output():
    s = Options().settings()
    assert set(s) == {"dilate_mm", "dilate", "extend_anterior", "fill", "keep_body", "keep_skin", "keep_rtplan",
                      "keep_rtdose", "remove_roi"}
    assert s["fill"] == "air" and s["extend_anterior"] is True and s["dilate_mm"] == 3.0 and s["dilate"] == 2
