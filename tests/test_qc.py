"""QC pictures: the font, and the ROI overlap picture."""
import numpy as np
import pytest
import SimpleITK as sitk

from defacing.font import ascii_text, render
from defacing.masks import Axes
from defacing.qc import MAX_ROIS, roi_label, write_roi_picture


def test_font_renders_every_printable_character():
    for code in range(33, 127):
        glyph = render(chr(code), scale=1)
        assert glyph.shape == (7, 5) and glyph.any(), chr(code)
    assert not render(" ", scale=1).any()
    assert render("AB", scale=2).shape == (14, 22)                # 2 x 5 pixels + 1 space, doubled
    assert np.array_equal(render("é", 1), render("?", 1))          # outside ASCII
    assert ascii_text("Glaskörper_Außen") == "Glaskoerper_Aussen"
    assert not np.array_equal(render("O", 1), render("0", 1))


def test_roi_label():
    assert roi_label("Eye_L", 100.0, 8.4, 0.0) == "Eye_L: 100% in region (8.4 ml)"
    assert roi_label("PTV", 0.0, 0.0, 6.5) == "PTV: 6.5 mm away"
    assert len(roi_label("x" * 80, 1.0, 1.0, 0.0)) < 70


def head(shape=(40, 60, 50)):
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]]
    before = np.where(((z - 20) / 18) ** 2 + ((y - 32) / 25) ** 2 + ((x - 25) / 20) ** 2 <= 1, 40, -1000)
    region = np.zeros(shape, bool)
    region[12:28, :12, 10:40] = True                              # front of the head (y index 0 = anterior)
    return before.astype(np.int16), region


AXES = Axes(spacing=(2.0, 2.0, 2.0), anterior=(1, False), si_axis=0, superior=True)


def test_roi_picture(tmp_path):
    before, region = head()
    eye = np.zeros_like(region)
    eye[18:22, 8:12, 14:18] = True
    near = np.zeros_like(region)
    near[18:22, 14:18, 30:34] = True                               # just behind the region
    path = tmp_path / "QC_roi_overlap.png"
    near_slices = {z: near[z] for z in range(near.shape[0]) if near[z].any()}   # as from the contours
    assert write_roi_picture(before, region, [("Eye_L: 100% in region (0.5 ml)", eye),
                                              ("Optic: 4 mm away", near_slices)], AXES, path)
    img = sitk.ReadImage(str(path))
    pic = sitk.GetArrayFromImage(img)
    assert img.GetNumberOfComponentsPerPixel() == 3 and pic.dtype == np.uint8
    red = (pic[..., 0] > 200) & (pic[..., 1] < 120) & (pic[..., 2] < 120)
    yellow = (pic[..., 0] > 200) & (pic[..., 1] > 180) & (pic[..., 2] < 80)
    assert red.any() and yellow.any()                               # both ROIs (and their swatches) drawn
    legend = pic[:, pic.shape[1] // 2:]
    assert (legend == 255).all(axis=2).sum() > 500                  # white text


def test_roi_picture_without_rois_or_with_many(tmp_path):
    before, region = head()
    assert not write_roi_picture(before, region, [], AXES, tmp_path / "none.png")
    assert not (tmp_path / "none.png").exists()
    many = []
    for k in range(MAX_ROIS + 3):
        m = np.zeros_like(region)
        m[10 + k, 5:10, 20:25] = True
        many.append((f"ROI_{k}: 100% in region (0.1 ml)", m))
    assert write_roi_picture(before, region, many[:MAX_ROIS], AXES, tmp_path / "many.png", more=3)
    assert sitk.ReadImage(str(tmp_path / "many.png")).GetHeight() >= (MAX_ROIS + 4) * 20


@pytest.mark.parametrize("bad", [Axes(spacing=(2.0, 2.0, 2.0), anterior=(0, True), si_axis=0)])
def test_roi_picture_skips_unclear_orientation(tmp_path, bad):
    before, region = head()
    assert not write_roi_picture(before, region, [("x", region)], bad, tmp_path / "x.png")
