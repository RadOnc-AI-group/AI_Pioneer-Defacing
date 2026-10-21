"""
QC pictures (confidential: they show the face).

QC_defacing*.png, one per CT series: two rows (top: input, bottom: de-faced output) of two views
  left   side view: silhouette of the soft tissue seen from the side, face to the left
  right  front view: the skin surface seen from the front, shaded by its depth
Superior is at the top. A face (nose, lips, eyes) visible in the bottom row means the
de-facing failed for this series. The pictures make visual QC of a whole cohort quick.

QC_roi_overlap*.png, one per structure set with ROIs in or near the de-faced region: the
same two views of the input, the de-faced region in blue, and every ROI that overlaps the
region or lies within ROI_NEAR_MM of it in its own colour: the part inside the region
filled, the outline of the whole ROI drawn. A legend with the ROI names and their overlap
is written into the picture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .font import HEIGHT as FONT_HEIGHT
from .font import render as render_text
from .masks import Axes, mask_parts
from .pii import TEXT_MIN_PIXELS, burned_in_text_pixels

PART_MIN_ML = 1.0       # a separate part of the face mask of this size ...
PART_MIN_GAP_MM = 10.0  # ... and this far from the main part is reported
SKIN_HU = -300.0        # surface threshold (the air fill is far below, soft-tissue fill above)
MARGIN_MM = 80.0        # head-feet range shown around the de-faced region
PIXEL_MM = 1.0
ROI_NEAR_MM = 10.0      # ROIs this close to the de-faced region are shown
MAX_ROIS = 12           # at most this many ROIs are drawn (most affected first)
REGION_RGB = (0.25, 0.55, 1.0)
ROI_RGB = [(1.0, 0.25, 0.25), (1.0, 0.85, 0.1), (0.2, 0.9, 0.3), (1.0, 0.5, 0.0), (0.9, 0.3, 0.9),
           (0.1, 0.9, 0.9), (1.0, 0.6, 0.7), (0.65, 0.4, 0.15), (0.6, 1.0, 0.6), (1.0, 1.0, 1.0),
           (0.7, 0.7, 1.0), (0.85, 0.65, 0.3)]
TEXT_SCALE = 2


def _oriented(vol: np.ndarray, axes: Axes) -> tuple[np.ndarray, tuple] | None:
    """vol with numpy axes (head-feet increasing superior, anterior, left-right)."""
    si, (ap, ap_inc) = axes.si_axis, axes.anterior
    if si == ap:
        return None
    lr = 3 - si - ap
    v = np.transpose(vol, (si, ap, lr))
    if not axes.superior:
        v = v[::-1]
    if not ap_inc:
        v = v[:, ::-1]
    return v, (axes.spacing[si], axes.spacing[ap], axes.spacing[lr])


def _to_mm(img: np.ndarray, spacing: tuple) -> np.ndarray:
    from scipy.ndimage import zoom
    factors = [max(s / PIXEL_MM, 1e-3) for s in spacing]
    return zoom(img.astype(np.float32), factors, order=1)


def _views(tissue: np.ndarray, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    s_si, s_ap, s_lr = spacing
    # side view: fraction of tissue along each left-right line (gives some depth cue)
    side = np.sqrt(tissue.mean(axis=2))[:, ::-1]                # (si, ap), face to the left
    # front view: depth of the first tissue voxel seen from the front, shaded (light from the front)
    n_ap = tissue.shape[1]
    hit = tissue[:, ::-1, :]                                    # index 0 = most anterior
    first = np.argmax(hit, axis=1).astype(np.float32)           # (si, lr)
    seen = hit.any(axis=1)
    depth = first * s_ap
    depth[~seen] = n_ap * s_ap
    g_si, g_lr = np.gradient(depth, s_si, s_lr)
    front = np.where(seen, 1.0 / np.sqrt(1.0 + g_si ** 2 + g_lr ** 2), 0.0)
    front *= np.where(seen, 1.0 - 0.6 * depth / (n_ap * s_ap), 0.0)
    return (_to_mm(side, (s_si, s_ap))[::-1], _to_mm(front, (s_si, s_lr))[::-1])   # superior at the top


def _grid(rows: list[list[np.ndarray]], gap: int = 4) -> np.ndarray:
    heights = [max(p.shape[0] for p in row) for row in rows]
    widths = [max(row[c].shape[1] for row in rows) for c in range(len(rows[0]))]
    out = np.full((sum(heights) + gap * (len(rows) - 1), sum(widths) + gap * (len(widths) - 1)), 0.5, np.float32)
    y = 0
    for row, h in zip(rows, heights):
        x = 0
        for panel, w in zip(row, widths):
            out[y:y + panel.shape[0], x:x + panel.shape[1]] = np.clip(panel, 0, 1)
            x += w + gap
        y += h + gap
    return out


def write_qc_picture(before: np.ndarray, after: np.ndarray, region: np.ndarray, axes: Axes, path: Path) -> bool:
    """Write the QC picture (see module docstring); False if the orientation is unclear."""
    import SimpleITK as sitk
    oriented = [_oriented(v, axes) for v in (before, after, region)]
    if any(o is None for o in oriented):
        return False
    (b, spacing), (a, _), (r, _) = oriented
    zs = np.flatnonzero(r.any(axis=(1, 2)))
    if zs.size:                                                 # head-feet range around the region
        margin = int(np.ceil(MARGIN_MM / spacing[0]))
        crop = slice(max(0, zs[0] - margin), min(len(b), zs[-1] + margin + 1))
        b, a = b[crop], a[crop]
    rows = [list(_views(v > SKIN_HU, spacing)) for v in (b, a)]
    picture = (_grid(rows) * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(picture), str(path))
    return True


def automatic_checks(mask_path: Path, defaced_hu: np.ndarray, si_axis: int, label: str) -> list[str]:
    """Flags for the report: a face mask in several parts (a false detection elsewhere would
    also remove everything in front of it), and possible burned-in text in the output."""
    flags = []
    parts = mask_parts(mask_path)
    apart = [(ml, gap) for ml, gap in parts[1:] if ml >= PART_MIN_ML and gap >= PART_MIN_GAP_MM]
    if apart:
        flags.append(f"{label}: the face mask has {len(apart)} separate part(s) away from the face (e.g. "
                     f"{apart[0][0]:.0f} ml, {apart[0][1]:.0f} mm away): check the QC picture for a false detection "
                     "(everything in front of it was removed as well)")
    n = burned_in_text_pixels(defaced_hu, si_axis)
    if n >= TEXT_MIN_PIXELS:
        flags.append(f"{label}: possible burned-in text or graphics ({n} bright pixels outside the body at the same "
                     "position in most slices; or metal outside the body): check the images")
    return flags


# ---------------------------------------------------------------------------
# ROIs in and near the de-faced region
# ---------------------------------------------------------------------------

def _projections(mask: np.ndarray, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """A 3D mask (oriented, cropped like the views) projected onto the side and front views."""
    s_si, s_ap, s_lr = spacing
    side = _to_mm(mask.any(axis=2)[:, ::-1], (s_si, s_ap))[::-1] > 0.5
    front = _to_mm(mask.any(axis=1), (s_si, s_lr))[::-1] > 0.5
    return side, front


def _paint(rgb: np.ndarray, where: np.ndarray, colour, alpha: float) -> None:
    rgb[where] = (1 - alpha) * rgb[where] + alpha * np.asarray(colour, np.float32)


def _outline(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, border_value=0)


def _legend(entries: list[tuple[tuple, str]], height: int) -> np.ndarray:
    """Colour swatches and texts, one line each, on a dark background (RGB)."""
    line = FONT_HEIGHT * TEXT_SCALE + 6
    width = 12 + max(render_text(text, TEXT_SCALE).shape[1] for _, text in entries) + 26
    out = np.full((max(height, line * len(entries) + 8), width, 3), 0.12, np.float32)
    for k, (colour, text) in enumerate(entries):
        y = 6 + k * line
        if colour is not None:
            out[y:y + FONT_HEIGHT * TEXT_SCALE, 8:8 + FONT_HEIGHT * TEXT_SCALE] = colour
        glyphs = render_text(text, TEXT_SCALE)
        x = 8 + FONT_HEIGHT * TEXT_SCALE + 8
        out[y:y + glyphs.shape[0], x:x + glyphs.shape[1]][glyphs] = 1.0
    return out


def roi_label(name: str, pct: float, ml: float, distance_mm) -> str:
    if pct > 0:
        return f"{name[:32]}: {pct:.0f}% in region ({ml:.1f} ml)"
    return f"{name[:32]}: {distance_mm} mm away"


def _volume(mask, shape: tuple) -> np.ndarray:
    """A 3D mask, also from {slice index: 2D mask} (only one ROI is held as a volume at a time)."""
    if isinstance(mask, np.ndarray):
        return mask
    out = np.zeros(shape, bool)
    for z, m in mask.items():
        out[z] = m
    return out


def write_roi_picture(before: np.ndarray, region: np.ndarray, rois: list[tuple[str, object]], axes: Axes,
                      path: Path, more: int = 0) -> bool:
    """Write QC_roi_overlap*.png (see module docstring). rois: (legend text, mask on the grid of
    `before`: a 3D array or {slice index: 2D mask}), most affected first. False if nothing is drawn."""
    import SimpleITK as sitk
    if not rois:
        return False
    oriented = _oriented(before, axes)
    if oriented is None:
        return False
    b, spacing = oriented
    r = _oriented(region, axes)[0]
    zs = np.flatnonzero(r.any(axis=(1, 2)))
    crop = slice(None)
    if zs.size:
        margin = int(np.ceil(MARGIN_MM / spacing[0]))
        crop = slice(max(0, zs[0] - margin), min(len(b), zs[-1] + margin + 1))
    side_bg, front_bg = _views(b[crop] > SKIN_HU, spacing)
    views = [np.repeat(np.clip(v, 0, 1)[..., None] * 0.8, 3, axis=2) for v in (side_bg, front_bg)]
    for view, proj in zip(views, _projections(r[crop], spacing)):
        _paint(view, proj, REGION_RGB, 0.35)
    entries = [(REGION_RGB, "de-faced region"), (None, "filled: ROI inside the region,"),
               (None, "outline: the whole ROI")]
    drawn = [(text, mask, ROI_RGB[k % len(ROI_RGB)]) for k, (text, mask) in enumerate(rois[:MAX_ROIS])]
    size = {id(mask): int(sum(m.sum() for m in mask.values())) if isinstance(mask, dict) else int(mask.sum())
            for _, mask, _ in drawn}
    for text, mask, colour in sorted(drawn, key=lambda e: -size[id(e[1])]):   # large first: small stay visible
        mask = _volume(mask, region.shape)
        m = _oriented(mask, axes)[0][crop]
        inside = _oriented(mask & region, axes)[0][crop]
        for view, whole, part in zip(views, _projections(m, spacing), _projections(inside, spacing)):
            _paint(view, part, colour, 0.6)
            view[_outline(whole)] = colour
    entries += [(colour, text) for text, _, colour in drawn]
    if more:
        entries.append((None, f"... and {more} more (see roi_overlap.csv)"))
    gap = np.full((max(v.shape[0] for v in views), 4, 3), 0.5, np.float32)
    panels = [np.pad(v, ((0, gap.shape[0] - v.shape[0]), (0, 0), (0, 0))) for v in views]
    picture = np.concatenate([panels[0], gap, panels[1], gap], axis=1)
    legend = _legend(entries, picture.shape[0])
    picture = np.pad(picture, ((0, legend.shape[0] - picture.shape[0]), (0, 0), (0, 0)))
    picture = np.concatenate([picture, legend], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray((np.clip(picture, 0, 1) * 255).round().astype(np.uint8), isVector=True),
                    str(path))
    return True
