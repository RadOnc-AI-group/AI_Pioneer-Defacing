"""
Face check (--face-check): does a face recognition model still find the patient's face?

The CT is rendered as a surface before and after de-facing, from the front and turned to
both sides. On every view RetinaFace looks for a face and FaceNet512 compares the face of
the de-faced render with the face of the original render.

This is a per-case proxy, not proof of anonymity: the models were not trained on CT
renders, and they compare the patient with themselves. (How well one patient can be found
among others is a cohort question.) Only scores are kept, never embeddings or renders.

Optional packages:  pip install -r requirements-reid.txt
Model weights are downloaded on first use; run once with internet access:
    python -m defacing.reid --download
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from .common import Options, PatientError, log
from .masks import Axes
from .qc import MARGIN_MM, SKIN_HU, _oriented

MODEL, DETECTOR = "Facenet512", "retinaface"
DEFAULT_THRESHOLD = 0.30    # DeepFace's cosine-distance threshold of FaceNet512: below it, "same person"
YAWS = (-30, 0, 30)         # views: degrees around the head-feet axis
RENDER_MM = 1.0
FLAG_RULES, FAIL_RULES = ("match", "any", "none"), ("never", "match", "any")


def check_face_check() -> str | None:
    """None if the optional packages are installed, else a message."""
    if importlib.util.find_spec("deepface") is None:
        return "--face-check needs optional packages: pip install -r requirements-reid.txt"
    return None


def render_views(hu: np.ndarray, region: np.ndarray, axes: Axes, skin_hu: float = SKIN_HU,
                 yaws: tuple[int, ...] = YAWS) -> dict[int, np.ndarray]:
    """{yaw: BGR image} of the skin surface around the de-faced region (superior at the top).
    skin_hu is the surface threshold, yaws the views."""
    from scipy.ndimage import label, rotate, zoom
    oriented, reg = _oriented(hu, axes), _oriented(region, axes)
    if oriented is None or reg is None:
        return {}
    vol, spacing = oriented
    zs = np.flatnonzero(reg[0].any(axis=(1, 2)))
    if zs.size:
        margin = int(np.ceil(MARGIN_MM / spacing[0]))
        vol = vol[max(0, zs[0] - margin):zs[-1] + margin + 1]
    tissue = zoom((vol > skin_hu).astype(np.float32), [s / RENDER_MM for s in spacing], order=1)
    labels, found = label(tissue > 0.5)
    if found > 1:                                            # the head only: no couch behind it
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        tissue = (labels == int(np.argmax(sizes))).astype(np.float32)
    views = {}
    for yaw in yaws:
        turned = tissue if yaw == 0 else rotate(tissue, yaw, axes=(1, 2), reshape=False, order=1)
        hit = turned[:, ::-1, :] > 0.5                       # index 0 = most anterior
        seen = hit.any(axis=1)
        depth = np.argmax(hit, axis=1).astype(np.float32)
        depth[~seen] = hit.shape[1]
        g_si, g_lr = np.gradient(depth)
        shade = np.clip(1.0 / np.sqrt(1.0 + g_si ** 2 + g_lr ** 2), 0, 1) * (1.0 - 0.5 * depth / hit.shape[1])
        gray = np.where(seen, shade, 0.0)[::-1]              # superior at the top
        views[yaw] = np.repeat((gray[..., None] * 255).round().astype(np.uint8), 3, axis=2)
    return views


def detect_face(image: np.ndarray) -> dict | None:
    """The most confident face of an image with its embedding, or None."""
    os.environ.setdefault("DEEPFACE_LOG_LEVEL", "40")
    from deepface import DeepFace
    try:
        faces = DeepFace.represent(image, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=True,
                                   align=True)
    except ValueError as e:
        if "could not be detected" in str(e):
            return None
        raise
    return max(faces, key=lambda f: float(f.get("face_confidence", 0.0))) if faces else None


def _cosine_distance(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(1.0 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def assess(before: np.ndarray, after: np.ndarray, region: np.ndarray, axes: Axes, opts: Options,
           label: str) -> tuple[dict, list[str]]:
    """
    Per-case result and CHECK lines. status is
      match         a face is found after de-facing and is close to the original face
      face_found    a face is found after de-facing, but it does not match (or no original to compare)
      no_face       no face is found after de-facing
      inconclusive  neither render shows a face, so the score says nothing
    score is the highest cosine similarity of a view (0 to 1); 0 if no face is found after
    de-facing, None if inconclusive.
    """
    views_before, views_after = render_views(before, region, axes), render_views(after, region, axes)
    if not views_before:
        return {"status": "inconclusive", "score": None, "faces_before": 0, "faces_after": 0,
                "note": "orientation unclear, no render"}, []
    found_before, found_after, best = 0, 0, None
    for yaw in YAWS:
        a, b = detect_face(views_before[yaw]), detect_face(views_after[yaw])
        found_before += a is not None
        found_after += b is not None
        if a is not None and b is not None:
            d = _cosine_distance(a["embedding"], b["embedding"])
            best = d if best is None else min(best, d)
    if not found_after:
        status, score = ("inconclusive", None) if not found_before else ("no_face", 0.0)
    elif best is not None and best < opts.face_check_threshold:
        status, score = "match", round(max(0.0, 1.0 - best), 3)
    else:
        status, score = "face_found", round(max(0.0, 1.0 - best), 3) if best is not None else None
    result = {"status": status, "score": score, "faces_before": found_before, "faces_after": found_after,
              "views": len(YAWS), "threshold": opts.face_check_threshold,
              "min_distance": None if best is None else round(best, 3)}

    detected = status in ("match", "face_found")
    text = (f"{label}: face check: a face was found after de-facing ({found_after} of {len(YAWS)} views)"
            + (f" and it matches the original (similarity {score:.2f})" if status == "match" else ""))
    if detected:
        log.warning(f"    ! {text}")
    if (opts.face_check_fail == "match" and status == "match") or (opts.face_check_fail == "any" and detected):
        raise PatientError(f"{text}; nothing was written for this patient (--face-check-fail {opts.face_check_fail})")
    flagged = status == "match" if opts.face_check_flag == "match" else (opts.face_check_flag == "any" and detected)
    return result, [f"{text}; check the QC picture"] if flagged else []


def main(argv: list[str]) -> int:
    if "--download" not in argv:
        print(__doc__)
        return 2
    problem = check_face_check()
    if problem:
        print(problem)
        return 1
    detect_face(np.zeros((160, 160, 3), np.uint8))           # downloads the detector; no face, so ...
    from deepface import DeepFace
    DeepFace.build_model(MODEL)                              # ... the recognition model separately
    print(f"Model weights are ready ({DETECTOR}, {MODEL}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
