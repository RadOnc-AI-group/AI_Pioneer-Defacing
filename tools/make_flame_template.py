#!/usr/bin/env python3
"""
Convert a FLAME 2023 Open download (flame2023_Open.pkl) to the template used by --fill template.

Only the mean head (v_template) and the triangles (f) are kept, in millimetres. The pickle
is read with an allow-list, so a tampered file cannot run code, and chumpy is not needed.

    python tools/make_flame_template.py flame2023_Open.pkl flame_template.npz

Licence: FLAME 2023 Open is CC-BY-4.0 with additional terms of its authors, see the NOTICE
file this tool writes next to its output. Read them before sharing images made with it.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REST, FACE, NECK, EYE = 0, 1, 2, 3      # the labels of defacing.template

# Fractions of the head measured from the nose tip (H: to the top of the head, D: to the back)
BROW_FRACTION = 0.35        # above nose + this * H: forehead, part of the rest of the head
CHIN_FRACTION = 0.55        # below nose - this * H: neck
FACE_DEPTH_FRACTION = 0.45  # the face is the front part of the head, this share of D

NOTICE = """\
{name} is derived from FLAME 2023 Open.

Credit: FLAME, Max Planck Institute for Intelligent Systems (https://flame.is.tue.mpg.de).
Licence: Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/) with
the additional terms of the FLAME model licence (https://flame.is.tue.mpg.de/modellicense.html),
which among others do not allow using the model to create fake, libelous, misleading or defamatory
content, apart from analyses in peer-reviewed scientific research. Read them before sharing
images made with this template.

Changes made: only the mean shape (v_template) and the triangles were kept, converted from metres
to millimetres and to (left-right, anterior, superior) axes; each vertex was given a region label
by a geometric rule; shape, expression and pose spaces were dropped.

Cite the FLAME paper when reporting results: Li T, Bolkart T, Black MJ, Li H, Romero J. Learning a
model of facial shape and expression from 4D scans. ACM Transactions on Graphics 36(6), 2017.
"""


class _Stub:
    """Stands in for chumpy and scipy objects of the pickle, which are not needed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["state"] = state


_NUMPY = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
          ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
          ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
          ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode")}


class _AllowList(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _NUMPY:
            return super().find_class(module, name)
        if module in ("copy_reg", "copyreg") and name == "_reconstructor":
            return super().find_class(module, name)
        if module == "__builtin__" and name == "object" or module == "builtins" and name == "object":
            return object
        if module.split(".")[0] in ("chumpy", "scipy"):
            return _Stub
        raise pickle.UnpicklingError(f"not allowed in a FLAME file: {module}.{name}")


def _array(value) -> np.ndarray:
    """The numbers of a value that may be a numpy array or a chumpy object."""
    if isinstance(value, np.ndarray):
        return value
    state = getattr(value, "state", None) or getattr(value, "__dict__", {})
    if isinstance(state, dict):
        for key in ("x", "a", "r"):
            if isinstance(state.get(key), np.ndarray):
                return state[key]
    raise ValueError(f"could not read the numbers of a {type(value).__name__}")


def read_flame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices in mm as left-right, anterior, superior; triangles) of a FLAME pickle."""
    with open(path, "rb") as fh:
        model = _AllowList(fh, encoding="latin1").load()
    vertices = np.asarray(_array(model["v_template"]), float)
    triangles = np.asarray(_array(model["f"]), np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("v_template and f must have three columns")
    height = float(np.ptp(vertices[:, 1]))
    if not 0.1 < height < 0.6:
        raise ValueError(f"expected a head in metres, found a height of {height:g}")
    return vertices[:, [0, 2, 1]] * 1000.0, triangles


def label_vertices(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """REST, FACE, NECK or EYE per vertex. Eyes are the separate parts of the mesh."""
    n = len(vertices)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, parts = connected_components(graph, directed=False)
    head = parts == int(np.argmax(np.bincount(parts)))
    tip = vertices[head][np.argmax(vertices[head, 1])]
    height = float(vertices[head, 2].max() - tip[2])
    depth = float(tip[1] - vertices[head, 1].min())
    ap, si = vertices[:, 1], vertices[:, 2]
    chin, brow = tip[2] - CHIN_FRACTION * height, tip[2] + BROW_FRACTION * height
    labels = np.full(n, REST, np.uint8)
    labels[(ap > tip[1] - FACE_DEPTH_FRACTION * depth) & (si > chin) & (si < brow)] = FACE
    labels[si <= chin] = NECK
    labels[~head] = EYE
    return labels


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flame", type=Path, help="flame2023_Open.pkl")
    p.add_argument("output", type=Path, help="the template to write (.npz)")
    args = p.parse_args(argv)
    vertices, triangles = read_flame(args.flame)
    labels = label_vertices(vertices, triangles)
    np.savez_compressed(args.output, vertices=vertices.astype(np.float32), triangles=triangles.astype(np.int32), labels=labels,
                        source=np.array("FLAME 2023 Open"))
    notice = args.output.with_name(args.output.name + ".NOTICE.txt")
    notice.write_text(NOTICE.format(name=args.output.name), encoding="utf-8")
    counts = {name: int((labels == value).sum()) for name, value in (("rest", REST), ("face", FACE), ("neck", NECK), ("eye", EYE))}
    print(f"Wrote {args.output} ({len(vertices)} vertices, {len(triangles)} triangles; {counts}) and {notice.name}")
    print("Head size in mm (left-right, anterior, superior):", np.ptp(vertices, axis=0).round(1).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
