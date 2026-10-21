#!/usr/bin/env python3
"""Unit tests for the audit core (data-free)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audit.core import (  # noqa: E402
    Grid, astar, build_layers, degrade, hazard_recall, obb_overlap,
    rectangle_corners,
)


def test_grid_roundtrip():
    g = Grid(res=0.5, x_min=-5, x_max=5, y_min=-5, y_max=5)
    assert g.shape == (20, 20)
    row, col = g.cell_of(np.array([[0.0, 0.0], [4.9, -4.9]]))
    assert (row[0], col[0]) == (10, 10)
    assert g.inside(row, col).all()


def test_obb_overlap():
    a = rectangle_corners((0, 0), 0.0, 4.0, 2.0)
    b = rectangle_corners((1, 0), 0.0, 4.0, 2.0)
    c = rectangle_corners((10, 0), 0.0, 4.0, 2.0)
    assert obb_overlap(a, b)
    assert not obb_overlap(a, c)


def test_astar_detour_and_blocked():
    blocked = np.zeros((20, 20), dtype=bool)
    blocked[:, 10] = True
    blocked[5, 10] = False
    path = astar(blocked, (10, 2), (10, 18))
    assert path is not None
    assert any(c == 10 and r == 5 for r, c in path)

    blocked[:, 10] = True
    assert astar(blocked, (10, 2), (10, 18)) is None


def test_degrade():
    pts = np.zeros((100, 3))
    pts[:, 0] = np.linspace(0, 99, 100)
    rng = np.random.default_rng(0)
    kept = degrade(pts, "range", 20.0, rng)
    assert len(kept) == 21
    dropped = degrade(pts, "drop", 0.5, rng)
    assert 20 <= len(dropped) <= 80
    noisy = degrade(pts, "noise", 0.1, np.random.default_rng(0))
    assert noisy.shape == pts.shape
    assert np.isfinite(noisy).all()


def test_layers_wall():
    g = Grid(res=1.0, x_min=-10, x_max=10, y_min=-10, y_max=10)
    xs = np.arange(0, 8.01, 0.05)
    ys = np.arange(-2, 2.01, 0.05)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    wall = np.c_[np.full_like(ys, 5.0), ys, np.ones_like(ys)]
    pts = np.vstack([ground, wall])
    layers = build_layers(pts, g, n_bins=720)
    row3, col3 = g.cell_of(np.array([3.0, 0.0]))
    row5, col5 = g.cell_of(np.array([5.0, 0.0]))
    row7, col7 = g.cell_of(np.array([7.0, 0.0]))
    assert layers["free"][row3, col3]
    assert layers["occupied"][row5, col5]
    assert layers["unknown"][row7, col7]


def test_hazard_recall():
    m1 = np.zeros((4, 4), dtype=bool)
    m1[0, 0] = m1[0, 1] = True
    m2 = np.zeros((4, 4), dtype=bool)
    m2[3, 3] = m2[3, 2] = True
    occ = np.zeros((4, 4), dtype=bool)
    occ[0, 0] = True
    hazards = [
        {"mask": m1, "ttc": 1.0, "d": 0.0},
        {"mask": m2, "ttc": 1.0, "d": 0.0},
    ]
    out = hazard_recall(hazards, occ, coverage_thresh=0.1)
    assert out["n_hazards"] == 2
    assert abs(out["recall"] - 0.5) < 1e-9
    assert abs(out["false_safe_rate"] - 0.5) < 1e-9
    assert out["n_urgent"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
