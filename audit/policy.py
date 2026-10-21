#!/usr/bin/env python3
"""Longitudinal policies for the dynamic degradation demo."""

from __future__ import annotations

import numpy as np


def scan_forward(layers, grid, half_width=1.2, min_dist=2.5,
                 max_dist=30.0, step=0.5):
    """Scan a forward corridor in the ego frame (x forward)."""
    X, Y = grid.centers()
    corridor = (X >= min_dist) & (X <= max_dist) & (np.abs(Y) <= half_width)
    occ = layers["occupied"] & corridor
    unk = layers["unknown"] & corridor

    def first(mask):
        for d in np.arange(min_dist + step, max_dist + 1e-6, step):
            sel = corridor & (X <= d) & (X > d - step)
            if mask[sel].any():
                return float(d)
        return float(max_dist)

    n = max(int(corridor.sum()), 1)
    occ_d, unk_d = first(occ), first(unk)
    return {
        "occ_dist": occ_d,
        "unk_dist": unk_d,
        "free_dist": min(occ_d, unk_d),
        "unknown_ratio": float(unk.sum() / n),
        "occupied_ratio": float(occ.sum() / n),
    }


def speed_command(policy, scan, v, v_target, dt,
                  decel=3.0, accel=1.5):
    """Return the next speed for a policy.

    perception: brakes only for occupied cells; treats unknown as free.
    map: ignores live obstacles entirely.
    degradation_aware: brakes for occupied cells and caps speed by the
    distance to unknown space, so it slows down before it goes blind.
    """
    if policy == "map":
        return min(v_target, v + accel * dt)

    d_occ = scan["occ_dist"]
    d_brake = v * v / (2.0 * decel) + 1.5

    if policy == "perception":
        if d_occ <= d_brake:
            return max(0.0, v - decel * dt)
        return min(v_target, v + accel * dt)

    if policy == "degradation_aware":
        if d_occ <= d_brake:
            return max(0.0, v - decel * dt)
        d_free = scan["free_dist"]
        if d_free <= 4.0:
            return max(0.0, v - decel * dt)
        v_cap = max(0.0, 0.5 * (d_free - 2.0))
        return max(0.0, min(v_target, v_cap, v + accel * dt))

    raise ValueError(policy)
