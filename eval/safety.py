#!/usr/bin/env python3
"""RSS-style safety monitor and minimal-risk maneuver state machine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.sim import wrap_angle


@dataclass
class SafetyCommand:
    mode: str
    v_cap: float | None = None
    accel_cap: float | None = None
    reason: str = ""
    free_dist: float = np.inf
    unknown_dist: float = np.inf
    ttc: float = np.inf
    safe_dist: float = 0.0
    unknown_ratio: float = 0.0


class SafetyMonitor:
    def __init__(self, rho=1.0, b_max=3.0, margin=2.0,
                 caution_factor=1.4, unknown_caution=0.35,
                 unknown_stop=0.65, ttc_caution=3.0, ttc_stop=1.5,
                 recovery=1.5, mrm_recovery=0.8, use_unknown=True):
        self.rho = float(rho)
        self.b_max = float(b_max)
        self.margin = float(margin)
        self.caution_factor = float(caution_factor)
        self.unknown_caution = float(unknown_caution)
        self.unknown_stop = float(unknown_stop)
        self.ttc_caution = float(ttc_caution)
        self.ttc_stop = float(ttc_stop)
        self.recovery = float(recovery)
        self.mrm_recovery = float(mrm_recovery)
        self.use_unknown = bool(use_unknown)
        self.mode = "CRUISE"
        self._clear_since = None

    def reset(self):
        self.mode = "CRUISE"
        self._clear_since = None

    def _forward(self, ego, world, ref=None, max_dist=50.0, half_width=0.9):
        if ref is not None:
            s0, d0 = ref.project(ego.xy)
            ss = np.arange(s0 + 1.0, min(ref.length, s0 + max_dist), 0.5)
            if len(ss) < 2:
                return max_dist, max_dist, 1.0, 0.0
            p = ref.to_world(ss, np.full(len(ss), d0))
            yaw = ref.heading(ss)
            normal = np.column_stack([-np.sin(yaw), np.cos(yaw)])
            offs = np.array([-half_width, -half_width / 2, 0.0,
                             half_width / 2, half_width])
            pts = (p[:, None, :] + normal[:, None, :] * offs[None, :, None]).reshape(-1, 2)
            along = np.repeat(ss - s0, len(offs))
        else:
            X, Y = world.grid.centers()
            dx, dy = X - ego.x, Y - ego.y
            along = dx * np.cos(ego.yaw) + dy * np.sin(ego.yaw)
            lat = -dx * np.sin(ego.yaw) + dy * np.cos(ego.yaw)
            m = (along > 1.0) & (along <= max_dist) & (np.abs(lat) <= half_width)
            pts = np.column_stack([X[m], Y[m]])
            along = along[m]
        row, col = world.grid.cell_of(pts)
        ok = world.grid.inside(row, col)
        occ = np.zeros(len(pts), dtype=bool)
        unk = np.zeros(len(pts), dtype=bool)
        free = np.zeros(len(pts), dtype=bool)
        occ[ok] = world.layers["occupied_low"][row[ok], col[ok]]
        unk[ok] = world.layers["unknown"][row[ok], col[ok]]
        free[ok] = world.layers["free"][row[ok], col[ok]]
        d_occ = float(along[occ].min()) if occ.any() else max_dist
        d_unk = float(along[unk].min()) if unk.any() else max_dist
        if not self.use_unknown:
            d_unk = max_dist
            unk[:] = False
        return min(d_occ, d_unk), d_unk, float(free.mean()), float(unk.mean())

    def _ttc(self, ego, forecasts, max_dist=30.0):
        best = np.inf
        for f in forecasts:
            for prob, traj, sigma in f["forecast"].samples():
                if len(traj) < 2:
                    continue
                d = np.diff(traj, axis=0) / np.diff(f["forecast"].t)[:, None]
                v_agent = float(np.linalg.norm(d[0]))
                dx = traj[:, 0] - ego.x
                dy = traj[:, 1] - ego.y
                along = dx * np.cos(ego.yaw) + dy * np.sin(ego.yaw)
                lat = -dx * np.sin(ego.yaw) + dy * np.cos(ego.yaw)
                m = (along > 0.5) & (along < max_dist) & (np.abs(lat) < 1.6 + sigma)
                if not m.any():
                    continue
                gap = float(along[m].min())
                closing = max(ego.v - v_agent, 0.1)
                best = min(best, gap / closing)
        return best

    def update(self, ego, world, forecasts=None, t=0.0, ref=None):
        forecasts = forecasts or []
        free_d, unk_d, free_ratio, unk_ratio = self._forward(ego, world, ref)
        ttc = self._ttc(ego, forecasts)
        v_rel = ego.v
        for f in forecasts:
            v_agent = float(f.get("v", 0.0))
            px, py = f["forecast"].mean()[0]
            dx, dy = px - ego.x, py - ego.y
            along = dx * np.cos(ego.yaw) + dy * np.sin(ego.yaw)
            lat = -dx * np.sin(ego.yaw) + dy * np.cos(ego.yaw)
            if 0.0 < along < free_d + 5.0 and abs(lat) < 1.5:
                v_rel = min(v_rel, max(ego.v - v_agent, 0.0))
        safe = v_rel * self.rho + v_rel * v_rel / (2.0 * self.b_max) + self.margin

        hard = (free_d <= safe) or (ttc <= self.ttc_stop) or (unk_ratio >= self.unknown_stop)
        soft = (free_d <= self.caution_factor * safe) or (ttc <= self.ttc_caution) \
            or (unk_ratio >= self.unknown_caution)

        if hard:
            self.mode = "MRM"
            self._clear_since = None
        elif self.mode == "MRM":
            # Leave MRM as soon as the hard condition clears; a still-soft
            # condition degrades to CAUTION instead of latching forever.
            if self._clear_since is None:
                self._clear_since = t
            if t - self._clear_since >= self.mrm_recovery:
                self.mode = "CAUTION" if soft else "CRUISE"
                self._clear_since = None
        elif soft:
            self.mode = "CAUTION"
            self._clear_since = None
        elif self.mode == "CAUTION":
            if self._clear_since is None:
                self._clear_since = t
            if t - self._clear_since >= self.recovery:
                self.mode = "CRUISE"
                self._clear_since = None
        else:
            self.mode = "CRUISE"

        if self.mode == "MRM":
            return SafetyCommand("MRM", v_cap=0.0, accel_cap=-2.5, reason="stop",
                                 free_dist=free_d, unknown_dist=unk_d, ttc=ttc,
                                 safe_dist=safe, unknown_ratio=unk_ratio)
        if self.mode == "CAUTION":
            cap = max(0.0, 0.35 * (free_d - self.margin))
            return SafetyCommand("CAUTION", v_cap=cap, accel_cap=None,
                                 reason="slow", free_dist=free_d, unknown_dist=unk_d,
                                 ttc=ttc, safe_dist=safe, unknown_ratio=unk_ratio)
        return SafetyCommand("CRUISE", reason="clear", free_dist=free_d,
                             unknown_dist=unk_d, ttc=ttc, safe_dist=safe,
                             unknown_ratio=unk_ratio)
