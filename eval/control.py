#!/usr/bin/env python3
"""Path-tracking controllers: sampling-based MPPI and a pure-pursuit baseline."""

from __future__ import annotations

import numpy as np

from eval.sim import VehicleParams, rollout_kinematic, wrap_angle


def _reference(plan, times):
    t = np.maximum.accumulate(plan.t)
    x = np.interp(times, t, plan.xy[:, 0])
    y = np.interp(times, t, plan.xy[:, 1])
    v = np.interp(times, t, plan.v)
    return np.column_stack([x, y]), v


class MPPIController:
    """Sampling MPC over the kinematic model with grid-based risk costs."""

    def __init__(self, params: VehicleParams | None = None, horizon: int = 25,
                 samples: int = 192, temperature: float = 2.0, seed: int = 0):
        self.p = params or VehicleParams()
        self.horizon = int(horizon)
        self.samples = int(samples)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.base = None
        self.prev = np.zeros(2)
        self.sigma = np.array([0.8, 0.15])
        self.bounds = np.array([[-self.p.max_decel, self.p.max_accel],
                                [-self.p.max_steer, self.p.max_steer]])

    def _rollout_cost(self, ego, plan, world, ref_xy, ref_v, accel, steer):
        x, y, yaw, v = rollout_kinematic(ego.x, ego.y, ego.yaw, ego.v,
                                         accel, steer, self.p)
        dx = x - ref_xy[None, :, 0]
        dy = y - ref_xy[None, :, 1]
        cost = 2.5 * np.sum(dx ** 2 + dy ** 2, axis=1)
        cost += 0.6 * np.sum((v - ref_v[None, :]) ** 2, axis=1)

        pts = np.column_stack([x.ravel(), y.ravel()])
        row, col = world.grid.cell_of(pts)
        inside = world.grid.inside(row, col)
        d_occ = np.zeros(len(pts))
        unk = np.zeros(len(pts), dtype=bool)
        d_occ[inside] = world.dist_occ[row[inside], col[inside]]
        unk[inside] = world.layers["unknown"][row[inside], col[inside]]
        cost += 45.0 * np.sum(np.clip(1.5 - d_occ.reshape(accel.shape), 0.0, None) ** 2, axis=1)
        cost += 25.0 * np.sum(unk.reshape(accel.shape), axis=1)
        cost += 0.08 * np.sum(accel ** 2 + 2.0 * steer ** 2, axis=1)
        da = np.diff(np.column_stack([np.full(len(accel), self.prev[0]), accel]), axis=1)
        ds = np.diff(np.column_stack([np.full(len(steer), self.prev[1]), steer]), axis=1)
        cost += 0.3 * np.sum(da ** 2, axis=1) + 0.6 * np.sum(ds ** 2, axis=1)
        cost -= 0.05 * (x[:, -1] - ego.x) * np.cos(ego.yaw) - 0.05 * (y[:, -1] - ego.y) * np.sin(ego.yaw)
        return cost

    def reset(self):
        self.base = None
        self.prev = np.zeros(2)

    def step(self, ego, plan, world):
        if self.base is None:
            self.base = np.tile([0.0, 0.0], (self.horizon, 1))
        times = np.arange(1, self.horizon + 1) * self.p.dt
        ref_xy, ref_v = _reference(plan, times)

        noise = self.rng.normal(0.0, 1.0, (self.samples, self.horizon, 2)) * self.sigma
        ctrl = self.base[None, :, :] + noise
        ctrl[:, :, 0] = np.clip(ctrl[:, :, 0], self.bounds[0, 0], self.bounds[0, 1])
        ctrl[:, :, 1] = np.clip(ctrl[:, :, 1], self.bounds[1, 0], self.bounds[1, 1])

        cost = self._rollout_cost(ego, plan, world, ref_xy, ref_v,
                                  ctrl[:, :, 0], ctrl[:, :, 1])
        w = np.exp(-(cost - cost.min()) / self.temperature)
        w = w / max(w.sum(), 1e-12)
        best = np.tensordot(w, ctrl, axes=(0, 0))

        accel, steer = float(best[0, 0]), float(best[0, 1])
        self.prev = np.array([accel, steer])
        self.base = np.vstack([best[1:], best[-1:]])
        return accel, steer


class PurePursuitController:
    """Lookahead path tracking plus a simple speed controller."""

    def __init__(self, params: VehicleParams | None = None, lookahead: float = 4.0,
                 gain: float = 1.2):
        self.p = params or VehicleParams()
        self.lookahead = float(lookahead)
        self.gain = float(gain)

    def reset(self):
        pass

    def step(self, ego, plan, world):
        ld = np.clip(1.5 + 0.6 * ego.v, 3.0, 8.0)
        d = np.hypot(plan.xy[:, 0] - ego.x, plan.xy[:, 1] - ego.y)
        idx = int(np.argmin(d))
        target = None
        for i in range(idx, len(plan.xy)):
            if np.hypot(plan.xy[i, 0] - ego.x, plan.xy[i, 1] - ego.y) >= ld:
                target = plan.xy[i]
                break
        if target is None:
            target = plan.xy[-1]
        alpha = float(wrap_angle(np.arctan2(target[1] - ego.y, target[0] - ego.x)
                                 - ego.yaw))
        steer = np.arctan2(2.0 * self.p.wheelbase * np.sin(alpha), ld)
        v_ref = float(np.interp(ld, d[:len(plan.v)], plan.v)) if len(plan.v) else 0.0
        accel = np.clip(self.gain * (v_ref - ego.v), -self.p.max_decel, self.p.max_accel)
        return float(accel), float(np.clip(steer, -self.p.max_steer, self.p.max_steer))
