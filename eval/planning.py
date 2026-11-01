#!/usr/bin/env python3
"""Risk-aware local planners: Frenet sampling and weighted A*."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from eval.sim import ReferencePath, VehicleParams, quintic_batch, wrap_angle

G = 9.81


@dataclass
class Plan:
    xy: np.ndarray
    v: np.ndarray
    t: np.ndarray
    cost: float = 0.0
    info: dict = field(default_factory=dict)


def _sample_times(horizon, dt):
    return np.arange(1, max(1, int(round(horizon / dt))) + 1) * dt


def _world_points(ref: ReferencePath, s, d):
    shape = np.shape(s)
    xy = ref.to_world(s.ravel(), d.ravel())
    return xy.reshape(shape + (2,))


def _agent_penalty(xy, times, forecasts):
    """Soft predicted-collision cost for one candidate trajectory."""
    if not forecasts:
        return 0.0
    cost = 0.0
    for f in forecasts:
        for prob, traj, sigma in f["forecast"].samples():
            if len(traj) == 0:
                continue
            px = np.interp(times, f["forecast"].t, traj[:, 0])
            py = np.interp(times, f["forecast"].t, traj[:, 1])
            dist = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
            margin = 1.6 + sigma
            cost += prob * float(np.sum(np.clip(1.0 - dist / margin, 0.0, 1.0) ** 2))
    return cost


class FrenetPlanner:
    """Quintic lateral/longitudinal sampling with a risk-aware cost."""

    def __init__(self, params: VehicleParams | None = None, horizon: float = 3.0,
                 dt: float = 0.2, lateral_offsets=(-3.5, -1.75, 0.0, 1.75, 3.5),
                 unknown_weight: float = 1.0, v_max: float = 10.0):
        self.p = params or VehicleParams()
        self.horizon = horizon
        self.dt = dt
        self.lateral = tuple(lateral_offsets)
        self.unknown_weight = float(unknown_weight)
        self.v_max = float(v_max)

    def plan(self, ego, ref: ReferencePath, world, forecasts=None) -> Plan:
        forecasts = forecasts or []
        s0, d0 = ref.project(ego.xy)
        yaw_ref = float(ref.heading(s0))
        d_dot = ego.v * np.sin(wrap_angle(ego.yaw - yaw_ref))
        v_s = max(ego.v * np.cos(wrap_angle(ego.yaw - yaw_ref)), 0.0)
        s0 = float(np.clip(s0, 0.0, ref.length - 0.5))

        v_limits = [float(self.p.mu * G / max(abs(float(ref.curvature(s0))), 1e-3)) ** 0.5]
        v_targets = [0.0, max(0.0, ego.v - 2.0), ego.v,
                     min(self.v_max, max(ego.v + 2.0, 2.0))]
        v_targets = sorted({round(float(np.clip(v, 0.0, min(self.v_max, v_limits[0]))), 2)
                            for v in v_targets})
        d_targets = [float(np.clip(d, -6.0, 6.0)) for d in self.lateral]

        S = len(d_targets) * len(v_targets)
        s_p0 = np.full(S, s0)
        s_v0 = np.full(S, v_s)
        s_a0 = np.zeros(S)
        d_p0 = np.full(S, d0)
        d_v0 = np.full(S, d_dot)
        s_p1 = np.empty(S)
        s_v1 = np.empty(S)
        d_p1 = np.empty(S)
        for i, dv in enumerate(d_targets):
            for j, vv in enumerate(v_targets):
                k = i * len(v_targets) + j
                s_p1[k] = s0 + 0.5 * (v_s + vv) * self.horizon
                s_v1[k] = vv
                d_p1[k] = dv

        t = _sample_times(self.horizon, self.dt)
        S_t, s_v, s_a = quintic_batch(s_p0, s_v0, s_a0, s_p1, s_v1,
                                      np.zeros(S), self.horizon, t)
        D_t, d_v, d_a = quintic_batch(d_p0, d_v0, np.zeros(S), d_p1,
                                      np.zeros(S), np.zeros(S), self.horizon, t)
        S_t = np.maximum(S_t, 0.0)
        xy = _world_points(ref, S_t, D_t)                      # (H, S, 2)
        speed = np.hypot(s_v, d_v)
        alat = d_a + ref.curvature(S_t) * s_v ** 2

        H = len(t)
        costs = np.zeros(S)
        feasible = np.ones(S, dtype=bool)
        d_occ, unk = world.risk_at(xy.reshape(-1, 2))
        d_occ = d_occ.reshape(H, S)
        unk = unk.reshape(H, S)
        feasible &= d_occ.min(axis=0) > 0.35
        costs += 40.0 * np.sum(np.clip(1.0 - d_occ / 2.0, 0.0, 1.0) ** 2, axis=0)

        row, col = world.grid.cell_of(xy.reshape(-1, 2))
        ok = world.grid.inside(row, col)
        d_unk = np.zeros(len(row))
        d_unk[ok] = world.dist_unknown[row[ok], col[ok]]
        d_unk = d_unk.reshape(H, S)
        costs += self.unknown_weight * (8.0 * unk.mean(axis=0)
                                        + 5.0 * np.clip(1.0 - d_unk.min(axis=0) / 3.0, 0.0, 1.0))

        row, col = world.grid.cell_of(xy.reshape(-1, 2))
        ok = world.grid.inside(row, col)
        drivable = np.zeros(len(row), dtype=bool)
        drivable[ok] = world.map_free[row[ok], col[ok]]
        costs += 2.0 * (1.0 - drivable.reshape(H, S).mean(axis=0))

        for k in range(S):
            costs[k] += _agent_penalty(xy[:, k], t, forecasts)
        costs += 0.08 * np.sum(np.clip(np.abs(alat) - 1.5, 0.0, None) ** 2, axis=0)
        costs += 0.02 * np.sum(np.diff(alat, axis=0) ** 2, axis=0)
        costs += 0.05 * np.sum(np.clip(np.abs(s_a) - 1.5, 0.0, None) ** 2, axis=0)
        costs -= 0.03 * S_t[-1] + 0.2 * np.abs(speed[-1] - v_s)

        best = int(np.argmin(np.where(feasible, costs, np.inf)))
        if not feasible[best]:
            return self._stop(ego, ref, s0, d0, d_dot, t)

        path = np.vstack([ego.xy[None, :], xy[:, best]])
        speeds = np.r_[ego.v, speed[:, best]]
        return Plan(xy=path, v=speeds, t=np.r_[0.0, t], cost=float(costs[best]),
                    info={"mode": "frenet", "s": S_t[:, best], "d": D_t[:, best],
                          "samples": xy, "costs": costs, "feasible": feasible})

    def _stop(self, ego, ref, s0, d0, d_dot, t):
        s_pos, s_vel, _ = quintic_batch(np.array([s0]), np.array([ego.v]), np.zeros(1),
                                        np.array([s0 + ego.v * self.horizon * 0.5]),
                                        np.zeros(1), np.zeros(1), self.horizon, t)
        d_pos, d_vel, _ = quintic_batch(np.array([d0]), np.array([d_dot]), np.zeros(1),
                                        np.array([d0]), np.zeros(1), np.zeros(1),
                                        self.horizon, t)
        xy = _world_points(ref, s_pos, d_pos)
        path = np.vstack([ego.xy[None, :], xy])
        speeds = np.r_[ego.v, np.maximum(0.0, np.hypot(s_vel.ravel(), d_vel.ravel()))]
        return Plan(xy=path, v=speeds, t=np.r_[0.0, t], cost=np.inf,
                    info={"mode": "stop", "fallback": True})


class RiskAStarPlanner:
    """Weighted A* on the three-state grid; used as the planning baseline."""

    def __init__(self, params: VehicleParams | None = None, unknown_weight: float = 1.0,
                 horizon_m: float = 30.0, inflate: float = 1.1):
        self.p = params or VehicleParams()
        self.unknown_weight = float(unknown_weight)
        self.horizon_m = float(horizon_m)
        self.inflate = float(inflate)

    def _cost_grid(self, world, forecasts, ref=None):
        cost = np.ones(world.grid.shape)
        cost[world.layers["occupied_low"]] = np.inf
        cost[world.layers["unknown"]] = 1.0 + 3.0 * self.unknown_weight
        cost[~world.map_free] += 2.0
        if ref is not None:
            row, col = world.grid.cell_of(ref.xy)
            ok = world.grid.inside(row, col)
            route = np.zeros(world.grid.shape, dtype=bool)
            route[row[ok], col[ok]] = True
            d_route = ndimage.distance_transform_edt(~route) * world.grid.res
            cost += 0.25 * np.minimum(d_route, 8.0)
        if forecasts:
            cells = []
            for f in forecasts:
                for prob, traj, _ in f["forecast"].samples():
                    row, col = world.grid.cell_of(traj)
                    ok = world.grid.inside(row, col)
                    cells.append(np.column_stack([row[ok], col[ok]]))
            if cells:
                mask = np.zeros(world.grid.shape, dtype=bool)
                c = np.vstack(cells)
                mask[c[:, 0], c[:, 1]] = True
                d = ndimage.distance_transform_edt(~mask) * world.grid.res
                cost += 4.0 * np.exp(-d / 2.5)
        return cost

    def _search(self, cost, start, goal, weight=1.3):
        ny, nx = cost.shape
        start = self._free_cell(cost, start)
        goal = self._free_cell(cost, goal)
        if start == goal:
            return [start], [start]
        nbr = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]

        def h(r, c):
            dr, dc = abs(r - goal[0]), abs(c - goal[1])
            return max(dr, dc) + (1.4142 - 1.0) * min(dr, dc)

        g = {start: 0.0}
        parent = {}
        heap = [(weight * h(*start), 0.0, start)]
        visited = []
        closed = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            visited.append(cur)
            if cur == goal:
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                return path[::-1], visited
            r, c = cur
            for dr, dc, step in nbr:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= ny or nc < 0 or nc >= nx or not np.isfinite(cost[nr, nc]):
                    continue
                ng = gc + step * float(cost[nr, nc])
                if ng < g.get((nr, nc), np.inf):
                    g[(nr, nc)] = ng
                    parent[(nr, nc)] = cur
                    heapq.heappush(heap, (ng + weight * h(nr, nc), ng, (nr, nc)))
        return None, visited

    @staticmethod
    def _free_cell(cost, cell):
        ny, nx = cost.shape
        cell = np.asarray(cell).astype(int).ravel()
        r = int(np.clip(cell[0], 0, ny - 1))
        c = int(np.clip(cell[1], 0, nx - 1))
        if np.isfinite(cost[r, c]):
            return (r, c)
        d, (ir, ic) = ndimage.distance_transform_edt(np.isfinite(cost),
                                                     return_indices=True)
        return int(ir[r, c]), int(ic[r, c])

    def plan(self, ego, ref: ReferencePath, world, forecasts=None) -> Plan:
        forecasts = forecasts or []
        s0, _ = ref.project(ego.xy)
        s_goal = min(ref.length - 1.0, s0 + self.horizon_m)
        goal_xy = ref.point(s_goal)
        cost = self._cost_grid(world, forecasts, ref)
        start = world.grid.cell_of(ego.xy)
        goal = world.grid.cell_of(goal_xy)
        cells, visited = self._search(cost, start, goal)
        if cells is None or len(cells) < 2:
            return self._stop(ego)

        rows = np.array([c[0] for c in cells])
        cols = np.array([c[1] for c in cells])
        X, Y = world.grid.centers()
        raw = np.column_stack([X[rows, cols], Y[rows, cols]])
        path = self._resample(self._smooth(raw), spacing=0.5)
        d_occ, _ = world.risk_at(path)
        if d_occ.min() < 0.55:
            path = self._resample(raw, spacing=0.5)
        v = self._speed(path, ego.v)
        dt = np.diff(path, axis=0) / np.maximum(v[1:, None], 1e-3)
        t = np.r_[0.0, np.cumsum(dt[:, 0] if dt.ndim > 1 else dt)]
        return Plan(xy=path, v=v, t=t, cost=float(cost[rows, cols].sum()),
                    info={"mode": "astar", "visited": visited,
                          "start": start, "goal": goal, "samples": path})

    @staticmethod
    def _smooth(path, window=5, iterations=2):
        out = path.copy()
        half = window // 2
        for _ in range(iterations):
            for i in range(half, len(out) - half):
                out[i] = out[i - half:i + half + 1].mean(axis=0)
        return out

    @staticmethod
    def _resample(path, spacing=0.5):
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] < spacing:
            return path
        sn = np.arange(0.0, s[-1], spacing)
        return np.column_stack([np.interp(sn, s, path[:, 0]),
                                np.interp(sn, s, path[:, 1])])

    def _speed(self, path, v0):
        d = np.diff(path, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        kappa = np.abs(np.diff(np.unwrap(yaw))) / np.maximum(
            np.linalg.norm(d[:-1], axis=1), 1e-3)
        kappa = np.r_[kappa, kappa[-1], kappa[-1]]
        limit = np.sqrt(self.p.mu * G / np.maximum(kappa, 1e-3))
        return np.minimum(np.minimum(v0 + 2.0, 10.0), limit)

    def _stop(self, ego):
        n = 8
        return Plan(xy=np.tile(ego.xy, (n, 1)),
                    v=np.linspace(ego.v, 0.0, n),
                    t=np.linspace(0.0, max(2.0, ego.v / 2.5), n),
                    cost=np.inf, info={"mode": "stop", "fallback": True})
