#!/usr/bin/env python3
"""Unit tests for the closed-loop stack (data-free)."""

import os
import sys

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.control import MPPIController  # noqa: E402
from eval.core import Grid  # noqa: E402
from eval.planning import FrenetPlanner, Plan, RiskAStarPlanner  # noqa: E402
from eval.predict import IMMPredictor  # noqa: E402
from eval.safety import SafetyMonitor  # noqa: E402
from eval.sim import (  # noqa: E402
    DynamicBicycle, EgoState, ReferencePath, VehicleParams, quintic_batch,
)
from eval.closed_loop import (  # noqa: E402
    polygon_distance, ego_corners, _shift_plan, _box_points,
)


class FakeWorld:
    def __init__(self, grid, occupied=None, unknown=None):
        shape = grid.shape
        self.grid = grid
        occ = np.zeros(shape, dtype=bool) if occupied is None else occupied
        unk = np.zeros(shape, dtype=bool) if unknown is None else unknown
        self.layers = {"occupied": occ, "occupied_low": occ, "unknown": unk,
                       "free": ~occ & ~unk}
        self.map_free = np.ones(shape, dtype=bool)
        self.dist_occ = ndimage.distance_transform_edt(~occ) * grid.res
        self.dist_unknown = ndimage.distance_transform_edt(~unk) * grid.res

    def risk_at(self, xy):
        row, col = self.grid.cell_of(np.atleast_2d(xy))
        ok = self.grid.inside(row, col)
        d = np.zeros(len(row))
        u = np.zeros(len(row), dtype=bool)
        d[ok] = self.dist_occ[row[ok], col[ok]]
        u[ok] = self.layers["unknown"][row[ok], col[ok]]
        return d, u


def _straight_ref(length=60.0, y=0.0):
    return ReferencePath(np.column_stack([np.linspace(0, length, int(length * 4) + 1),
                                          np.full(int(length * 4) + 1, y)]))


def test_dynamic_bicycle_steady_state():
    p = VehicleParams()
    car = DynamicBicycle(0, 0, 0, 8.0, p)
    for _ in range(40):
        car.step(0.0, 0.05)
    expected = 8.0 / p.wheelbase * np.tan(0.05)
    assert abs(car.r - expected) < 0.02
    assert abs(8.0 * car.r) < p.mu * 9.81
    assert abs(car.state.v - 8.0) < 0.3


def test_quintic_reaches_boundary():
    t = np.linspace(0, 3, 7)
    p, v, a = quintic_batch(np.array([0.0]), np.array([2.0]), np.array([0.5]),
                            np.array([20.0]), np.array([5.0]), np.array([0.0]),
                            3.0, t)
    assert np.allclose([p[0, 0], v[0, 0], a[0, 0]], [0.0, 2.0, 0.5])
    assert np.allclose([p[-1, 0], v[-1, 0], a[-1, 0]], [20.0, 5.0, 0.0])


def test_frenet_roundtrip():
    ref = ReferencePath(np.column_stack([np.linspace(0, 50, 101),
                                         np.linspace(0, 10, 101)]))
    xy = ref.to_world(np.array([10.0, 25.0]), np.array([1.5, -2.0]))
    s, d = ref.project(xy)
    assert np.allclose(s, [10.0, 25.0], atol=0.15)
    assert np.allclose(d, [1.5, -2.0], atol=0.15)


def test_astar_avoids_occupied():
    g = Grid(res=0.5, x_min=-5, x_max=15, y_min=-5, y_max=5)
    occ = np.zeros(g.shape, dtype=bool)
    wall = int((5.0 - g.x_min) / g.res)
    occ[:, wall] = True
    occ[int((4.0 - g.y_min) / g.res), wall] = False
    world = FakeWorld(g, occupied=occ)
    planner = RiskAStarPlanner()
    ego = EgoState(-2.0, 0.0, 0.0, 5.0)
    plan = planner.plan(ego, _straight_ref(20.0), world, [])
    assert plan.info["mode"] == "astar"
    assert len(plan.xy) > 3
    d, _ = world.risk_at(plan.xy)
    assert d.min() > 0.2


def test_frenet_planner_swerves():
    g = Grid(res=0.5, x_min=0, x_max=40, y_min=-8, y_max=8)
    occ = np.zeros(g.shape, dtype=bool)
    x0, x1 = int(14 / g.res), int(20 / g.res)
    y0, y1 = int((-3.0 - g.y_min) / g.res), int((0.6 - g.y_min) / g.res)
    occ[y0:y1, x0:x1] = True
    world = FakeWorld(g, occupied=occ)
    ref = _straight_ref(36.0)
    ego = EgoState(0.0, 0.0, 0.0, 6.0)
    plan = FrenetPlanner().plan(ego, ref, world, [])
    assert plan.info.get("mode") == "frenet"
    mask = (plan.xy[:, 0] > 13.5) & (plan.xy[:, 0] < 20.5)
    if mask.any():
        assert plan.xy[mask, 1].min() > 0.9
    d, _ = world.risk_at(plan.xy)
    assert d.min() > 0.3


def test_unknown_weight_is_used():
    g = Grid(res=0.5, x_min=0, x_max=40, y_min=-8, y_max=8)
    unk = np.zeros(g.shape, dtype=bool)
    unk[:, int(18 / g.res):int(20 / g.res)] = True
    world = FakeWorld(g, unknown=unk)
    plan_high = RiskAStarPlanner(unknown_weight=5.0)._cost_grid(world, [])
    plan_low = RiskAStarPlanner(unknown_weight=0.0)._cost_grid(world, [])
    row = int((0 - g.y_min) / g.res)
    col = int(19 / g.res)
    assert plan_high[row, col] > plan_low[row, col]


def test_mppi_tracks_straight():
    g = Grid(res=0.5, x_min=-10, x_max=50, y_min=-10, y_max=10)
    world = FakeWorld(g)
    p = VehicleParams()
    car = DynamicBicycle(0, 0, 0, 8.0, p)
    ctrl = MPPIController(params=p, horizon=25, samples=96, seed=1)

    def make_plan(x0):
        xy = np.column_stack([x0 + np.linspace(0, 40, 81), np.zeros(81)])
        return Plan(xy=xy, v=np.full(81, 8.0), t=np.arange(81) * 0.0625)

    plan = make_plan(0.0)
    ys, vs = [], []
    for k in range(20):
        if k % 2 == 0:
            plan = make_plan(car.x)
        a, s = ctrl.step(car.state, plan, world)
        car.step(a, s)
        if k % 2 == 1:
            _shift_plan(plan, p.dt)
        ys.append(car.y)
        vs.append(car.state.v)
    assert max(abs(np.array(ys))) < 1.0
    assert min(vs) > 6.5


def test_mrm_triggers_and_recovers():
    g = Grid(res=0.5, x_min=-10, x_max=60, y_min=-10, y_max=10)
    unk = np.zeros(g.shape, dtype=bool)
    unk[:, int(8 / g.res):] = True
    world = FakeWorld(g, unknown=unk)
    ref = _straight_ref(50.0)
    ego = EgoState(0, 0, 0, 8.0)
    m = SafetyMonitor()
    cmd = m.update(ego, world, [], 0.0, ref=ref)
    assert cmd.mode == "MRM"

    clear = FakeWorld(g)
    slow = EgoState(0, 0, 0, 0.0)
    for t in [0.1, 0.8, 1.6, 2.0]:
        cmd = m.update(slow, clear, [], t, ref=ref)
    assert cmd.mode == "CRUISE"


def test_blind_monitor_ignores_unknown():
    g = Grid(res=0.5, x_min=-10, x_max=60, y_min=-10, y_max=10)
    unk = np.zeros(g.shape, dtype=bool)
    unk[:, int(8 / g.res):] = True
    world = FakeWorld(g, unknown=unk)
    ref = _straight_ref(50.0)
    ego = EgoState(0, 0, 0, 8.0)
    cmd = SafetyMonitor(use_unknown=False).update(ego, world, [], 0.0, ref=ref)
    assert cmd.mode == "CRUISE"


def test_imm_turn_accuracy():
    p = IMMPredictor()
    t = np.arange(0, 2.1, 0.5)
    yaw = 0.2 * t
    xy = np.column_stack([5 / 0.2 * np.sin(yaw), 5 / 0.2 * (1 - np.cos(yaw))])
    f = p.forecast(t, xy, yaw_hist=yaw, horizon=3.0, dt_pred=0.5)
    assert max(m[0] for m in f.modes) > 0.9
    true_end = np.array([5 / 0.2 * np.sin(0.2 * 5.0), 5 / 0.2 * (1 - np.cos(0.2 * 5.0))])
    assert np.linalg.norm(f.mean()[-1] - true_end) < 1.5


def test_box_points_height():
    pts = _box_points((0.0, 0.0), (1.0, 1.0), base_z=2.0)
    assert pts.shape[1] == 3
    assert np.isclose(pts[:, 2].min(), 2.4)
    assert np.isclose(pts[:, 2].max(), 3.4)


def test_polygon_distance():
    p = VehicleParams()
    a = ego_corners(EgoState(0, 0, 0, 0), p)
    b = ego_corners(EgoState(3, 0, 0, 0), p)
    assert polygon_distance(a, b) == 0.0
    c = ego_corners(EgoState(10, 0, 0, 0), p)
    assert polygon_distance(a, c) > 4.0
