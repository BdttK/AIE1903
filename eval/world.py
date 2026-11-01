#!/usr/bin/env python3
"""World-frame replay, three-state occupancy, and agent tracks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import (  # noqa: E402
    MAP_RES, SceneData, load_map, quat_to_rot,
)
from eval.core import Grid, rectangle_corners  # noqa: E402


@dataclass
class WorldConfig:
    res: float = 0.4
    margin: float = 30.0
    z_min: float = 0.25
    z_max: float = 12.0
    block_z_max: float = 2.5
    min_block_r: float = 2.5
    min_obstacle_r: float = 2.0
    n_bins: int = 1440


def yaw_of_pose(pose):
    rot = quat_to_rot(pose["rotation"])
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def pose_xy(pose):
    return np.asarray(pose["translation"], dtype=float)[:2]


def drop_self_points(points_ego, x_max=4.8, half_width=1.1, z_max=1.7):
    """Remove the ego body's own returns (bonnet/roof) from a sweep."""
    p = np.asarray(points_ego)
    body = (p[:, 0] > 0.5) & (p[:, 0] < x_max) & (np.abs(p[:, 1]) < half_width) \
        & (p[:, 2] < z_max)
    return p[~body]


def sweep_to_world(points_ego, pose):
    rot = quat_to_rot(pose["rotation"])
    trans = np.asarray(pose["translation"], dtype=float)
    return np.asarray(points_ego)[:, :3] @ rot.T + trans[:3]


def _visibility(points_world, ego_xy, n_bins, pad):
    """Angular free range seen from ``ego_xy`` (absolute bearings)."""
    r = np.hypot(points_world[:, 0] - ego_xy[0], points_world[:, 1] - ego_xy[1])
    theta = np.arctan2(points_world[:, 1] - ego_xy[1],
                       points_world[:, 0] - ego_xy[0]) % (2 * np.pi)
    bins = np.minimum((theta / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    limit = np.full(n_bins, np.inf)
    np.minimum.at(limit, bins, r)
    limit = ndimage.minimum_filter1d(limit, size=9, mode="wrap")
    finite = np.isfinite(limit)
    if not finite.all():
        limit[~finite] = float(limit[finite].max()) if finite.any() else 0.0
    return np.maximum(limit - pad, 0.0)


def build_layers_world(points_world, ego_xy, grid: Grid, cfg: WorldConfig):
    """Free / occupied / unknown around the simulated ego in world frame."""
    p = np.asarray(points_world, dtype=float)
    r = np.hypot(p[:, 0] - ego_xy[0], p[:, 1] - ego_xy[1])
    keep = (p[:, 2] > cfg.z_min) & (p[:, 2] < cfg.z_max) & (r > cfg.min_obstacle_r)
    occupied = np.zeros(grid.shape, dtype=bool)
    occupied_low = np.zeros(grid.shape, dtype=bool)
    low = keep & (p[:, 2] < cfg.block_z_max)
    for src, dst in ((keep, occupied), (low, occupied_low)):
        pts = p[src]
        if len(pts):
            row, col = grid.cell_of(pts[:, :2])
            ok = grid.inside(row, col)
            dst[row[ok], col[ok]] = True

    block = low & (r > cfg.min_block_r)
    limit = _visibility(p[block], ego_xy, cfg.n_bins, grid.res)

    X, Y = grid.centers()
    R = np.hypot(X - ego_xy[0], Y - ego_xy[1])
    T = np.arctan2(Y - ego_xy[1], X - ego_xy[0]) % (2 * np.pi)
    cb = np.minimum((T / (2 * np.pi) * cfg.n_bins).astype(int), cfg.n_bins - 1)
    free = (R <= limit[cb]) & ~occupied_low
    unknown = ~free & ~occupied_low
    return {"free": free, "occupied": occupied, "occupied_low": occupied_low,
            "unknown": unknown}


def map_drivable_world(sc: SceneData, grid: Grid, factor: int = 5, thresh: int = 127):
    if not sc.map_file:
        return np.ones(grid.shape, dtype=bool)
    arr, x_max, y_max = load_map(os.path.join(sc.root, sc.map_file), factor)
    scale = MAP_RES * factor
    X, Y = grid.centers()
    col = np.floor(X / scale).astype(int)
    row = np.floor((y_max - Y) / scale).astype(int)
    ok = (col >= 0) & (col < arr.shape[1]) & (row >= 0) & (row < arr.shape[0])
    out = np.zeros(grid.shape, dtype=bool)
    out[ok] = arr[row[ok], col[ok]] > thresh
    return out


class AgentTrack:
    def __init__(self, instance, category, size):
        self.instance = instance
        self.category = category
        self.size = np.asarray(size, dtype=float)
        self.t, self.xy, self.yaw, self.speed = [], [], [], []

    def add(self, t, xy, yaw, speed):
        self.t.append(float(t))
        self.xy.append(np.asarray(xy, dtype=float))
        self.yaw.append(float(yaw))
        self.speed.append(float(speed))

    def finalize(self):
        order = np.argsort(self.t)
        self.t = np.asarray(self.t)[order]
        self.xy = np.asarray(self.xy)[order]
        self.yaw = np.unwrap(np.asarray(self.yaw)[order])
        self.speed = np.asarray(self.speed)[order]
        return len(self.t) >= 3

    def at(self, t):
        t = float(np.clip(t, self.t[0], self.t[-1]))
        xy = np.array([np.interp(t, self.t, self.xy[:, 0]),
                       np.interp(t, self.t, self.xy[:, 1])])
        return xy, float(np.interp(t, self.t, self.yaw)), float(np.interp(t, self.t, self.speed))

    def history(self, t, window=2.0):
        m = (self.t <= t + 1e-6) & (self.t >= t - window)
        if not m.any():
            m = np.zeros(len(self.t), dtype=bool)
            m[:min(3, len(self.t))] = True
        return self.t[m], self.xy[m], self.yaw[m], self.speed[m]

    def corners(self, t):
        xy, yaw, _ = self.at(t)
        return rectangle_corners(xy, yaw, self.size[1], self.size[0])


class SceneReplay:
    """Replays logged ego motion and LiDAR sweeps for a scene."""

    def __init__(self, sc: SceneData):
        self.sc = sc
        self.tokens = sc.sample_tokens
        self.times = np.array([sc.samples[t]["timestamp"] for t in self.tokens],
                              dtype=float) / 1e6
        self.times -= self.times[0]
        self.poses = [sc.ego_pose(t) for t in self.tokens]
        self.real_xy = np.array([pose_xy(p) for p in self.poses])
        self.real_yaw = np.array([yaw_of_pose(p) for p in self.poses])
        self.real_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.real_xy, axis=0), axis=1))]
        self.real_v = np.r_[np.diff(self.real_s) / np.maximum(np.diff(self.times), 1e-3),
                            np.diff(self.real_s)[-1] / max(np.diff(self.times)[-1], 1e-3)]

        self.sweeps = []
        for rec in sc.scene_sample_data:
            if rec["filename"].split("/")[1] != "LIDAR_TOP":
                continue
            pose = sc.ego_poses[rec["ego_pose_token"]]
            self.sweeps.append({
                "filename": rec["filename"],
                "pose": pose,
                "xy": pose_xy(pose),
                "t": rec["timestamp"] / 1e6 - self.times[0],
                "calib": sc.calib[rec["calibrated_sensor_token"]],
            })
        self.sweeps.sort(key=lambda s: s["t"])
        self._cache = {}
        self.agents = self._collect_agents()

    def _collect_agents(self):
        raw = {}
        for i, token in enumerate(self.tokens):
            t = self.times[i]
            for ann in self.sc.ann_by_sample[token]:
                it = ann["instance_token"]
                if it not in raw:
                    raw[it] = AgentTrack(it, ann["_category"], ann["size"])
                rot = quat_to_rot(ann["rotation"])
                yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
                xy = np.asarray(ann["translation"], dtype=float)[:2]
                raw[it].add(t, xy, yaw, 0.0)

        route = self.real_xy
        out = []
        for track in raw.values():
            if not track.finalize():
                continue
            dt = np.diff(track.t)
            step = np.linalg.norm(np.diff(track.xy, axis=0), axis=1)
            speed = float(np.sum(step) / max(np.sum(dt), 1e-3))
            if speed < 0.5:
                continue
            dist = np.linalg.norm(track.xy[:, None, :] - route[None, :, :], axis=2).min()
            if dist > 30.0:
                continue
            v = step / np.maximum(dt, 1e-3)
            track.speed = np.r_[v, v[-1]]
            out.append(track)
        return out

    def sweep_points(self, idx):
        sw = self.sweeps[idx]
        key = sw["filename"]
        if key not in self._cache:
            pts = np.fromfile(os.path.join(self.sc.root, key),
                              dtype=np.float32).reshape(-1, 5)
            rot = quat_to_rot(sw["calib"]["rotation"])
            trans = np.asarray(sw["calib"]["translation"])
            self._cache[key] = pts[:, :3] @ rot.T + trans
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def sweep_near(self, ego_xy):
        d = np.linalg.norm(np.array([s["xy"] for s in self.sweeps]) - ego_xy, axis=1)
        return int(np.argmin(d))

    def agents_at(self, t):
        out = []
        for track in self.agents:
            if track.t[0] - 1e-6 <= t <= track.t[-1] + 1e-6:
                xy, yaw, v = track.at(t)
                out.append({"track": track, "xy": xy, "yaw": yaw,
                            "v": v, "corners": track.corners(t)})
        return out


class WorldModel:
    def __init__(self, sc: SceneData, replay: SceneReplay, cfg: WorldConfig | None = None):
        self.cfg = cfg or WorldConfig()
        self.sc = sc
        self.replay = replay
        lo = replay.real_xy.min(axis=0) - self.cfg.margin
        hi = replay.real_xy.max(axis=0) + self.cfg.margin
        self.grid = Grid(res=self.cfg.res, x_min=float(lo[0]), x_max=float(hi[0]),
                         y_min=float(lo[1]), y_max=float(hi[1]))
        self.map_free = map_drivable_world(sc, self.grid)
        self.layers = None
        self.dist_occ = None
        self.dist_unknown = None
        self.sweep_idx = None

    def update(self, ego_xy, points_ego=None):
        self.sweep_idx = self.replay.sweep_near(ego_xy)
        points = (self.replay.sweep_points(self.sweep_idx)
                  if points_ego is None else np.asarray(points_ego))
        points = drop_self_points(points)
        pose = self.replay.sweeps[self.sweep_idx]["pose"]
        world = sweep_to_world(points, pose)
        self.layers = build_layers_world(world, np.asarray(ego_xy), self.grid, self.cfg)
        self.dist_occ = ndimage.distance_transform_edt(~self.layers["occupied_low"]) * self.grid.res
        self.dist_unknown = ndimage.distance_transform_edt(~self.layers["unknown"]) * self.grid.res
        return self.layers

    def risk_at(self, xy):
        row, col = self.grid.cell_of(np.atleast_2d(xy))
        ok = self.grid.inside(row, col)
        d_occ = np.full(len(row), 0.0)
        unk = np.zeros(len(row), dtype=bool)
        d_occ[ok] = self.dist_occ[row[ok], col[ok]]
        unk[ok] = self.layers["unknown"][row[ok], col[ok]]
        return d_occ, unk
