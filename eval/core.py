#!/usr/bin/env python3
"""Core BEV occupancy / free-space / planning utilities for the evaluation.

Coordinate convention: all planning is in the current ego frame (x forward,
y left, z up). Map sampling converts to global coordinates on demand.
"""

from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import (  # noqa: E402
    BOTTOM, MAP_RES, SceneData, load_map, quat_to_rot,
)


@dataclass
class Grid:
    res: float = 0.2
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0

    @property
    def shape(self):
        return (int(round((self.y_max - self.y_min) / self.res)),
                int(round((self.x_max - self.x_min) / self.res)))

    def centers(self):
        ny, nx = self.shape
        x = self.x_min + (np.arange(nx) + 0.5) * self.res
        y = self.y_min + (np.arange(ny) + 0.5) * self.res
        return np.meshgrid(x, y)  # X, Y each (ny, nx)

    def cell_of(self, xy):
        col = np.floor((np.asarray(xy)[..., 0] - self.x_min) / self.res).astype(int)
        row = np.floor((np.asarray(xy)[..., 1] - self.y_min) / self.res).astype(int)
        return row, col

    def inside(self, row, col):
        ny, nx = self.shape
        return (row >= 0) & (row < ny) & (col >= 0) & (col < nx)


def pose_arrays(pose):
    return quat_to_rot(pose["rotation"]), np.asarray(pose["translation"], dtype=float)


def ego_to_global(xy, pose):
    rot, trans = pose_arrays(pose)
    return np.asarray(xy) @ rot[:2, :2].T + trans[:2]


def global_to_ego(xy, pose):
    rot, trans = pose_arrays(pose)
    return (np.asarray(xy) - trans[:2]) @ rot[:2, :2]


def goal_in_ego(sc, token, horizon_m=25.0):
    cur = sc.ego_pose(token)
    _, trans = pose_arrays(cur)
    idx = sc.sample_tokens.index(token)
    prev = trans[:2]
    acc = 0.0
    target = prev
    for nxt in sc.sample_tokens[idx + 1:]:
        p = np.asarray(sc.ego_pose(nxt)["translation"], dtype=float)[:2]
        acc += float(np.linalg.norm(p - prev))
        prev, target = p, p
        if acc >= horizon_m:
            break
    return global_to_ego(target, cur)


def obstacle_mask(points, grid, z_min=0.25, z_max=12.0, min_r=2.0):
    r = np.hypot(points[:, 0], points[:, 1])
    m = (points[:, 2] > z_min) & (points[:, 2] < z_max) & (r > min_r)
    row, col = grid.cell_of(points[m, :2])
    ok = grid.inside(row, col)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[row[ok], col[ok]] = True
    return mask, points[m]


def free_space(points, grid, obs_points, n_bins=1440, range_pad=None):
    """Angular visibility profile: free up to the nearest obstacle per bearing."""
    ny, nx = grid.shape
    xy = points[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    theta = np.arctan2(xy[:, 1], xy[:, 0]) % (2 * np.pi)
    bins = np.minimum((theta / (2 * np.pi) * n_bins).astype(int), n_bins - 1)

    free_range = np.zeros(n_bins)
    np.maximum.at(free_range, bins, r)
    free_range = ndimage.maximum_filter1d(free_range, size=9, mode="wrap")
    obs_range = np.full(n_bins, np.inf)
    if obs_points is not None and len(obs_points):
        oxy = obs_points[:, :2]
        orr = np.hypot(oxy[:, 0], oxy[:, 1])
        oth = np.arctan2(oxy[:, 1], oxy[:, 0]) % (2 * np.pi)
        ob = np.minimum((oth / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
        np.minimum.at(obs_range, ob, orr)
    limit = np.minimum(free_range, obs_range)
    pad = grid.res if range_pad is None else range_pad
    limit = np.maximum(limit - pad, 0.0)

    X, Y = grid.centers()
    R = np.hypot(X, Y)
    T = np.arctan2(Y, X) % (2 * np.pi)
    cb = np.minimum((T / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    return R <= limit[cb]


def boxes_to_masks(boxes_ego, grid, inflate=0.0):
    """Return a per-box boolean mask list and a union mask."""
    ny, nx = grid.shape
    X, Y = grid.centers()
    pts = np.column_stack([X.ravel(), Y.ravel()])
    union = np.zeros(grid.shape, dtype=bool)
    masks = []
    for corners, category in boxes_ego:
        foot = corners[BOTTOM][:, :2]
        center = foot.mean(axis=0)
        if inflate > 0:
            v = foot - center
            n = np.linalg.norm(v, axis=1, keepdims=True)
            foot = center + v * (1.0 + inflate / np.maximum(n, 1e-6))
        inside = Path(foot, closed=True).contains_points(pts).reshape(ny, nx)
        masks.append((inside, category))
        union |= inside
    return masks, union


def map_drivable(sc, token, grid, factor=5, thresh=127, pose=None):
    arr, x_max, y_max = load_map(os.path.join(sc.root, sc.map_file), factor)
    scale = MAP_RES * factor
    X, Y = grid.centers()
    pose = sc.ego_pose(token) if pose is None else pose
    g = ego_to_global(np.column_stack([X.ravel(), Y.ravel()]), pose)
    col = np.floor(g[:, 0] / scale).astype(int)
    row = np.floor((y_max - g[:, 1]) / scale).astype(int)
    ok = (col >= 0) & (col < arr.shape[1]) & (row >= 0) & (row < arr.shape[0])
    out = np.zeros(X.size, dtype=bool)
    out[ok] = arr[row[ok], col[ok]] > thresh
    return out.reshape(grid.shape), ok.reshape(grid.shape)


def degrade(points, kind, param, rng):
    if kind == "clean":
        return points
    r = np.hypot(points[:, 0], points[:, 1])
    if kind == "range":
        return points[r <= float(param)]
    if kind == "drop":
        return points[rng.random(len(points)) >= float(param)]
    if kind == "noise":
        out = points.copy()
        xy = out[:, :2]
        r = np.hypot(xy[:, 0], xy[:, 1])
        th = np.arctan2(xy[:, 1], xy[:, 0])
        r = np.maximum(r + rng.normal(0.0, float(param), len(r)), 0.0)
        out[:, 0] = r * np.cos(th)
        out[:, 1] = r * np.sin(th)
        return out
    if kind == "combo":
        keep = points[r <= float(param[0])]
        return keep[rng.random(len(keep)) >= float(param[1])]
    raise ValueError(kind)


def build_layers(points, grid, z_min=0.25, n_bins=1440,
                 block_z_max=2.5, min_block_r=2.5, min_obstacle_r=2.0):
    occ, _ = obstacle_mask(points, grid, z_min=z_min, min_r=min_obstacle_r)
    occ_low, _ = obstacle_mask(points, grid, z_min=z_min, z_max=block_z_max,
                               min_r=min_obstacle_r)
    rr = np.hypot(points[:, 0], points[:, 1])
    block = ((points[:, 2] > z_min) & (points[:, 2] < block_z_max)
             & (rr > min_block_r))
    free = free_space(points, grid, points[block], n_bins=n_bins)
    free &= ~occ
    unknown = ~free & ~occ
    return {"free": free, "occupied": occ, "occupied_low": occ_low,
            "unknown": unknown}


def _nearest_free(blocked, cell):
    if not blocked[cell]:
        return cell
    dist, (ir, ic) = ndimage.distance_transform_edt(
        blocked, return_distances=True, return_indices=True)
    return (int(ir[cell]), int(ic[cell]))


def astar(blocked, start, goal, diag=True):
    ny, nx = blocked.shape
    start = _nearest_free(blocked, tuple(int(v) for v in start))
    goal = _nearest_free(blocked, tuple(int(v) for v in goal))
    if start == goal:
        return [start]
    if diag:
        nbr = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.41421356), (-1, 1, 1.41421356),
               (1, -1, 1.41421356), (1, 1, 1.41421356)]
    else:
        nbr = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

    def h(r, c):
        dr, dc = abs(r - goal[0]), abs(c - goal[1])
        if diag:
            return (max(dr, dc) + (1.41421356 - 1.0) * min(dr, dc))
        return dr + dc

    gscore = {start: 0.0}
    parent = {}
    heap = [(h(*start), 0.0, start)]
    closed = set()
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        r, c = cur
        for dr, dc, cost in nbr:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx or blocked[nr, nc]:
                continue
            ng = g + cost
            if ng < gscore.get((nr, nc), np.inf):
                gscore[(nr, nc)] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(heap, (ng + h(nr, nc), ng, (nr, nc)))
    return None


def path_metrics(path, grid, gt_union, ego_radius_m=1.0):
    if not path:
        return {"success": False, "collision": None, "min_clearance": None, "length": None}
    rr = np.array([p[0] for p in path])
    cc = np.array([p[1] for p in path])
    rr = np.clip(rr, 0, grid.shape[0] - 1)
    cc = np.clip(cc, 0, grid.shape[1] - 1)
    gt_infl = ndimage.binary_dilation(
        gt_union, ndimage.generate_binary_structure(2, 1),
        iterations=max(1, int(round(ego_radius_m / grid.res))))
    collision = bool(gt_infl[rr, cc].any())
    dist = ndimage.distance_transform_edt(~gt_union) * grid.res
    clearance = float(dist[rr, cc].min())
    length = float(np.sum(np.hypot(np.diff(rr), np.diff(cc))) * grid.res)
    return {"success": True, "collision": collision,
            "min_clearance": clearance, "length": length}


def perception_metrics(layers, gt_union, box_masks, gt_free, eval_mask):
    free = layers["free"] & eval_mask
    occ = layers["occupied"] & eval_mask
    unknown = layers["unknown"] & eval_mask
    gt_obs = gt_union & eval_mask
    gt_free = gt_free & eval_mask
    known = free | occ
    inter = np.logical_and(free, gt_free).sum()
    union = np.logical_or(free, gt_free)[known].sum()
    iou = float(inter / union) if union else float("nan")
    ffr = float(np.logical_and(gt_obs, free).sum() / max(gt_obs.sum(), 1))
    unknown_ratio = float(unknown.sum() / max(eval_mask.sum(), 1))
    recalls, detected = [], []
    for mask, _ in box_masks:
        m = mask & eval_mask
        n = int(m.sum())
        if n < 4:
            continue
        hit = int((m & occ).sum())
        recalls.append(hit / n)
        detected.append(hit >= max(1, 0.05 * n))
    mean_cov = float(np.mean(recalls)) if recalls else float("nan")
    recall = float(np.mean(detected)) if detected else float("nan")
    return {"free_iou": iou, "false_free_rate": ffr, "unknown_ratio": unknown_ratio,
            "obstacle_recall": recall, "mean_box_coverage": mean_cov,
            "n_boxes": len(recalls)}


def future_ego_path(sc, token, horizon_s=3.0):
    """Actual ego path in the current ego frame, sampled at keyframes."""
    cur = sc.ego_pose(token)
    t0 = sc.samples[token]["timestamp"]
    idx = sc.sample_tokens.index(token)
    pts = [(0.0, np.zeros(2))]
    for nxt in sc.sample_tokens[idx + 1:]:
        dt = (sc.samples[nxt]["timestamp"] - t0) / 1e6
        if dt > horizon_s:
            break
        p = np.asarray(sc.ego_pose(nxt)["translation"], dtype=float)[:2]
        pts.append((dt, global_to_ego(p, cur)))
    return pts


def corridor_mask(grid, path_pts, half_width):
    X, Y = grid.centers()
    pts = np.column_stack([X.ravel(), Y.ravel()])
    d = np.full(len(pts), np.inf)
    for (_, a), (_, b) in zip(path_pts[:-1], path_pts[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            d = np.minimum(d, np.linalg.norm(pts - a, axis=1))
            continue
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.minimum(d, np.linalg.norm(pts - proj, axis=1))
    return (d <= half_width).reshape(grid.shape)


def _ann_boxes_in_ego(sc, anns, pose):
    rot, trans = pose_arrays(pose)
    out = []
    for ann in anns:
        corners = (ann["_corners"] - trans) @ rot
        out.append((corners, ann["_category"], ann["instance_token"]))
    return out


def rectangle_corners(center, heading, length=4.7, width=1.8):
    c, s = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.asarray(center)


def obb_overlap(a, b):
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def counterfactual_collision(sc, token, layers, grid, horizon_s=3.0,
                             coverage_thresh=0.1, ego_length=4.7, ego_width=1.8):
    """Constant-speed straight-line counterfactual: does the ego hit a GT actor
    that the current perception does not detect?"""
    idx = sc.sample_tokens.index(token)
    if idx + 1 >= len(sc.sample_tokens):
        return None
    cur = sc.ego_pose(token)
    t0 = sc.samples[token]["timestamp"]
    nxt = sc.sample_tokens[idx + 1]
    dt = (sc.samples[nxt]["timestamp"] - t0) / 1e6
    p1 = global_to_ego(np.asarray(sc.ego_pose(nxt)["translation"], dtype=float)[:2], cur)
    v = float(np.linalg.norm(p1) / max(dt, 1e-3))

    cur_boxes = {}
    for c, cat, inst in _ann_boxes_in_ego(sc, sc.ann_by_sample[token], cur):
        cur_boxes[inst] = (c, cat)
    occ = layers["occupied"]

    for fut in sc.sample_tokens[idx + 1:]:
        t = (sc.samples[fut]["timestamp"] - t0) / 1e6
        if t > horizon_s:
            break
        ego_c = rectangle_corners((v * t, 0.0), 0.0, ego_length, ego_width)
        for corners, cat, inst in _ann_boxes_in_ego(sc, sc.ann_by_sample[fut], cur):
            foot = corners[BOTTOM][:, :2]
            if not obb_overlap(ego_c, foot):
                continue
            if inst in cur_boxes:
                c0, _ = cur_boxes[inst]
                m, _ = boxes_to_masks([(c0, cat)], grid)
                m = m[0][0]
                cov = float((m & occ).sum() / max(m.sum(), 1))
                visible = True
            else:
                cov, visible = 0.0, False
            return {"collision": True, "instance": inst, "category": cat,
                    "ttc": float(t), "coverage": cov, "visible_now": visible,
                    "detected": cov >= coverage_thresh, "speed": v}
    return {"collision": False, "speed": v}


def _project_to_path(pts, path_pts):
    pts = np.asarray(pts, dtype=float)
    best_d = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    s_acc = 0.0
    for (_, a), (_, b) in zip(path_pts[:-1], path_pts[1:]):
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-9:
            continue
        t = np.clip(((pts - a) @ ab) / (L * L), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(pts - proj, axis=1)
        s = s_acc + t * L
        upd = d < best_d
        best_d[upd] = d[upd]
        best_s[upd] = s[upd]
        s_acc += L
    return best_s, best_d


def build_hazard_set(sc, token, grid, horizon_s=3.0,
                     longitudinal_max=40.0, lateral_max=10.0):
    """Precompute the safety-envelope actors for one frame (GT is fixed)."""
    path = future_ego_path(sc, token, horizon_s)
    if len(path) < 2:
        return [], 0.0
    v = float(np.linalg.norm(path[1][1]) / max(path[1][0], 1e-3))
    boxes = _ann_boxes_in_ego(sc, sc.ann_by_sample[token], sc.ego_pose(token))
    if not boxes:
        return [], v
    centers = np.array([c[BOTTOM][:, :2].mean(axis=0) for c, _, _ in boxes])
    s, d = _project_to_path(centers, path)
    hazards = []
    for (corners, cat, inst), si, di in zip(boxes, s, d):
        if not (0.0 <= si <= longitudinal_max and abs(di) <= lateral_max):
            continue
        m, _ = boxes_to_masks([(corners, cat)], grid)
        hazards.append({"instance": inst, "category": cat,
                        "s": float(si), "d": float(di),
                        "ttc": float(si / max(v, 0.5)),
                        "mask": m[0][0]})
    return hazards, v


def hazard_recall(hazards, occ, coverage_thresh=0.05, min_cells=1):
    if not hazards:
        return {"n_hazards": 0, "recall": float("nan"),
                "false_safe_rate": float("nan"), "n_urgent": 0,
                "n_conflict": 0, "mean_coverage": float("nan")}
    rows = []
    for h in hazards:
        m = h["mask"]
        n = int(m.sum())
        hit = int((m & occ).sum())
        cov = hit / max(n, 1)
        rows.append((hit >= max(min_cells, coverage_thresh * n), cov, h["ttc"], h["d"]))
    det = np.array([r[0] for r in rows])
    cov = np.array([r[1] for r in rows])
    ttc = np.array([r[2] for r in rows])
    dd = np.array([r[3] for r in rows])
    conflict = np.abs(dd) <= 3.5
    urgent = (ttc <= 2.0) & conflict
    return {"n_hazards": len(hazards),
            "recall": float(det.mean()),
            "false_safe_rate": float((~det & urgent).sum() / max(urgent.sum(), 1)),
            "n_urgent": int(urgent.sum()),
            "n_conflict": int(conflict.sum()),
            "mean_coverage": float(cov.mean())}


def hazard_metrics(sc, token, layers, grid, horizon_s=3.0,
                   longitudinal_max=40.0, lateral_max=10.0,
                   coverage_thresh=0.1):
    hazards, _ = build_hazard_set(sc, token, grid, horizon_s,
                                  longitudinal_max, lateral_max)
    return hazard_recall(hazards, layers["occupied"], coverage_thresh)
