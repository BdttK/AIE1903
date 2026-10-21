#!/usr/bin/env python3
"""Single-frame smoke test for the perception-degradation audit."""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import BOTTOM, SceneData
from audit.core import (
    Grid, astar, boxes_to_masks, build_layers, ego_to_global, goal_in_ego,
    map_drivable, path_metrics, perception_metrics,
)


def plan_blocked(layers, mode, map_free=None, map_valid=None, unknown_blocks=True):
    occ = layers["occupied"]
    unknown = layers["unknown"] if unknown_blocks else np.zeros_like(layers["unknown"])
    if mode == "perception":
        return occ | unknown
    if mode == "map":
        blocked = np.ones_like(occ)
        if map_valid is not None:
            blocked[map_valid] = ~map_free[map_valid]
        return blocked
    if mode == "fusion":
        blocked = occ | unknown
        if map_valid is not None:
            blocked |= map_valid & ~map_free
        return blocked
    raise ValueError(mode)


def draw_layers(ax, layers, grid, title):
    ny, nx = grid.shape
    img = np.zeros((ny, nx, 3), dtype=float)
    img[layers["free"]] = (0.72, 0.86, 0.95)
    img[layers["occupied"]] = (0.05, 0.05, 0.05)
    img[layers["unknown"]] = (0.62, 0.62, 0.62)
    ax.imshow(img, origin="lower",
              extent=(grid.x_min, grid.x_max, grid.y_min, grid.y_max))
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--param", type=float, default=30.0)
    ap.add_argument("--res", type=float, default=0.2)
    args = ap.parse_args()

    sc = SceneData(args.root, args.version, args.scene)
    token = sc.sample_tokens[args.frame]
    points, points_ego = sc.lidar_ego(token)
    if args.degradation != "clean":
        from audit.core import degrade
        points_ego = degrade(points_ego, args.degradation, args.param, np.random.default_rng(0))

    grid = Grid(res=args.res)
    layers = build_layers(points_ego, grid)
    box_masks, gt_union = boxes_to_masks(sc.boxes_ego(token), grid)
    map_free, map_valid = map_drivable(sc, token, grid)

    X, Y = grid.centers()
    eval_mask = (np.hypot(X, Y) <= 50.0) & (map_free | gt_union)
    gt_free = map_free & ~gt_union
    percept = perception_metrics(layers, gt_union, box_masks, gt_free, eval_mask)

    start = grid.cell_of(np.array([0.0, 0.0]))
    goal = goal_in_ego(sc, token, 15.0)
    goal_cell = grid.cell_of(goal)
    results = {}
    paths = {}
    for mode in ("perception", "map", "fusion"):
        blocked = plan_blocked(layers, mode, map_free, map_valid, unknown_blocks=True)
        path = astar(blocked, (int(start[0]), int(start[1])),
                     (int(goal_cell[0]), int(goal_cell[1])))
        paths[mode] = path
        results[mode] = path_metrics(path, grid, gt_union)

    print("scene", args.scene, "frame", args.frame, "token", token)
    print("points", len(points), "->", len(points_ego))
    print("map coverage %.3f" % map_valid.mean())
    print("perception", percept)
    for mode, m in results.items():
        print(mode, m)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    ax = axes[0, 0]
    m = (np.abs(points_ego[:, 0]) < 50) & (np.abs(points_ego[:, 1]) < 50)
    p = points_ego[m]
    ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="turbo", s=0.5, vmin=-0.5, vmax=3)
    ax.set_title("LiDAR ego frame (z color)", fontsize=9)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect("equal")

    draw_layers(axes[0, 1], layers, grid, "occupancy layers")

    ax = axes[1, 0]
    for corners, cat in sc.boxes_ego(token):
        foot = corners[BOTTOM][:, :2]
        if np.abs(foot).max() > 60:
            continue
        ax.add_patch(Polygon(foot, closed=True, fill=False, edgecolor="#d62728", linewidth=0.7))
    for mode, color in (("perception", "#1f77b4"), ("map", "#ff7f0e"), ("fusion", "#2ca02c")):
        path = paths[mode]
        if path:
            rr = np.array([q[0] for q in path]); cc = np.array([q[1] for q in path])
            x = grid.x_min + (cc + 0.5) * grid.res
            y = grid.y_min + (rr + 0.5) * grid.res
            ax.plot(x, y, color=color, linewidth=1.6, label=mode)
    ax.scatter(goal[0], goal[1], marker="X", c="black", s=60, label="goal")
    ax.add_patch(Rectangle((-1.0, -0.9), 4.7, 1.8, facecolor="black", alpha=0.5))
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("GT boxes + planned paths", fontsize=9)

    ax = axes[1, 1]
    ax.imshow(map_free, origin="lower", cmap="gray",
              extent=(grid.x_min, grid.x_max, grid.y_min, grid.y_max))
    ax.set_title("map drivable (white) + goal", fontsize=9)
    ax.scatter(goal[0], goal[1], marker="X", c="red", s=60)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect("equal")

    fig.suptitle(
        "%s f%03d | IoU %.3f FFR %.3f unk %.3f rec %.2f | "
        "percept collision=%s map collision=%s" % (
            args.scene, args.frame, percept["free_iou"], percept["false_free_rate"],
            percept["unknown_ratio"], percept["obstacle_recall"],
            results["perception"]["collision"], results["map"]["collision"]),
        fontsize=10)
    fig.tight_layout()
    out_dir = os.path.join("outputs", "audit_smoke")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s_f%03d_%s.png" % (args.scene, args.frame, args.degradation))
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
