#!/usr/bin/env python3
"""Controlled hazard injection: does perception see a virtual actor in time,
and does the planner avoid it?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import BOTTOM, SceneData  # noqa: E402
from eval.core import (  # noqa: E402
    Grid, astar, boxes_to_masks, build_layers, degrade, map_drivable,
    path_metrics, rectangle_corners,
)
from eval.experiments import DEFAULT_SCENES  # noqa: E402

ACTOR_SIZES = {
    "pedestrian": (0.6, 0.6, 1.7),
    "car": (4.5, 1.9, 1.6),
}


def actor_corners(s, lateral, length, width, height, heading=0.0):
    foot = rectangle_corners((s, lateral), heading, length, width)
    corners = np.zeros((8, 3))
    corners[[1, 3, 7, 5], :2] = foot
    corners[[0, 2, 6, 4], :2] = foot
    corners[[0, 2, 6, 4], 2] = height
    return corners


def actor_points(s, lateral, length, width, height, rng):
    x = s - length / 2.0
    ys = np.linspace(lateral - width / 2 + 0.03, lateral + width / 2 - 0.03, 9)
    zs = np.linspace(0.35, height, 14)
    yy, zz = np.meshgrid(ys, zs)
    pts = np.zeros((yy.size, 5), dtype=np.float32)
    pts[:, 0] = x + rng.normal(0, 0.02, yy.size)
    pts[:, 1] = yy.ravel() + rng.normal(0, 0.02, yy.size)
    pts[:, 2] = zz.ravel()
    pts[:, 3] = 100.0
    pts[:, 4] = 0.0
    return pts


def plan(grid, layers, map_free, injected_gt, mode, ego_radius_m=0.6):
    from scipy import ndimage
    it = max(1, int(round(ego_radius_m / grid.res)))
    occ = ndimage.binary_dilation(layers["occupied"], iterations=it)
    gt = ndimage.binary_dilation(injected_gt, iterations=it)
    if mode == "map":
        blocked = ~map_free
    elif mode == "oracle":
        blocked = (~map_free) | gt
    else:
        blocked = (~map_free) | occ
    start = grid.cell_of(np.array([0.0, 0.0]))
    goal = grid.cell_of(np.array([min(30.0, grid.x_max - 3.0), 0.0]))
    path = astar(blocked, (int(start[0]), int(start[1])),
                 (int(goal[0]), int(goal[1])))
    return path, path_metrics(path, grid, injected_gt, ego_radius_m=ego_radius_m)


def draw(ax, grid, layers, injected_gt, path, title):
    ext = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    img = np.zeros(grid.shape + (3,))
    img[layers["free"]] = (0.72, 0.86, 0.95)
    img[layers["occupied"]] = (0.05, 0.05, 0.05)
    img[layers["unknown"]] = (0.65, 0.65, 0.65)
    ax.imshow(img, origin="lower", extent=ext)
    rr, cc = np.where(injected_gt)
    if len(rr):
        x = grid.x_min + (cc + 0.5) * grid.res
        y = grid.y_min + (rr + 0.5) * grid.res
        ax.scatter(x, y, s=8, c="#d62728", marker="s", linewidths=0)
    if path:
        pr = np.array([p[0] for p in path])
        pc = np.array([p[1] for p in path])
        ax.plot(grid.x_min + (pc + 0.5) * grid.res,
                grid.y_min + (pr + 0.5) * grid.res,
                color="#2ca02c", linewidth=1.8)
    ax.add_patch(Rectangle((-2.35, -0.9), 4.7, 1.8, facecolor="black", alpha=0.5))
    ax.set_xlim(-5, min(35, grid.x_max))
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def run_case(sc, frame, degradation, distance, lateral, category, deg_kind,
             deg_param, seed, grid):
    token = sc.sample_tokens[frame]
    raw, ego = sc.lidar_ego(token)
    points = raw.copy()
    points[:, :3] = ego
    rng = np.random.default_rng(seed)
    length, width, height = ACTOR_SIZES[category]
    corners = actor_corners(distance, lateral, length, width, height)
    pts = np.vstack([points, actor_points(distance, lateral, length, width,
                                          height, rng)])
    pts = degrade(pts, deg_kind, deg_param, rng)
    layers = build_layers(pts, grid)
    injected_masks, injected_gt = boxes_to_masks([(corners, "injected")], grid)
    map_free, _ = map_drivable(sc, token, grid)
    im = injected_masks[0][0]
    n = int(im.sum())
    hit = int((im & layers["occupied"]).sum())
    cov = hit / max(n, 1)

    out = {"scene": sc.name, "frame": frame, "degradation": degradation,
           "distance": distance, "lateral": lateral, "category": category,
           "seed": seed, "detected": hit >= max(1, 0.05 * n),
           "coverage": cov, "n_cells": n, "hit_cells": hit}
    for mode in ("perception", "map", "oracle"):
        path, m = plan(grid, layers, map_free, injected_gt, mode)
        out[mode + "_success"] = bool(m["success"])
        out[mode + "_collision"] = m["collision"]
        out[mode + "_length"] = m["length"]
        out[mode + "_clearance"] = m["min_clearance"]
        if mode == "perception":
            out["_path"] = path
    return out, layers, injected_gt, map_free


def plot_case(sc, frame, degradation, distance, category, deg_kind, deg_param,
              seed, grid, out_path):
    out, layers, injected_gt, map_free = run_case(
        sc, frame, degradation, distance, 0.0, category,
        deg_kind, deg_param, seed, grid)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    # panel 1: clean reference
    out_c, lay_c, gt_c, mf_c = run_case(
        sc, frame, "clean", distance, 0.0, category, "clean", 0.0, seed, grid)
    draw(axes[0], grid, lay_c, gt_c, out_c["_path"], "clean perception")
    draw(axes[1], grid, layers, injected_gt, out["_path"],
         "%s (coverage %.2f)" % (degradation, out["coverage"]))
    # map-only and oracle use the same degraded layers but different blockers
    for ax, mode, title in ((axes[2], "map", "map-only"),
                            (axes[3], "oracle", "oracle GT")):
        path, m = plan(grid, layers, map_free, injected_gt, mode)
        draw(ax, grid, layers, injected_gt, path,
             "%s (collision %s)" % (title, m["collision"]))
    fig.suptitle("%s f%d | %s | %s at %.0f m"
                 % (sc.name, frame, degradation, category, distance), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary(df, out_path):
    df = df.copy()
    df["detected"] = df["detected"].astype(float)
    df["perception_collision"] = df["perception_collision"].astype(float)
    g = df.groupby(["degradation", "distance"]).agg(
        detected=("detected", "mean"),
        collision=("perception_collision", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for deg in g["degradation"].unique():
        sub = g[g["degradation"] == deg]
        axes[0].plot(sub["distance"], sub["detected"], marker="o", label=deg)
        axes[1].plot(sub["distance"], sub["collision"], marker="o", label=deg)
    axes[0].set_title("Detection rate vs distance", fontsize=9)
    axes[1].set_title("Collision rate vs distance", fontsize=9)
    for ax in axes:
        ax.set_xlabel("injected distance (m)", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 10, 20])
    ap.add_argument("--distances", nargs="+", type=float,
                    default=[10, 15, 20, 25])
    ap.add_argument("--degradations", nargs="+",
                    default=["clean", "range30", "range20", "drop50"])
    ap.add_argument("--category", default="pedestrian", choices=list(ACTOR_SIZES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/eval_results")
    args = ap.parse_args()

    from eval.experiments import parse_name
    grid = Grid(res=args.res, x_min=-30, x_max=35, y_min=-20, y_max=20)
    fig_dir = os.path.join(args.out, "figures")
    met_dir = os.path.join(args.out, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    rows = []
    for scene in args.scenes:
        sc = SceneData(args.root, args.version, scene)
        for frame in args.frames:
            if frame >= len(sc.sample_tokens):
                continue
            for name in args.degradations:
                kind, param = parse_name(name)
                nseed = 1 if kind in ("clean", "range") else args.seeds
                for distance in args.distances:
                    for seed in range(nseed):
                        out, _, _, _ = run_case(
                            sc, frame, name, distance, 0.0, args.category,
                            kind, param, seed, grid)
                        out.pop("_path", None)
                        rows.append(out)
        plot_case(sc, args.frames[0], args.degradations[-1],
                  max(args.distances) - 10, args.category,
                  *parse_name(args.degradations[-1]), 0, grid,
                  os.path.join(fig_dir, "injection_%s.png" % scene))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(met_dir, "injection_metrics.csv"), index=False)
    plot_summary(df, os.path.join(fig_dir, "injection_summary.png"))
    print("wrote", met_dir, fig_dir)


if __name__ == "__main__":
    main()
