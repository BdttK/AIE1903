#!/usr/bin/env python3
"""Map prior vs live perception discrepancy analysis (A/B/C classes)."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import BOTTOM, SceneData  # noqa: E402
from eval.core import (  # noqa: E402
    Grid, boxes_to_masks, build_layers, future_ego_path, map_drivable,
)
from eval.experiments import DEFAULT_SCENES, SCENE_META  # noqa: E402


def _path_cells(grid, path, step=0.25):
    pts = []
    for (_, a), (_, b) in zip(path[:-1], path[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(L / step)))
        for k in range(n):
            pts.append(a + (b - a) * (k / n))
    if path:
        pts.append(path[-1][1])
    if not pts:
        return np.array([], dtype=int), np.array([], dtype=int)
    rr, cc = grid.cell_of(np.array(pts))
    ok = grid.inside(rr, cc)
    return rr[ok], cc[ok]


def map_metrics(sc, token, grid, layers, gt_union, box_masks):
    map_free, map_valid = map_drivable(sc, token, grid)
    X, Y = grid.centers()
    R = np.hypot(X, Y)
    area = map_valid & (R <= 50.0)
    occ = layers["occupied_low"]
    free = layers["free"]
    unknown = layers["unknown"]

    A = map_free & occ & area
    near_road = ndimage.binary_dilation(
        map_free, iterations=max(1, int(round(3.0 / grid.res))))
    B = (~map_free) & free & area & near_road
    C = map_free & unknown & area
    n_area = max(int(area.sum()), 1)

    a_by_cat = Counter()
    explained_mask = np.zeros_like(A)
    for m, cat in box_masks:
        md = ndimage.binary_dilation(m, iterations=2)
        c = int((A & md).sum())
        if c:
            a_by_cat[cat] += c
        explained_mask |= md
    explained = int((A & explained_mask).sum())
    unexplained = int(A.sum()) - explained

    path = future_ego_path(sc, token, 3.0)
    rr, cc = _path_cells(grid, path)
    exp_cells = int(A[rr, cc].sum())
    path_cells = int(len(rr))

    total_a = int(A.sum())
    top_cat, top_n = (a_by_cat.most_common(1)[0] if a_by_cat else ("none", 0))
    return {
        "map_free_frac": float(map_free[area].mean()) if area.any() else float("nan"),
        "occ_on_map_frac": float(A.sum() / n_area),
        "free_off_map_frac": float(B.sum() / n_area),
        "unknown_on_map_frac": float(C.sum() / n_area),
        "A_cells": total_a, "B_cells": int(B.sum()), "C_cells": int(C.sum()),
        "A_area_m2": total_a * grid.res ** 2,
        "B_area_m2": float(B.sum()) * grid.res ** 2,
        "A_explained_frac": float(explained / total_a) if total_a else float("nan"),
        "A_unexplained_frac": float(unexplained / total_a) if total_a else float("nan"),
        "A_top_category": top_cat,
        "A_top_category_frac": float(top_n / total_a) if total_a else float("nan"),
        "ego_exposure_cells": exp_cells,
        "ego_exposure_ratio": float(exp_cells / max(path_cells, 1)),
        "path_cells": path_cells,
    }


class GlobalDiscrepancy:
    """Accumulate A/B cells in a fixed global grid for persistence maps."""

    def __init__(self, x_min, y_min, x_max, y_max, res=0.5):
        self.res = res
        self.x_min = float(np.floor(x_min / res) * res)
        self.y_min = float(np.floor(y_min / res) * res)
        self.nx = int(np.ceil((x_max - self.x_min) / res)) + 1
        self.ny = int(np.ceil((y_max - self.y_min) / res)) + 1
        self.a = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.b = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.frames = 0

    def add(self, grid, gx, gy, A, B):
        self.frames += 1
        for mask, arr in ((A, self.a), (B, self.b)):
            row, col = np.where(mask)
            if not len(row):
                continue
            x = gx[row, col]
            y = gy[row, col]
            c = np.floor((x - self.x_min) / self.res).astype(int)
            r = np.floor((y - self.y_min) / self.res).astype(int)
            ok = (c >= 0) & (c < self.nx) & (r >= 0) & (r < self.ny)
            np.add.at(arr, (r[ok], c[ok]), 1)

    def extent(self):
        return (self.x_min, self.x_min + self.nx * self.res,
                self.y_min, self.y_min + self.ny * self.res)


def _global_centers(sc, token, grid):
    from eval.core import ego_to_global
    X, Y = grid.centers()
    g = ego_to_global(np.column_stack([X.ravel(), Y.ravel()]), sc.ego_pose(token))
    return g[:, 0].reshape(grid.shape), g[:, 1].reshape(grid.shape)


def run_scene(scene_name, root, version, frame_step, grid, out_dir, verbose=True):
    sc = SceneData(root, version, scene_name)
    meta = SCENE_META.get(scene_name, {"city": "", "time": ""})
    rows = []
    gx0 = np.array([sc.ego_pose(t)["translation"][0] for t in sc.sample_tokens])
    gy0 = np.array([sc.ego_pose(t)["translation"][1] for t in sc.sample_tokens])
    acc = GlobalDiscrepancy(gx0.min() - 100, gy0.min() - 100,
                            gx0.max() + 100, gy0.max() + 100, res=0.5)
    for fi in range(0, len(sc.sample_tokens), frame_step):
        token = sc.sample_tokens[fi]
        _, pts = sc.lidar_ego(token)
        layers = build_layers(pts, grid)
        boxes = sc.boxes_ego(token)
        box_masks, gt_union = boxes_to_masks(boxes, grid)
        m = map_metrics(sc, token, grid, layers, gt_union, box_masks)
        m.update({"scene": scene_name, "city": meta["city"], "time": meta["time"],
                  "frame": fi, "token": token})
        rows.append(m)
        gx, gy = _global_centers(sc, token, grid)
        map_free, map_valid = map_drivable(sc, token, grid)
        X, Y = grid.centers()
        area = map_valid & (np.hypot(X, Y) <= 50.0)
        near_road = ndimage.binary_dilation(
            map_free, iterations=max(1, int(round(3.0 / grid.res))))
        acc.add(grid, gx, gy,
                map_free & layers["occupied_low"] & area,
                (~map_free) & layers["free"] & area & near_road)
        if verbose and (fi // max(frame_step, 1) + 1) % 20 == 0:
            print("  %s: frame %d/%d" % (scene_name, fi, len(sc.sample_tokens)),
                  flush=True)
    np.savez_compressed(
        os.path.join(out_dir, "map_global_%s.npz" % scene_name),
        a=acc.a, b=acc.b, frames=acc.frames,
        extent=np.array(acc.extent()), res=acc.res)
    return rows


def plot_frame(scene_name, root, version, frame_idx, grid, out_path):
    sc = SceneData(root, version, scene_name)
    token = sc.sample_tokens[frame_idx]
    _, pts = sc.lidar_ego(token)
    layers = build_layers(pts, grid)
    boxes = sc.boxes_ego(token)
    box_masks, gt_union = boxes_to_masks(boxes, grid)
    map_free, map_valid = map_drivable(sc, token, grid)
    X, Y = grid.centers()
    area = map_valid & (np.hypot(X, Y) <= 50.0)
    A = map_free & layers["occupied_low"] & area
    near_road = ndimage.binary_dilation(
        map_free, iterations=max(1, int(round(3.0 / grid.res))))
    B = (~map_free) & layers["free"] & area & near_road
    C = map_free & layers["unknown"] & area

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    ext = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)

    axes[0, 0].imshow(map_free, origin="lower", cmap="gray", extent=ext)
    axes[0, 0].set_title("map drivable (white)", fontsize=9)

    img = np.zeros(grid.shape + (3,))
    img[layers["free"]] = (0.72, 0.86, 0.95)
    img[layers["occupied"]] = (0.05, 0.05, 0.05)
    img[layers["unknown"]] = (0.65, 0.65, 0.65)
    axes[0, 1].imshow(img, origin="lower", extent=ext)
    axes[0, 1].set_title("perception: free / occupied / unknown", fontsize=9)

    axes[1, 0].imshow(map_free, origin="lower", cmap="gray", extent=ext, alpha=0.35)
    overlay = np.zeros(grid.shape + (4,))
    overlay[A] = (0.85, 0.10, 0.10, 0.85)
    overlay[B] = (0.10, 0.35, 0.85, 0.85)
    overlay[C] = (0.55, 0.55, 0.55, 0.35)
    axes[1, 0].imshow(overlay, origin="lower", extent=ext)
    for corners, cat in boxes:
        foot = corners[BOTTOM][:, :2]
        if np.abs(foot).max() > 55:
            continue
        axes[1, 0].add_patch(Polygon(foot, closed=True, fill=False,
                                     edgecolor="#2ca02c", linewidth=0.6))
    path = future_ego_path(sc, token, 3.0)
    pp = np.array([p for _, p in path])
    axes[1, 0].plot(pp[:, 0], pp[:, 1], color="#ff7f0e", linewidth=1.6)
    axes[1, 0].set_title("A=occupied-on-map (red), B=free-off-map (blue), C=unknown (gray)",
                         fontsize=8)

    cats = Counter()
    explained_mask = np.zeros_like(A)
    for m, cat in box_masks:
        md = ndimage.binary_dilation(m, iterations=2)
        c = int((A & md).sum())
        if c:
            cats[cat.replace("movable_object.", "mov.").replace("human.", "")] += c
        explained_mask |= md
    unexplained = int(A.sum()) - int((A & explained_mask).sum())
    if unexplained > 0:
        cats["unexplained"] = unexplained
    if cats:
        labels, vals = zip(*cats.most_common(8))
        axes[1, 1].barh(list(labels)[::-1], list(vals)[::-1], color="#d62728", alpha=0.8)
        axes[1, 1].set_xlabel("A cells", fontsize=8)
    axes[1, 1].set_title("A-class attribution", fontsize=9)

    for ax in axes.ravel():
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle("%s frame %d | A=%d B=%d C=%d | unexplained %.2f"
                 % (scene_name, frame_idx, A.sum(), B.sum(), C.sum(),
                    unexplained / max(A.sum(), 1)), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_persistence(npz_path, out_path):
    d = np.load(npz_path)
    a, b, frames = d["a"], d["b"], int(d["frames"])
    extent = d["extent"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, arr, title in ((axes[0], a, "A: occupied on map"),
                           (axes[1], b, "B: free off map")):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="magma",
                       interpolation="nearest")
        ax.set_title("%s (max count %d, frames %d)" % (title, arr.max(), frames),
                     fontsize=9)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/eval_results")
    ap.add_argument("--demo-frame", type=int, default=10)
    args = ap.parse_args()
    grid = Grid(res=args.res)
    fig_dir = os.path.join(args.out, "figures")
    met_dir = os.path.join(args.out, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    rows = []
    for scene in args.scenes:
        rows.extend(run_scene(scene, args.root, args.version, args.frame_step,
                              grid, met_dir))
        plot_frame(scene, args.root, args.version, args.demo_frame, grid,
                   os.path.join(fig_dir, "map_discrepancy_%s.png" % scene))
        plot_persistence(os.path.join(met_dir, "map_global_%s.npz" % scene),
                         os.path.join(fig_dir, "map_persistence_%s.png" % scene))
    pd.DataFrame(rows).to_csv(os.path.join(met_dir, "map_metrics.csv"), index=False)
    print("wrote", met_dir, fig_dir)


if __name__ == "__main__":
    main()
