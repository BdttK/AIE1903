#!/usr/bin/env python3
"""Figures for the degradation evaluation: curves, profiles, failure gallery."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import BOTTOM, SceneData  # noqa: E402
from eval.core import (  # noqa: E402
    Grid, boxes_to_masks, build_hazard_set, build_layers, corridor_mask,
    degrade, future_ego_path, map_drivable,
)
from eval.experiments import DEFAULT_DEGRADATIONS, parse_name  # noqa: E402

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def order_degradations(values):
    seen = list(dict.fromkeys(values))
    order = [d for d in DEFAULT_DEGRADATIONS if d in seen]
    order += [d for d in seen if d not in order]
    return order


def plot_curves(summary_scene, summary_deg, out_dir):
    order = order_degradations(summary_scene["degradation"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, scene in enumerate(summary_scene["scene"].unique()):
        sub = summary_scene[summary_scene["scene"] == scene].set_index("degradation")
        sub = sub.reindex(order)
        axes[0].errorbar(range(len(order)), sub["envelope_recall_mean"],
                         yerr=sub["envelope_recall_ci"], marker="o",
                         color=PALETTE[i % len(PALETTE)], label=scene, capsize=2)
        axes[1].plot(range(len(order)), sub["ffr_mean"], marker="o",
                     color=PALETTE[i % len(PALETTE)], label=scene)
    for ax, title, ylab in ((axes[0], "Safety-envelope recall", "recall"),
                            (axes[1], "False-free rate", "false-free rate")):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_degradation_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, tod in enumerate(["day", "night"]):
        sub = summary_deg[summary_deg["time"] == tod].set_index("degradation")
        sub = sub.reindex(order)
        ax.errorbar(range(len(order)), sub["envelope_recall_mean"],
                    yerr=sub["envelope_recall_ci"], marker="o",
                    color=PALETTE[i], label=tod, capsize=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_title("Envelope recall: day vs night", fontsize=10)
    ax.set_ylabel("recall", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_day_night.png"), dpi=150)
    plt.close(fig)


def plot_tradeoff(summary_scene, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, scene in enumerate(summary_scene["scene"].unique()):
        sub = summary_scene[summary_scene["scene"] == scene]
        ax.plot(sub["ffr_mean"], sub["envelope_recall_mean"], marker="o",
                color=PALETTE[i % len(PALETTE)], label=scene)
        for _, row in sub.iterrows():
            ax.annotate(row["degradation"], (row["ffr_mean"], row["envelope_recall_mean"]),
                        fontsize=5, alpha=0.6)
    ax.set_xlabel("false-free rate", fontsize=8)
    ax.set_ylabel("envelope recall", fontsize=8)
    ax.set_title("Safety-relevant perception trade-off", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tradeoff.png"), dpi=150)
    plt.close(fig)


def plot_distance_profile(hazards_csv, out_dir):
    df = pd.read_csv(hazards_csv)
    keep = ["clean", "range30", "range20", "range10", "drop50", "drop75"]
    df = df[df["degradation"].isin(keep)].copy()
    bins = np.arange(0, 45, 5)
    df["bin"] = pd.cut(df["s"], bins)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, deg in enumerate(keep):
        sub = df[df["degradation"] == deg]
        g = sub.groupby("bin", observed=True)["detected"].mean()
        axes[0].plot([b.mid for b in g.index], g.values, marker="o", label=deg,
                     color=PALETTE[i])
        g2 = sub.groupby("bin", observed=True)["coverage"].mean()
        axes[1].plot([b.mid for b in g2.index], g2.values, marker="o", label=deg,
                     color=PALETTE[i])
    axes[0].set_title("Detection rate vs longitudinal distance", fontsize=10)
    axes[1].set_title("Mean box coverage vs distance", fontsize=10)
    for ax in axes:
        ax.set_xlabel("distance along ego path (m)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_distance_profile.png"), dpi=150)
    plt.close(fig)


def plot_category_profile(hazards_csv, out_dir):
    df = pd.read_csv(hazards_csv)
    df = df[df["degradation"].isin(["clean", "range20", "range10"])].copy()
    df["cat"] = (df["category"].str.replace("movable_object.", "mov.", regex=False)
                 .str.replace("human.", "", regex=False)
                 .str.replace("vehicle.", "", regex=False))
    top = df["cat"].value_counts().head(6).index
    df = df[df["cat"].isin(top)]
    g = df.groupby(["cat", "degradation"])["detected"].mean().unstack()
    g = g.reindex(top)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(g))
    w = 0.8 / max(len(g.columns), 1)
    for i, deg in enumerate(g.columns):
        ax.bar(x + i * w, g[deg].values, width=w, label=deg,
               color=PALETTE[i])
    ax.set_xticks(x + w * (len(g.columns) - 1) / 2)
    ax.set_xticklabels(g.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("detection rate", fontsize=8)
    ax.set_title("Detection by category", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_category_profile.png"), dpi=150)
    plt.close(fig)


def render_failure(sc, token, degradation, grid, out_path):
    kind, param = parse_name(degradation)
    raw, ego = sc.lidar_ego(token)
    pts = raw.copy()
    pts[:, :3] = ego
    pts = degrade(pts, kind, param, np.random.default_rng(0))
    layers = build_layers(pts, grid)
    boxes = sc.boxes_ego(token)
    box_masks, gt_union = boxes_to_masks(boxes, grid)
    hazards, _ = build_hazard_set(sc, token, grid)
    path = future_ego_path(sc, token, 3.0)
    env = corridor_mask(grid, path, 10.0)
    map_free, map_valid = map_drivable(sc, token, grid)
    X, Y = grid.centers()
    area = map_valid & (np.hypot(X, Y) <= 50.0)
    A = map_free & layers["occupied_low"] & area

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    ext = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)

    m = (np.abs(pts[:, 0]) < 50) & (np.abs(pts[:, 1]) < 50)
    p = pts[m]
    axes[0, 0].scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="turbo", s=0.4,
                       vmin=-0.5, vmax=3)
    axes[0, 0].set_title("LiDAR ego frame (z color)", fontsize=9)

    img = np.zeros(grid.shape + (3,))
    img[layers["free"]] = (0.72, 0.86, 0.95)
    img[layers["occupied"]] = (0.05, 0.05, 0.05)
    img[layers["unknown"]] = (0.65, 0.65, 0.65)
    axes[0, 1].imshow(img, origin="lower", extent=ext)
    axes[0, 1].set_title("occupancy: free / occupied / unknown", fontsize=9)

    axes[1, 0].imshow(np.where(env, 1.0, 0.0), origin="lower", extent=ext,
                      cmap="Blues", alpha=0.35)
    for corners, cat in boxes:
        foot = corners[BOTTOM][:, :2]
        if np.abs(foot).max() > 55:
            continue
        axes[1, 0].add_patch(Polygon(foot, closed=True, fill=False,
                                     edgecolor="#888888", linewidth=0.5))
    for h in hazards:
        m = h["mask"]
        n = int(m.sum())
        hit = int((m & layers["occupied"]).sum())
        rr, cc = np.where(m)
        axes[1, 0].scatter(grid.x_min + (cc + 0.5) * grid.res,
                           grid.y_min + (rr + 0.5) * grid.res,
                           s=3, color="#2ca02c" if hit >= max(1, 0.05 * n)
                           else "#d62728", linewidths=0)
    axes[1, 0].plot([p[1][0] for p in path], [p[1][1] for p in path],
                    color="#ff7f0e", linewidth=1.5)
    axes[1, 0].set_title("GT boxes + safety envelope (green detected / red missed)",
                         fontsize=8)

    axes[1, 1].imshow(map_free, origin="lower", cmap="gray", extent=ext, alpha=0.6)
    ov = np.zeros(grid.shape + (4,))
    ov[A] = (0.85, 0.10, 0.10, 0.85)
    axes[1, 1].imshow(ov, origin="lower", extent=ext)
    axes[1, 1].set_title("map drivable + A (occupied on map)", fontsize=9)

    for ax in axes.ravel():
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle("%s | %s | frame %s" % (sc.name, degradation,
                                        sc.sample_tokens.index(token)), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def failure_gallery(per_frame_csv, root, version, grid, out_dir, degradation="range10"):
    df = pd.read_csv(per_frame_csv)
    sub = df[df["degradation"] == degradation]
    scenes = list(sub["scene"].unique())
    for scene in scenes:
        s = sub[sub["scene"] == scene]
        if s.empty:
            continue
        row = s.sort_values("envelope_recall").iloc[0]
        sc = SceneData(root, version, scene)
        token = sc.sample_tokens[int(row["frame"])]
        render_failure(sc, token, degradation, grid,
                       os.path.join(out_dir, "failure_%s.png" % scene))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/eval_results/metrics")
    ap.add_argument("--figures", default="outputs/eval_results/figures")
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--gallery-degradation", default="range10")
    ap.add_argument("--skip-gallery", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)
    grid = Grid(res=args.res)

    summary_scene = pd.read_csv(os.path.join(args.metrics, "summary_by_scene.csv"))
    summary_deg = pd.read_csv(os.path.join(args.metrics, "summary_by_degradation.csv"))
    plot_curves(summary_scene, summary_deg, args.figures)
    plot_tradeoff(summary_scene, args.figures)
    hazards_csv = os.path.join(args.metrics, "hazards.csv")
    if os.path.exists(hazards_csv):
        plot_distance_profile(hazards_csv, args.figures)
        plot_category_profile(hazards_csv, args.figures)
    if not args.skip_gallery:
        failure_gallery(os.path.join(args.metrics, "per_frame.csv"),
                        args.root, args.version, grid, args.figures,
                        args.gallery_degradation)
    print("wrote", args.figures)


if __name__ == "__main__":
    main()
