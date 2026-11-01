#!/usr/bin/env python3
"""Figures and video for closed-loop scenarios."""

from __future__ import annotations

import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from eval.core import rectangle_corners


def _rgba(layers):
    h, w = layers["free"].shape
    out = np.ones((h, w, 4))
    out[layers["unknown"]] = [0.84, 0.84, 0.84, 1.0]
    out[layers["occupied_low"]] = [0.22, 0.22, 0.22, 1.0]
    tall = layers["occupied"] & ~layers["occupied_low"]
    out[tall] = [0.55, 0.42, 0.35, 1.0]
    return out


def _extent(grid):
    return (grid.x_min, grid.x_max, grid.y_min, grid.y_max)


def _draw_base(ax, rec, xlim, ylim):
    grid = rec["grid"]
    ax.imshow(_rgba(rec["layers"]), extent=_extent(grid), origin="lower", zorder=0)
    ax.contour(grid.centers()[0], grid.centers()[1], rec["map_free"].astype(float),
               levels=[0.5], colors=["#7fb3d5"], linewidths=0.8, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_agents(ax, rec):
    for a in rec["agents"]:
        ax.add_patch(Polygon(a["corners"], closed=True, facecolor="#e06666",
                             edgecolor="#990000", alpha=0.55, linewidth=0.8, zorder=4))


def _draw_forecasts(ax, rec):
    for f in rec["forecasts"]:
        for prob, traj, _ in f["forecast"].modes:
            ax.plot(traj[:, 0], traj[:, 1], "--", color="#c0392b",
                    alpha=0.25 + 0.6 * prob, linewidth=1.2, zorder=5)


def _draw_plan(ax, rec):
    plan = rec["plan"]
    info = plan.info if plan is not None else {}
    if info.get("mode") == "frenet" and "samples" in info:
        samples = info["samples"]
        feasible = info.get("feasible")
        for k in range(samples.shape[1]):
            ok = True if feasible is None else bool(feasible[k])
            ax.plot(samples[:, k, 0], samples[:, k, 1],
                    color="#5dade2" if ok else "#f5b7b1",
                    alpha=0.45 if ok else 0.3, linewidth=0.8, zorder=2)
    elif info.get("mode") == "astar" and "visited" in info:
        g = rec["grid"]
        X, Y = g.centers()
        v = np.array(info["visited"])
        if len(v):
            ax.scatter(X[v[:, 0], v[:, 1]], Y[v[:, 0], v[:, 1]], s=1.5,
                       c="#5dade2", alpha=0.35, linewidths=0, zorder=2)
    if plan is not None:
        ax.plot(plan.xy[:, 0], plan.xy[:, 1], color="#27ae60", linewidth=2.2, zorder=6)
        ax.plot(plan.xy[0, 0], plan.xy[0, 1], "o", color="#27ae60",
                markersize=4, zorder=6)


def _draw_ego(ax, rec, trail=None, show_status=True):
    e = rec["ego"]
    if trail is not None:
        ax.plot(trail[:, 0], trail[:, 1], color="#1f77b4", linewidth=1.8,
                alpha=0.8, zorder=5)
    ax.add_patch(Polygon(rectangle_corners(e.xy, e.yaw, 4.7, 1.8), closed=True,
                         facecolor="#1f77b4", edgecolor="#0b3d66",
                         alpha=0.85, zorder=7))
    ax.arrow(e.x, e.y, 3.0 * np.cos(e.yaw), 3.0 * np.sin(e.yaw),
             width=0.08, color="#0b3d66", zorder=8)
    if not show_status:
        return
    mode = rec["safety"].mode
    color = {"CRUISE": "#1e8449", "CAUTION": "#b9770e", "MRM": "#922b21"}[mode]
    ax.text(0.02, 0.97, f"{mode}  v={e.v:4.1f} m/s  free={rec['safety'].free_dist:4.1f} m",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, color="white",
            bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=3.0), zorder=9)


def _overview(arrays, metrics, out_path):
    t = arrays["t"]
    v = arrays["v"]
    modes = arrays["safety_mode"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    route = arrays["route_xy"]
    ax.plot(route[:, 0], route[:, 1], color="#b0b0b0", linewidth=2, label="route")
    ax.plot(arrays["x"], arrays["y"], color="#1f77b4", linewidth=2, label="ego")
    ax.plot(arrays["x"][0], arrays["y"][0], "go", markersize=7, label="start")
    ax.plot(arrays["x"][-1], arrays["y"][-1], "rs", markersize=7, label="end")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Executed trajectory")

    ax = axes[0, 1]
    ax.plot(t, v, color="#1f77b4", linewidth=2)
    for mode, color in [("CAUTION", "#f5cba7"), ("MRM", "#f1948a")]:
        m = modes == mode
        if m.any():
            ax.fill_between(t, 0, 12, where=m, color=color, alpha=0.5,
                            step="mid", label=mode)
    ax.set_xlabel("t (s)"); ax.set_ylabel("speed (m/s)"); ax.set_ylim(0, 12)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("Speed and safety mode")

    ax = axes[1, 0]
    ax.plot(t, arrays["free_d"], color="#2c3e50", linewidth=2, label="free distance")
    ax.plot(t, arrays["safe_dist"], color="#c0392b", linestyle="--", label="RSS safe")
    ax.set_xlabel("t (s)"); ax.set_ylabel("distance (m)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, arrays["unk_ratio"], color="#8e44ad", alpha=0.5, label="unknown ratio")
    ax2.set_ylabel("unknown ratio"); ax2.set_ylim(0, 1)
    ax.set_title("Forward visibility")

    ax = axes[1, 1]
    ax.plot(t, arrays["gap"], color="#16a085", linewidth=2, label="min gap")
    ax.plot(t, np.clip(arrays["ttc"], 0, 10), color="#d35400", linewidth=1.5,
            label="TTC (clipped 10 s)")
    ax.set_xlabel("t (s)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Closest interaction")
    fig.suptitle(f"{metrics['scene']}  {metrics['degradation']}  "
                 f"{metrics['planner']}+{metrics['controller']}  "
                 f"collision={metrics['collision']}  success={metrics['success']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _frames(records, out_dir, fps=10, max_frames=60):
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    idx = np.linspace(0, len(records) - 1, min(max_frames, len(records))).astype(int)
    trail = np.array([[r["ego"].x, r["ego"].y] for r in records])
    made = []
    for n, i in enumerate(idx):
        rec = records[i]
        e = rec["ego"]
        xlim = (e.x - 30, e.x + 30)
        ylim = (e.y - 30, e.y + 30)
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), dpi=80)
        for ax, title in zip(axes, ["Prediction", "Planning", "Execution"]):
            _draw_base(ax, rec, xlim, ylim)
            _draw_agents(ax, rec)
            ax.set_title(title, fontsize=11)
        _draw_forecasts(axes[0], rec)
        _draw_plan(axes[1], rec)
        _draw_plan(axes[2], rec)
        _draw_ego(axes[0], rec, show_status=False)
        _draw_ego(axes[1], rec, show_status=False)
        _draw_ego(axes[2], rec, trail=trail[:i + 1])
        fig.suptitle(f"{rec['t']:.1f} s   {rec['safety'].mode}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(frames_dir, "%04d.png" % n)
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        made.append(path)
    return frames_dir, made


def _video(frames_dir, made, out_path, fps=10):
    if not made:
        return None
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", os.path.join(frames_dir, "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path], check=False)
        if os.path.exists(out_path):
            return out_path
    return None


def render_scenario(records, metrics, arrays, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _overview(arrays, metrics, os.path.join(out_dir, "overview.png"))
    frames_dir, made = _frames(records, out_dir)
    video = _video(frames_dir, made, os.path.join(out_dir, "closed_loop.mp4"))
    if made:
        shutil.copyfile(made[-1], os.path.join(out_dir, "final.png"))
    print("rendered", out_dir, "video" if video else "frames")
    return out_dir


def plot_comparison(csv_path, out_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "safety" not in df:
        df["safety"] = "aware"
    if "progress_ratio" not in df:
        df["progress_ratio"] = np.clip(
            df["goal_progress"] / df["route_length"].clip(lower=1e-3), 0.0, 1.0)
    coll_col = "collision_moving" if "collision_moving" in df else "collision"
    df["combo"] = df["planner"] + "+" + df["controller"]
    scatter_df = df[df["collision_stop"] == 0] if "collision_stop" in df else df
    degs = [d for d in ["clean", "range20", "range10", "drop75", "noise01"]
            if d in set(df["degradation"])] or sorted(set(df["degradation"]))
    combos = sorted(set(df["combo"]))
    colors = dict(zip(combos, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                               "#8c564b", "#e377c2"][:len(combos)]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    series = sorted(set(zip(df["combo"], df["safety"])))
    width = 0.8 / max(len(series), 1)
    x = np.arange(len(degs))
    for k, (combo, safety) in enumerate(series):
        sub = df[(df["combo"] == combo) & (df["safety"] == safety)]
        succ = [float(sub[sub["degradation"] == d]["progress_ratio"].mean())
                if len(sub[sub["degradation"] == d]) else np.nan for d in degs]
        coll = [float(sub[sub["degradation"] == d][coll_col].mean())
                if len(sub[sub["degradation"] == d]) else np.nan for d in degs]
        mrm = [float(sub[sub["degradation"] == d]["mrm_time"].mean())
               if len(sub[sub["degradation"] == d]) else np.nan for d in degs]
        label = combo if safety == "aware" else f"{combo} (blind)"
        hatch = None if safety == "aware" else "//"
        for ax, vals in [(axes[0, 0], succ), (axes[0, 1], coll), (axes[1, 0], mrm)]:
            ax.bar(x + k * width, vals, width, color=colors[combo],
                   hatch=hatch, edgecolor="white", label=label)

    for ax, title, ylabel in [
            (axes[0, 0], "Task progress", "fraction of route"),
            (axes[0, 1], "Moving collision rate", "rate"),
            (axes[1, 0], "Time in MRM", "seconds")]:
        ax.set_xticks(x + width * (len(series) - 1) / 2)
        ax.set_xticklabels(degs, rotation=15)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)

    ax = axes[1, 1]
    for combo in combos:
        sub = scatter_df[scatter_df["combo"] == combo]
        for safety, marker in [("aware", "o"), ("blind", "x")]:
            s = sub[sub["safety"] == safety]
            if len(s):
                ax.scatter(s["mean_speed"], s["min_gap"], marker=marker, s=45,
                           color=colors[combo], alpha=0.8,
                           label=f"{combo} ({safety})")
    ax.set_xlabel("mean speed (m/s)"); ax.set_ylabel("minimum gap (m)")
    ax.set_title("Safety-efficiency trade-off"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_comparison(args.metrics, os.path.join(args.out, "closed_loop_comparison.png"))
    print("wrote", args.out)


if __name__ == "__main__":
    _main()
