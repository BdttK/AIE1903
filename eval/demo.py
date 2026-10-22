#!/usr/bin/env python3
"""Minimal dynamic demo: replay a scene, inject a crossing pedestrian, and
compare perception-only / map-only / degradation-aware policies."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import (  # noqa: E402
    BOTTOM, SceneData, quat_to_rot,
)
from eval.core import (  # noqa: E402
    Grid, _ann_boxes_in_ego, build_layers, degrade, global_to_ego,
    obb_overlap, rectangle_corners,
)
from eval.policy import scan_forward, speed_command  # noqa: E402
from eval.experiments import parse_name  # noqa: E402

POLICIES = ["perception", "map", "degradation_aware"]
POLICY_COLORS = {"perception": "#1f77b4", "map": "#ff7f0e",
                 "degradation_aware": "#2ca02c"}
PED_SIZE = (0.6, 0.6, 1.7)
EGO_SIZE = (4.7, 1.8)


def yaw_of(pose):
    rot = quat_to_rot(pose["rotation"])
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def box_corners_3d(center, heading, length, width, height):
    foot = rectangle_corners(center, heading, length, width)
    corners = np.zeros((8, 3))
    corners[[1, 3, 7, 5], :2] = foot
    corners[[0, 2, 6, 4], :2] = foot
    corners[[0, 2, 6, 4], 2] = height
    return corners


def pedestrian_points(center, heading, rng, size=PED_SIZE):
    length, width, height = size
    foot = rectangle_corners(center, heading, length, width)
    pts = []
    for z in (0.4, 0.9, 1.4):
        for i in range(4):
            a, b = foot[i], foot[(i + 1) % 4]
            n = max(2, int(np.linalg.norm(b - a) / 0.08))
            for k in range(n):
                p = a + (b - a) * (k / n)
                pts.append([p[0], p[1], z])
    pts = np.array(pts, dtype=float)
    out = np.zeros((len(pts), 5), dtype=np.float32)
    out[:, :3] = pts
    out[:, 3] = 100.0
    out[:, 4] = 0.0
    out[:, :2] += rng.normal(0.0, 0.01, (len(out), 2))
    return out


class Replay:
    def __init__(self, root, version, scene_name):
        self.sc = SceneData(root, version, scene_name)
        self.name = scene_name
        self.tokens = self.sc.sample_tokens
        self.poses = [self.sc.ego_pose(t) for t in self.tokens]
        self.xy = np.array([p["translation"][:2] for p in self.poses], dtype=float)
        self.times = np.array([self.sc.samples[t]["timestamp"] for t in self.tokens],
                              dtype=float) / 1e6
        self.t0 = float(self.times[0])
        self.times -= self.t0
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        yaw = np.unwrap(np.arctan2(np.diff(self.xy[:, 1]), np.diff(self.xy[:, 0])))
        self.yaw = np.r_[yaw, yaw[-1]] if len(yaw) else np.zeros(1)
        dt = np.maximum(np.diff(self.times), 1e-3)
        self.speeds = np.r_[np.diff(self.s) / dt, np.diff(self.s)[-1] / dt[-1]] \
            if len(self.s) > 1 else np.zeros(1)
        self.cam = [os.path.join(self.sc.root,
                                 self.sc.sd_by_sample[t]["CAM_FRONT"]["filename"])
                    for t in self.tokens]
        self.anns = [self.sc.ann_by_sample[t] for t in self.tokens]

        self.sweeps = []
        for rec in self.sc.scene_sample_data:
            if rec["filename"].split("/")[1] != "LIDAR_TOP":
                continue
            t = rec["timestamp"] / 1e6 - self.t0
            sw = {
                "filename": rec["filename"],
                "pose": self.sc.ego_poses[rec["ego_pose_token"]],
                "calib": self.sc.calib[rec["calibrated_sensor_token"]],
                "t": t,
                "s": float(np.interp(t, self.times, self.s)),
            }
            self.sweeps.append(sw)
        self.sweeps.sort(key=lambda x: x["s"])
        self.sweep_s = np.array([x["s"] for x in self.sweeps])
        self._cache = {}

    def index_at(self, s):
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        return min(max(i, 0), len(self.s) - 1)

    def index_at_time(self, t):
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        return min(max(i, 0), len(self.times) - 1)

    def yaw_at(self, s):
        i = self.index_at(s)
        j = min(i + 1, len(self.s) - 1)
        span = max(self.s[j] - self.s[i], 1e-9)
        frac = float(np.clip((s - self.s[i]) / span, 0.0, 1.0))
        return float((1 - frac) * self.yaw[i] + frac * self.yaw[j])

    def point_at(self, s, lateral=0.0):
        i = self.index_at(s)
        j = min(i + 1, len(self.s) - 1)
        span = max(self.s[j] - self.s[i], 1e-9)
        frac = float(np.clip((s - self.s[i]) / span, 0.0, 1.0))
        xy = (1 - frac) * self.xy[i] + frac * self.xy[j]
        yaw = (1 - frac) * self.yaw[i] + frac * self.yaw[j]
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        return xy + normal * lateral

    def time_at(self, s):
        return float(np.interp(s, self.s, self.times))

    def sweep_at(self, s):
        i = int(np.searchsorted(self.sweep_s, s))
        if i >= len(self.sweeps):
            i = len(self.sweeps) - 1
        elif i > 0 and abs(self.sweep_s[i - 1] - s) < abs(self.sweep_s[i] - s):
            i -= 1
        return self.sweeps[i]

    def sweep_points(self, sw):
        key = sw["filename"]
        if key not in self._cache:
            pts = np.fromfile(os.path.join(self.sc.root, key),
                              dtype=np.float32).reshape(-1, 5)
            self._cache[key] = pts
            if len(self._cache) > 48:
                self._cache.pop(next(iter(self._cache)))
        pts = self._cache[key]
        rc = quat_to_rot(sw["calib"]["rotation"])
        tc = np.asarray(sw["calib"]["translation"])
        return pts[:, :3] @ rc.T + tc


def pedestrian_state(replay, t, inject_s, t_eta, speed=1.0):
    y = float(np.clip(speed * (t - t_eta), -4.0, 4.0))
    center = replay.point_at(inject_s, y)
    return center, replay.yaw_at(inject_s), y


def simulate(replay, grid, policy, degradation, inject_s=18.0,
             ped_speed=1.0, dt=0.1, max_steps=260, seed=0,
             start_s=0.0, target_speed=8.0, start_speed=0.0):
    kind, param = parse_name(degradation)
    rng = np.random.default_rng(seed)
    s, t = float(start_s), 0.0
    v = float(start_speed)
    t_eta = ((inject_s - start_s) / max(target_speed, 0.5)
             + target_speed / (2.0 * 1.5))
    records = []
    first_collision = None
    crashed = False

    for step in range(max_steps):
        sw = replay.sweep_at(s)
        s = sw["s"]
        render_pose = sw["pose"]
        idx = replay.index_at_time(sw["t"])
        pts = replay.sweep_points(sw)

        pg, yaw_g, y = pedestrian_state(replay, t, inject_s, t_eta, ped_speed)
        c_sim = global_to_ego(np.asarray(pg), render_pose)
        yaw_sim = yaw_g - yaw_of(render_pose)
        pts = np.vstack([pts, pedestrian_points(c_sim, yaw_sim, rng)[:, :3]])
        pts = degrade(pts, kind, param, rng)

        layers = build_layers(pts, grid)
        scan = scan_forward(layers, grid)
        v_target = float(target_speed)
        v_cmd = 0.0 if crashed else speed_command(policy, scan, v, v_target, dt)
        v = max(0.0, v_cmd)
        s = min(s + v * dt, replay.s[-1])
        t += dt

        sw2 = replay.sweep_at(s)
        coll_pose = sw2["pose"]
        pg2, yaw_g2, y2 = pedestrian_state(replay, t, inject_s, t_eta, ped_speed)
        c2 = global_to_ego(np.asarray(pg2), coll_pose)
        foot = rectangle_corners(c2, yaw_g2 - yaw_of(coll_pose),
                                 PED_SIZE[0], PED_SIZE[1])
        ego = rectangle_corners((0.0, 0.0), 0.0, *EGO_SIZE)
        collision = bool(obb_overlap(ego, foot))
        clearance = float(np.linalg.norm(c2) - 1.2)
        if collision:
            crashed = True
            if first_collision is None:
                first_collision = t

        pg_r, yaw_r, _ = pedestrian_state(replay, t, inject_s, t_eta, ped_speed)
        c_r = global_to_ego(np.asarray(pg_r), render_pose)
        code = np.zeros(grid.shape, dtype=np.uint8)
        code[layers["occupied"]] = 1
        code[layers["unknown"]] = 2
        records.append({
            "step": step, "t": t, "s": s, "v": v, "v_target": v_target,
            "policy": policy, "keyframe": idx,
            "occ_dist": scan["occ_dist"], "unk_dist": scan["unk_dist"],
            "free_dist": scan["free_dist"],
            "unknown_ratio": scan["unknown_ratio"],
            "occ_ratio": scan["occupied_ratio"],
            "ped_y": y2, "clearance": clearance, "collision": collision,
            "first_collision": first_collision is not None,
            "pose": render_pose,
            "ped_center": np.asarray(c_r, dtype=float).copy(),
            "ped_yaw": float(yaw_r - yaw_of(render_pose)),
            "layer_code": code,
            "points": np.asarray(pts[::10, :3], dtype=float).copy(),
        })
        if s >= replay.s[-1] - 0.05:
            break

    return records


def _layer_image(layers, grid):
    img = np.zeros(grid.shape + (3,))
    img[layers["free"]] = (0.72, 0.86, 0.95)
    img[layers["occupied"]] = (0.05, 0.05, 0.05)
    img[layers["unknown"]] = (0.65, 0.65, 0.65)
    return img


def render_frames(replay, grid, records, out_dir, degradation, fps=10, dpi=110):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ext = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    ax_time = axes[1, 1]
    ax_time2 = ax_time.twinx()

    for k, rec in enumerate(records):
        for ax in axes.ravel():
            ax.clear()
        ax_time2.clear()
        sim_pose = rec["pose"]
        idx = rec["keyframe"]

        ax = axes[0, 0]
        with Image.open(replay.cam[idx]) as im:
            ax.imshow(np.asarray(im.convert("RGB")))
        ax.set_title("CAM_FRONT (keyframe %d)" % idx, fontsize=9)
        ax.axis("off")
        info = ("t=%.1fs  v=%.1f m/s\npolicy: %s\nfree=%.1fm  unknown=%.0f%%"
                % (rec["t"], rec["v"], rec["policy"], rec["free_dist"],
                   100 * rec["unknown_ratio"]))
        if rec["first_collision"]:
            info += "\nCOLLISION"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", fontsize=9,
                color="white", bbox=dict(facecolor="black", alpha=0.65))
        ax.text(0.02, 0.02, "virtual pedestrian: BEV only", transform=ax.transAxes,
                fontsize=7, color="yellow")

        ax = axes[0, 1]
        p = rec["points"]
        m = (np.abs(p[:, 0]) < 40) & (np.abs(p[:, 1]) < 20)
        p = p[m]
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="turbo", s=0.6,
                       vmin=-0.5, vmax=3, rasterized=True)
        for corners, cat, _ in _ann_boxes_in_ego(replay.sc, replay.anns[idx], sim_pose):
            foot = corners[BOTTOM][:, :2]
            if np.abs(foot).max() > 42:
                continue
            ax.add_patch(Polygon(foot, closed=True, fill=False,
                                 edgecolor="#888888", linewidth=0.6))
        foot = rectangle_corners(rec["ped_center"], rec["ped_yaw"],
                                 PED_SIZE[0], PED_SIZE[1])
        ax.add_patch(Polygon(foot, closed=True, fill=True, facecolor="#d62728",
                             alpha=0.75, edgecolor="#d62728"))
        ax.add_patch(Rectangle((-2.35, -0.9), 4.7, 1.8, facecolor="black", alpha=0.55))
        ax.set_xlim(-10, 40)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title("degraded LiDAR BEV + GT boxes + virtual pedestrian", fontsize=9)
        ax.tick_params(labelsize=7)

        ax = axes[1, 0]
        code = rec["layer_code"]
        img = np.zeros(grid.shape + (3,))
        img[code == 0] = (0.72, 0.86, 0.95)
        img[code == 1] = (0.05, 0.05, 0.05)
        img[code == 2] = (0.65, 0.65, 0.65)
        ax.imshow(img, origin="lower", extent=ext)
        ax.plot([0, 30], [1.2, 1.2], color="#2ca02c", linewidth=0.8)
        ax.plot([0, 30], [-1.2, -1.2], color="#2ca02c", linewidth=0.8)
        ax.axvline(rec["occ_dist"], color="#d62728", linestyle="--", linewidth=1)
        ax.axvline(rec["unk_dist"], color="#7f7f7f", linestyle=":", linewidth=1)
        ax.set_xlim(-10, 35)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title("occupancy + forward corridor (red=occ, gray=unknown)",
                     fontsize=9)
        ax.tick_params(labelsize=7)

        tt = [r["t"] for r in records[:k + 1]]
        vv = [r["v"] for r in records[:k + 1]]
        fd = [r["free_dist"] for r in records[:k + 1]]
        ax_time.plot(tt, vv, color=POLICY_COLORS.get(rec["policy"], "#333"),
                     linewidth=1.6, label="speed")
        ax_time.set_ylim(0, 9)
        ax_time.set_xlim(0, max(records[-1]["t"], 1.0))
        ax_time.set_xlabel("time (s)", fontsize=8)
        ax_time.set_ylabel("speed (m/s)", fontsize=8)
        ax_time.grid(alpha=0.3)
        ax_time2.plot(tt, fd, color="#7f7f7f", linewidth=1.0, linestyle="--",
                      label="free dist")
        ax_time2.set_ylim(0, 32)
        ax_time2.set_ylabel("free distance (m)", fontsize=8)
        if rec["first_collision"]:
            ax_time.axvline(rec["t"], color="#d62728", linewidth=1.0)
        ax_time.set_title("timeline: speed / forward free distance", fontsize=9)
        ax_time.tick_params(labelsize=7)
        ax_time2.tick_params(labelsize=7)

        fig.suptitle("%s | %s | %s | t=%.1fs"
                     % (replay.name, degradation, rec["policy"], rec["t"]),
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "%04d.png" % k), dpi=dpi)
    plt.close(fig)


def frames_to_video(frame_dir, out_path, fps=10):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frame_dir, "%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out_path]
    subprocess.run(cmd, check=True)


def summarize_metrics(records, policy):
    v = np.array([r["v"] for r in records])
    t = np.array([r["t"] for r in records])
    first = next((r for r in records if r["collision"]), None)
    pre = records[:records.index(first) + 1] if first else records
    v_pre = np.array([r["v"] for r in pre])
    stops = int((v_pre < 0.5).sum())
    return {
        "policy": policy,
        "n_steps": len(records),
        "duration_s": float(t[-1]) if len(t) else 0.0,
        "distance_m": float(records[-1]["s"]) if records else 0.0,
        "avg_speed": float(v.mean()) if len(v) else 0.0,
        "max_speed": float(v.max()) if len(v) else 0.0,
        "collided": bool(first),
        "first_collision_s": float(first["t"]) if first else None,
        "min_clearance_m": float(min(r["clearance"] for r in records))
        if records else None,
        "stop_steps_pre": stops,
        "stopped_ratio_pre": float(stops / max(len(pre), 1)),
    }


def plot_comparison(all_records, metrics, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for policy in POLICIES:
        recs = all_records.get(policy)
        if not recs:
            continue
        t = [r["t"] for r in recs]
        axes[0].plot(t, [r["v"] for r in recs], color=POLICY_COLORS[policy],
                     label=policy)
        axes[1].plot(t, [r["free_dist"] for r in recs],
                     color=POLICY_COLORS[policy], label=policy)
        axes[2].plot(t, [r["clearance"] for r in recs],
                     color=POLICY_COLORS[policy], label=policy)
        first = next((r for r in recs if r["collision"]), None)
        if first:
            axes[0].axvline(first["t"], color=POLICY_COLORS[policy],
                            linestyle=":", linewidth=1, alpha=0.7)
            axes[2].scatter([first["t"]], [first["clearance"]],
                            color=POLICY_COLORS[policy], marker="X", s=90,
                            zorder=5)
            axes[2].annotate("collision", (first["t"], first["clearance"]),
                             textcoords="offset points", xytext=(4, 6),
                             fontsize=7, color=POLICY_COLORS[policy])
    titles = ["speed (m/s)", "forward free distance (m)", "clearance to pedestrian (m)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--degradation", default="range10")
    ap.add_argument("--inject-distance", type=float, default=18.0)
    ap.add_argument("--ped-speed", type=float, default=1.0)
    ap.add_argument("--start-distance", type=float, default=0.0)
    ap.add_argument("--target-speed", type=float, default=8.0)
    ap.add_argument("--start-speed", type=float, default=0.0)
    ap.add_argument("--policies", nargs="+", default=POLICIES,
                    choices=POLICIES)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="outputs/eval_results/demo")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    grid = Grid(res=0.2, x_min=-40, x_max=40, y_min=-20, y_max=20)
    replay = Replay(args.root, args.version, args.scene)
    os.makedirs(args.out, exist_ok=True)

    all_records, metrics = {}, []
    for policy in args.policies:
        print("simulating", policy, flush=True)
        records = simulate(replay, grid, policy, args.degradation,
                           args.start_distance + args.inject_distance,
                           args.ped_speed, args.dt, args.max_steps,
                           start_s=args.start_distance,
                           target_speed=args.target_speed,
                           start_speed=args.start_speed)
        all_records[policy] = records
        metrics.append(summarize_metrics(records, policy))
        if not args.no_video:
            frame_dir = os.path.join(args.out, "frames_%s" % policy)
            print("rendering", policy, len(records), "frames", flush=True)
            render_frames(replay, grid, records, frame_dir, args.degradation, args.fps)
            video = os.path.join(args.out, "demo_%s_%s_%s.mp4"
                                 % (args.scene, args.degradation, policy))
            frames_to_video(frame_dir, video, args.fps)
            shutil.rmtree(frame_dir, ignore_errors=True)
            print("saved", video, flush=True)

    tag = "%s_%s" % (args.scene, args.degradation)
    with open(os.path.join(args.out, "metrics_%s.csv" % tag), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)
    plot_comparison(all_records, metrics,
                    os.path.join(args.out, "comparison_%s.png" % tag))
    for m in metrics:
        print(m)


if __name__ == "__main__":
    main()
