#!/usr/bin/env python3
"""Visualize one scene of nuScenes v1.0-mini.

Outputs: dashboard (6 cameras + boxes, LiDAR BEV, ego trajectory, stats),
front-camera close-up, BEV sequence for all keyframes, and a BEV GIF.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
GROUP_COLORS = {
    "vehicle": "#377eb8",
    "human": "#e41a1c",
    "movable_object": "#ff7f00",
    "static_object": "#4daf4a",
}
BOX_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]
BOTTOM = [1, 3, 7, 5]


def color_of(category):
    return GROUP_COLORS.get(category.split(".")[0], "#777777")


def load_json(meta_dir, name):
    with open(os.path.join(meta_dir, name + ".json")) as f:
        return json.load(f)


def quat_to_rot(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def box_corners(translation, size, rotation):
    width, length, height = size
    sx = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * length / 2.0
    sy = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * width / 2.0
    sz = np.array([1, -1, 1, -1, 1, -1, 1, -1]) * height / 2.0
    local = np.stack([sx, sy, sz])
    return (quat_to_rot(rotation) @ local).T + np.asarray(translation)


class SceneData:
    def __init__(self, root, version, scene_name):
        self.root = root
        self.meta = os.path.join(root, version)
        scenes = load_json(self.meta, "scene")
        names = [s["name"] for s in scenes]
        if scene_name is None:
            scene_name = names[0]
        scene = next((s for s in scenes if s["name"] == scene_name), None)
        if scene is None:
            raise SystemExit("Unknown scene %r. Available: %s" % (scene_name, names))
        self.scene = scene
        self.name = scene_name

        self.samples = {s["token"]: s for s in load_json(self.meta, "sample")}
        self.logs = {l["token"]: l for l in load_json(self.meta, "log")}
        self.log = self.logs[scene["log_token"]]
        self.ego_poses = {e["token"]: e for e in load_json(self.meta, "ego_pose")}
        self.calib = {c["token"]: c for c in load_json(self.meta, "calibrated_sensor")}
        self.categories = {c["token"]: c["name"] for c in load_json(self.meta, "category")}
        self.instances = {i["token"]: i for i in load_json(self.meta, "instance")}

        self.sample_tokens = []
        token = scene["first_sample_token"]
        while token:
            self.sample_tokens.append(token)
            token = self.samples[token]["next"]
        sample_set = set(self.sample_tokens)

        self.scene_sample_data = []
        self.sd_by_sample = defaultdict(dict)
        for rec in load_json(self.meta, "sample_data"):
            if rec["sample_token"] not in sample_set:
                continue
            self.scene_sample_data.append(rec)
            if rec["is_key_frame"]:
                channel = rec["filename"].split("/")[1]
                self.sd_by_sample[rec["sample_token"]][channel] = rec

        self.ann_by_sample = defaultdict(list)
        for ann in load_json(self.meta, "sample_annotation"):
            if ann["sample_token"] not in sample_set:
                continue
            ann["_category"] = self.categories[self.instances[ann["instance_token"]]["category_token"]]
            ann["_corners"] = box_corners(ann["translation"], ann["size"], ann["rotation"])
            self.ann_by_sample[ann["sample_token"]].append(ann)

        self.map_file = None
        for m in load_json(self.meta, "map"):
            if scene["log_token"] in m["log_tokens"]:
                self.map_file = m["filename"]
                break

    # ---------- geometry ----------
    def lidar_ego(self, sample_token):
        rec = self.sd_by_sample[sample_token]["LIDAR_TOP"]
        path = os.path.join(self.root, rec["filename"])
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        cal = self.calib[rec["calibrated_sensor_token"]]
        rot = quat_to_rot(cal["rotation"])
        trans = np.asarray(cal["translation"])
        return points, points[:, :3] @ rot.T + trans

    def boxes_ego(self, sample_token):
        rec = self.sd_by_sample[sample_token]["LIDAR_TOP"]
        pose = self.ego_poses[rec["ego_pose_token"]]
        rot = quat_to_rot(pose["rotation"])
        trans = np.asarray(pose["translation"])
        out = []
        for ann in self.ann_by_sample[sample_token]:
            corners = (ann["_corners"] - trans) @ rot
            out.append((corners, ann["_category"]))
        return out

    def ego_pose(self, sample_token):
        rec = self.sd_by_sample[sample_token]["CAM_FRONT"]
        return self.ego_poses[rec["ego_pose_token"]]

    def camera_segments(self, sample_token, channel):
        rec = self.sd_by_sample[sample_token][channel]
        pose = self.ego_poses[rec["ego_pose_token"]]
        cal = self.calib[rec["calibrated_sensor_token"]]
        rot_e = quat_to_rot(pose["rotation"])
        trans_e = np.asarray(pose["translation"])
        rot_c = quat_to_rot(cal["rotation"])
        trans_c = np.asarray(cal["translation"])
        k = np.asarray(cal["camera_intrinsic"], dtype=float)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

        segments = []
        for ann in self.ann_by_sample[sample_token]:
            corners_ego = (ann["_corners"] - trans_e) @ rot_e
            corners_cam = (corners_ego - trans_c) @ rot_c
            uv = np.full((8, 2), np.nan)
            front = corners_cam[:, 2] > 0.5
            uv[front, 0] = fx * corners_cam[front, 0] / corners_cam[front, 2] + cx
            uv[front, 1] = fy * corners_cam[front, 1] / corners_cam[front, 2] + cy
            color = color_of(ann["_category"])
            for i, j in BOX_EDGES:
                if front[i] and front[j]:
                    segments.append((np.array([uv[i], uv[j]]), color))
        return segments

    # ---------- plotting ----------
    def plot_bev(self, ax, sample_token, limit=50, point_size=1.2, legend=True):
        _, points = self.lidar_ego(sample_token)
        mask = (
            (np.abs(points[:, 0]) < limit)
            & (np.abs(points[:, 1]) < limit)
            & (points[:, 2] > -1.5)
            & (points[:, 2] < 4.0)
        )
        p = points[mask]
        ax.scatter(
            p[:, 0], p[:, 1], c=p[:, 2], cmap="turbo", vmin=-0.5, vmax=3.0,
            s=point_size, linewidths=0, alpha=0.9, rasterized=True,
        )

        for corners, category in self.boxes_ego(sample_token):
            foot = corners[BOTTOM][:, :2]
            if np.abs(foot.mean(axis=0)).max() > limit + 8:
                continue
            color = color_of(category)
            ax.add_patch(Polygon(foot, closed=True, fill=False, edgecolor=color, linewidth=1.0))
            center = foot.mean(axis=0)
            front = corners[[1, 3]][:, :2].mean(axis=0)
            ax.plot([center[0], front[0]], [center[1], front[1]], color=color, linewidth=0.8)

        ax.add_patch(Rectangle((-1.0, -0.9), 4.7, 1.8, facecolor="black", alpha=0.55,
                               edgecolor="black", linewidth=0.8))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlabel("x forward (m)", fontsize=8)
        ax.set_ylabel("y left (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        if legend:
            handles = [
                Line2D([], [], color=GROUP_COLORS["vehicle"], label="vehicle"),
                Line2D([], [], color=GROUP_COLORS["human"], label="human"),
                Line2D([], [], color=GROUP_COLORS["movable_object"], label="movable"),
                Line2D([], [], color=GROUP_COLORS["static_object"], label="static"),
                Rectangle((0, 0), 1, 1, facecolor="black", alpha=0.55, label="ego"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)

    def plot_trajectory(self, ax):
        xy = np.array([[self.ego_pose(t)["translation"][0],
                        self.ego_pose(t)["translation"][1]] for t in self.sample_tokens])
        dist = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        ax.plot(xy[:, 0], xy[:, 1], color="#888888", linewidth=1.2, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="plasma",
                   s=28, zorder=2)
        ax.scatter(*xy[0], c="lime", edgecolor="black", s=110, marker="o", zorder=3, label="start")
        ax.scatter(*xy[-1], c="red", edgecolor="black", s=110, marker="X", zorder=3, label="end")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("global x (m)", fontsize=8)
        ax.set_ylabel("global y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="best")
        ax.set_title("Ego trajectory (%.1f m)" % dist, fontsize=11)

    def plot_stats(self, ax):
        counts = Counter()
        for token in self.sample_tokens:
            for ann in self.ann_by_sample[token]:
                counts[ann["_category"]] += 1
        top = counts.most_common(8)[::-1]
        labels = [name.replace("human.pedestrian.", "ped.").replace("movable_object.", "mov.")
                  for name, _ in top]
        colors = [color_of(name) for name, _ in top]
        ax.barh(labels, [n for _, n in top], color=colors, alpha=0.85)
        ax.set_xlabel("boxes in scene", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)
        total = sum(counts.values())
        n_inst = len({a["instance_token"] for t in self.sample_tokens
                      for a in self.ann_by_sample[t]})
        ax.set_title("Annotations: %d boxes / %d instances" % (total, n_inst), fontsize=11)

    def file_stats(self):
        stats = defaultdict(lambda: [0, 0])
        for rec in self.scene_sample_data:
            path = os.path.join(self.root, rec["filename"])
            size = os.path.getsize(path) if os.path.exists(path) else 0
            group = rec["filename"].split("/")[1]
            if group.startswith("CAM_"):
                group = "CAMERAS"
            elif group == "LIDAR_TOP":
                group = "LIDAR"
            else:
                group = "RADAR"
            stats[group][0] += 1
            stats[group][1] += size
        return stats


_MAP_CACHE = {}
MAP_RES = 0.1  # metres per pixel in the original map PNG


def load_map(path, factor=5):
    key = (path, factor)
    if key not in _MAP_CACHE:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            im = im.convert("L")
            width, height = im.size
            small = im.resize((width // factor, height // factor), Image.BILINEAR)
        _MAP_CACHE[key] = (np.asarray(small), width * MAP_RES, height * MAP_RES)
    return _MAP_CACHE[key]


def trajectory_xy(sc):
    return np.array([[sc.ego_pose(t)["translation"][0],
                      sc.ego_pose(t)["translation"][1]] for t in sc.sample_tokens])


def map_crop(sc, margin=100.0, factor=5):
    array, x_max, y_max = load_map(os.path.join(sc.root, sc.map_file), factor)
    xy = trajectory_xy(sc)
    x0 = max(0.0, float(xy[:, 0].min()) - margin)
    x1 = min(x_max, float(xy[:, 0].max()) + margin)
    y0 = max(0.0, float(xy[:, 1].min()) - margin)
    y1 = min(y_max, float(xy[:, 1].max()) + margin)
    span = 200.0
    if x1 - x0 < span:
        x0 = max(0.0, min(x0, x_max - span))
        x1 = min(x_max, x0 + span)
    if y1 - y0 < span:
        y0 = max(0.0, min(y0, y_max - span))
        y1 = min(y_max, y0 + span)

    scale = MAP_RES * factor
    c0 = max(0, min(int(round(x0 / scale)), array.shape[1] - 1))
    c1 = max(c0 + 1, min(int(round(x1 / scale)), array.shape[1]))
    r0 = max(0, min(int(round((y_max - y1) / scale)), array.shape[0] - 1))
    r1 = max(r0 + 1, min(int(round((y_max - y0) / scale)), array.shape[0]))
    extent = (c0 * scale, c1 * scale, y_max - r1 * scale, y_max - r0 * scale)
    return array[r0:r1, c0:c1], extent, xy


def render_map_trajectory(sc, out_path):
    crop, extent, xy = map_crop(sc)
    travel = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() if len(xy) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
    ax.plot(xy[:, 0], xy[:, 1], color="#ff3355", linewidth=1.2, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="plasma", s=14, zorder=3)
    ax.scatter(*xy[0], c="lime", edgecolor="black", s=90, marker="o", zorder=4, label="start")
    ax.scatter(*xy[-1], c="red", edgecolor="black", s=90, marker="X", zorder=4, label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("%s | %s | route %.1f m" % (sc.name, sc.log["location"], travel), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_maps_overview(summaries, out_path):
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["location"], summary["map_path"])].append(summary)

    fig, axes = plt.subplots(2, 2, figsize=(22, 17))
    for ax, ((location, map_path), items) in zip(axes.flat, sorted(groups.items())):
        array, x_max, y_max = load_map(map_path, 5)
        ax.imshow(array[::2, ::2], cmap="gray", extent=(0, x_max, 0, y_max),
                  origin="upper", interpolation="nearest")
        colors = plt.cm.tab10.colors
        for i, summary in enumerate(items):
            xy = summary["xy"]
            color = colors[i % len(colors)]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()],
                    label=summary["name"], zorder=4)
            ax.scatter(*xy[0], color=color, edgecolor="black", linewidth=0.6, s=55, zorder=5)
            ax.scatter(*xy[-1], color=color, marker="X", edgecolor="black", linewidth=0.6, s=75, zorder=5)
            ax.annotate(summary["name"], xy[0], xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", edgecolor="none", alpha=0.65))
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("global x (m)")
        ax.set_ylabel("global y (m)")
        ax.set_title("%s | %d scene(s)" % (location, len(items)), fontsize=13)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("HD map coverage and scene routes (white = drivable area, 1 m/px)", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


_VECTOR_CACHE = {}


class VectorMap:
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)
        self.nodes = {n["token"]: (n["x"], n["y"]) for n in self.data.get("node", [])}
        self.polygons = {p["token"]: p for p in self.data.get("polygon", [])}
        self.lines = {l["token"]: l for l in self.data.get("line", [])}
        self._xy = {}

    def polygon(self, token):
        key = ("p", token)
        if key not in self._xy:
            poly = self.polygons.get(token)
            pts = [self.nodes[t] for t in poly["exterior_node_tokens"]
                   if poly and t in self.nodes] if poly else []
            self._xy[key] = np.asarray(pts) if len(pts) >= 3 else None
        return self._xy[key]

    def line(self, token):
        key = ("l", token)
        if key not in self._xy:
            line = self.lines.get(token)
            pts = [self.nodes[t] for t in line["node_tokens"]
                   if line and t in self.nodes] if line else []
            self._xy[key] = np.asarray(pts) if len(pts) >= 2 else None
        return self._xy[key]


def load_vector_map(path):
    if path not in _VECTOR_CACHE:
        _VECTOR_CACHE[path] = VectorMap(path)
    return _VECTOR_CACHE[path]


def _in_view(points, xlim, ylim):
    return not (points[:, 0].max() < xlim[0] or points[:, 0].min() > xlim[1]
                or points[:, 1].max() < ylim[0] or points[:, 1].min() > ylim[1])


def draw_vector_map(ax, vm, xlim, ylim, detail=True, legend=False):
    def polygon(token, **kwargs):
        pts = vm.polygon(token)
        if pts is not None and _in_view(pts, xlim, ylim):
            ax.add_patch(Polygon(pts, closed=True, **kwargs))

    for area in vm.data.get("drivable_area", []):
        for token in area.get("polygon_tokens", []):
            polygon(token, facecolor="#f2f2f2", edgecolor="none", zorder=0)

    for item in vm.data.get("road_segment", []):
        polygon(item["polygon_token"], facecolor="#e9e9e9", edgecolor="#d2d2d2",
                linewidth=0.3, zorder=1)

    if detail:
        for item in vm.data.get("walkway", []):
            polygon(item["polygon_token"], facecolor="#f7edd8", edgecolor="#e7d9bd",
                    linewidth=0.2, zorder=2)
        for item in vm.data.get("carpark_area", []):
            polygon(item["polygon_token"], facecolor="#e3f2e1", edgecolor="#cfe3cd",
                    linewidth=0.2, zorder=2)

    for item in vm.data.get("lane", []):
        polygon(item["polygon_token"], facecolor="#8ec6e8", edgecolor="#3b82b8",
                linewidth=0.25, alpha=0.30, zorder=3)
    for item in vm.data.get("lane_connector", []):
        polygon(item["polygon_token"], facecolor="#9fe0e0", edgecolor="#26a6a6",
                linewidth=0.4, alpha=0.12, zorder=3)
    for item in vm.data.get("ped_crossing", []):
        polygon(item["polygon_token"], facecolor="#ffd54f", edgecolor="#e0a800",
                linewidth=0.3, zorder=4)
    for item in vm.data.get("stop_line", []):
        polygon(item["polygon_token"], facecolor="none", edgecolor="#d62728",
                linewidth=0.9, zorder=5)

    if detail:
        for item in vm.data.get("lane_divider", []):
            pts = vm.line(item.get("line_token"))
            if pts is not None and _in_view(pts, xlim, ylim):
                ax.plot(pts[:, 0], pts[:, 1], color="#999999", linewidth=0.35,
                        linestyle=":", zorder=4)
        for item in vm.data.get("road_divider", []):
            pts = vm.line(item.get("line_token"))
            if pts is not None and _in_view(pts, xlim, ylim):
                ax.plot(pts[:, 0], pts[:, 1], color="#ff8c00", linewidth=0.7,
                        linestyle="--", zorder=4)

    for item in vm.data.get("traffic_light", []):
        pts = vm.line(item.get("line_token"))
        if pts is not None:
            mid = pts.mean(axis=0)
            if xlim[0] <= mid[0] <= xlim[1] and ylim[0] <= mid[1] <= ylim[1]:
                ax.scatter(mid[0], mid[1], s=10, color="#2ca02c", edgecolor="black",
                           linewidth=0.3, zorder=6)

    if legend:
        handles = [
            Line2D([], [], color="#e9e9e9", marker="s", markersize=8,
                   markeredgecolor="#d2d2d2", label="road segment"),
            Line2D([], [], color="#3b82b8", marker="s", markersize=8,
                   markerfacecolor="#8ec6e8", label="lane"),
            Line2D([], [], color="#26a6a6", marker="s", markersize=8,
                   markerfacecolor="#9fe0e0", label="lane connector"),
            Line2D([], [], color="#e0a800", marker="s", markersize=8,
                   markerfacecolor="#ffd54f", label="ped crossing"),
            Line2D([], [], color="#d62728", marker="s", markersize=8,
                   markerfacecolor="none", label="stop line"),
            Line2D([], [], color="#2ca02c", marker="o", markersize=5,
                   markeredgecolor="black", label="traffic light"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.85)


def vector_limits(sc, margin=80.0, span=220.0):
    xy = trajectory_xy(sc)
    x0 = max(0.0, float(xy[:, 0].min()) - margin)
    x1 = float(xy[:, 0].max()) + margin
    y0 = max(0.0, float(xy[:, 1].min()) - margin)
    y1 = float(xy[:, 1].max()) + margin
    if x1 - x0 < span:
        cx = (x0 + x1) / 2.0
        x0, x1 = cx - span / 2.0, cx + span / 2.0
    if y1 - y0 < span:
        cy = (y0 + y1) / 2.0
        y0, y1 = cy - span / 2.0, cy + span / 2.0
    return (max(0.0, x0), x1), (max(0.0, y0), y1)


def render_vector_map_trajectory(sc, out_path):
    vm = load_vector_map(os.path.join(sc.root, "expansion", sc.log["location"] + ".json"))
    xy = trajectory_xy(sc)
    xlim, ylim = vector_limits(sc)
    travel = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() if len(xy) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_vector_map(ax, vm, xlim, ylim, detail=True, legend=True)
    ax.plot(xy[:, 0], xy[:, 1], color="#ff1f4b", linewidth=2.2, zorder=10,
            path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()])
    ax.scatter(*xy[0], c="lime", edgecolor="black", s=90, marker="o", zorder=11, label="start")
    ax.scatter(*xy[-1], c="red", edgecolor="black", s=90, marker="X", zorder=11, label="end")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_title("%s | %s | vector HD map | route %.1f m"
                 % (sc.name, sc.log["location"], travel), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_vector_maps_overview(summaries, out_path):
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["location"], summary["vector_path"])].append(summary)

    fig, axes = plt.subplots(2, 2, figsize=(22, 17))
    for ax, ((location, map_path), items) in zip(axes.flat, sorted(groups.items())):
        vm = load_vector_map(map_path)
        width, height = vm.data["canvas_edge"]
        draw_vector_map(ax, vm, (0.0, width), (0.0, height), detail=False, legend=False)
        colors = plt.cm.tab10.colors
        for i, summary in enumerate(items):
            xy = summary["xy"]
            color = colors[i % len(colors)]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()],
                    label=summary["name"], zorder=10)
            ax.scatter(*xy[0], color=color, edgecolor="black", linewidth=0.6, s=55, zorder=11)
            ax.scatter(*xy[-1], color=color, marker="X", edgecolor="black", linewidth=0.6, s=75, zorder=11)
            ax.annotate(summary["name"], xy[0], xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color="white", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                                  edgecolor="none", alpha=0.65))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_xlabel("global x (m)")
        ax.set_ylabel("global y (m)")
        ax.set_title("%s | %d scene(s)" % (location, len(items)), fontsize=13)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Vector HD map (lane polygons, crossings, traffic lights) and scene routes", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_dashboard(sc, out_path):
    first = sc.sample_tokens[0]
    fig = plt.figure(figsize=(26, 15))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.1], hspace=0.08, wspace=0.04)

    for i, channel in enumerate(CAMERAS):
        ax = fig.add_subplot(gs[i // 3, (i % 3) * 2:(i % 3) * 2 + 2])
        rec = sc.sd_by_sample[first][channel]
        image = plt.imread(os.path.join(sc.root, rec["filename"]))
        ax.imshow(image)
        for segment, color in sc.camera_segments(first, channel):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.3, alpha=0.9)
        ax.set_xlim(0, rec["width"])
        ax.set_ylim(rec["height"], 0)
        ax.set_title(channel.replace("CAM_", "").replace("_", " "), fontsize=12)
        ax.axis("off")

    ax_bev = fig.add_subplot(gs[2, 0:2])
    sc.plot_bev(ax_bev, first)
    ax_bev.set_title("LiDAR BEV (keyframe 0, colored by height)", fontsize=12)

    ax_traj = fig.add_subplot(gs[2, 2:4])
    sc.plot_trajectory(ax_traj)

    ax_stats = fig.add_subplot(gs[2, 4:6])
    sc.plot_stats(ax_stats)

    stats = sc.file_stats()
    files = sum(v[0] for v in stats.values())
    mb = sum(v[1] for v in stats.values()) / 1048576.0
    duration = (sc.samples[sc.sample_tokens[-1]]["timestamp"]
                - sc.samples[sc.sample_tokens[0]]["timestamp"]) / 1e6
    fig.suptitle(
        "%s  |  %s %s  |  %s  |  %d keyframes / %.1f s  |  %d files / %.1f MB"
        % (sc.name, sc.log["vehicle"], sc.log["date_captured"][:10], sc.log["location"],
           len(sc.sample_tokens), duration, files, mb),
        fontsize=16,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_front_camera(sc, out_path):
    token = sc.sample_tokens[0]
    channel = "CAM_FRONT"
    rec = sc.sd_by_sample[token][channel]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(os.path.join(sc.root, rec["filename"])))
    for segment, color in sc.camera_segments(token, channel):
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2.0, alpha=0.95)
    ax.set_xlim(0, rec["width"])
    ax.set_ylim(rec["height"], 0)
    ax.set_title("%s  |  %s  |  first keyframe" % (sc.name, channel), fontsize=14)
    ax.axis("off")
    handles = [
        Line2D([], [], color=GROUP_COLORS["vehicle"], linewidth=2, label="vehicle"),
        Line2D([], [], color=GROUP_COLORS["human"], linewidth=2, label="human"),
        Line2D([], [], color=GROUP_COLORS["movable_object"], linewidth=2, label="movable"),
        Line2D([], [], color=GROUP_COLORS["static_object"], linewidth=2, label="static"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_sequence(sc, out_path):
    n = len(sc.sample_tokens)
    cols = 7
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.1))
    first_ts = sc.samples[sc.sample_tokens[0]]["timestamp"]
    for idx, token in enumerate(sc.sample_tokens):
        ax = axes.flat[idx]
        sc.plot_bev(ax, token, limit=42, point_size=0.5, legend=False)
        t = (sc.samples[token]["timestamp"] - first_ts) / 1e6
        ax.set_title("k%02d  t=%.1fs" % (idx, t), fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=5)
    for idx in range(n, rows * cols):
        axes.flat[idx].axis("off")
    fig.suptitle("%s  |  39 keyframes LiDAR BEV (boxes projected)" % sc.name, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def render_gif(sc, out_path):
    from PIL import Image

    frames = []
    first_ts = sc.samples[sc.sample_tokens[0]]["timestamp"]
    for idx, token in enumerate(sc.sample_tokens):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        sc.plot_bev(ax, token, limit=45, point_size=0.8, legend=(idx == 0))
        t = (sc.samples[token]["timestamp"] - first_ts) / 1e6
        ax.set_title("%s  |  keyframe %02d / %.1fs" % (sc.name, idx, t), fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())))
        plt.close(fig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def render_scene(sc, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stats = sc.file_stats()
    print("scene:", sc.name, "|", sc.log["location"], "|", sc.log["date_captured"][:10])
    print("keyframes:", len(sc.sample_tokens))
    print("annotations:", sum(len(v) for v in sc.ann_by_sample.values()))
    print("instances:", len({a["instance_token"] for v in sc.ann_by_sample.values() for a in v}))
    print("map:", sc.map_file)
    for group in sorted(stats):
        print("  %-8s %5d files %8.1f MB" % (group, stats[group][0], stats[group][1] / 1048576.0))

    render_dashboard(sc, os.path.join(out_dir, sc.name + "_dashboard.png"))
    render_front_camera(sc, os.path.join(out_dir, sc.name + "_cam_front_boxes.png"))
    render_sequence(sc, os.path.join(out_dir, sc.name + "_bev_sequence.png"))
    render_gif(sc, os.path.join(out_dir, sc.name + "_bev.gif"))
    render_map_trajectory(sc, os.path.join(out_dir, sc.name + "_map_trajectory.png"))
    render_vector_map_trajectory(sc, os.path.join(out_dir, sc.name + "_vector_map_trajectory.png"))

    return {
        "name": sc.name,
        "location": sc.log["location"],
        "map_path": os.path.join(sc.root, sc.map_file),
        "vector_path": os.path.join(sc.root, "expansion", sc.log["location"] + ".json"),
        "xy": trajectory_xy(sc),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join(project, "data", "nuscenes_full"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default="scene-0061")
    parser.add_argument("--out", default=os.path.join(project, "outputs", "datasets_visualize_results"))
    parser.add_argument("--all", action="store_true", help="render every scene into its own subfolder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.all:
        scene_names = [s["name"] for s in load_json(os.path.join(args.root, args.version), "scene")]
        summaries = []
        for name in scene_names:
            print("\n=== %s ===" % name)
            sc = SceneData(args.root, args.version, name)
            summaries.append(render_scene(sc, os.path.join(args.out, name)))
        render_maps_overview(summaries, os.path.join(args.out, "maps_overview.png"))
        render_vector_maps_overview(summaries, os.path.join(args.out, "vector_maps_overview.png"))
    else:
        sc = SceneData(args.root, args.version, args.scene)
        render_scene(sc, os.path.join(args.out, sc.name))
    print("\nsaved to", args.out)


if __name__ == "__main__":
    main()
