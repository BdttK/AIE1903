#!/usr/bin/env python3
"""Rasterize the vector HD map JSON into per-layer masks.

Also compares the rasterized road layer with the official map PNG.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_nuscenes_mini_scene import SceneData, trajectory_xy

Image.MAX_IMAGE_PIXELS = None

LAYERS = ["drivable", "road", "walkway", "carpark", "lane", "connector",
          "crossing", "stop_line"]

COLORS = {
    "drivable": "#f2f2f2",
    "road": "#e2e2e2",
    "walkway": "#f7edd8",
    "carpark": "#e3f2e1",
    "lane": "#8ec6e8",
    "connector": "#9fe0e0",
    "crossing": "#ffd54f",
    "stop_line": "#d62728",
}


def load_vector(path):
    with open(path) as f:
        data = json.load(f)
    nodes = {n["token"]: (n["x"], n["y"]) for n in data.get("node", [])}
    polygons = {p["token"]: p for p in data.get("polygon", [])}
    return data, nodes, polygons


def layer_tokens(data, layer):
    if layer == "drivable":
        return [t for area in data.get("drivable_area", [])
                for t in area.get("polygon_tokens", [])]
    if layer == "road":
        return [item["polygon_token"] for item in data.get("road_segment", [])]
    if layer == "lane":
        return [item["polygon_token"] for item in data.get("lane", [])]
    if layer == "connector":
        return [item["polygon_token"] for item in data.get("lane_connector", [])]
    if layer == "crossing":
        return [item["polygon_token"] for item in data.get("ped_crossing", [])]
    if layer == "stop_line":
        return [item["polygon_token"] for item in data.get("stop_line", [])]
    if layer == "walkway":
        return [item["polygon_token"] for item in data.get("walkway", [])]
    if layer == "carpark":
        return [item["polygon_token"] for item in data.get("carpark_area", [])]
    raise ValueError(layer)


def rasterize(tokens, nodes, polygons, extent, resolution):
    x0, x1, y0, y1 = extent
    width = int(round((x1 - x0) / resolution))
    height = int(round((y1 - y0) / resolution))
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for token in tokens:
        poly = polygons.get(token)
        if not poly:
            continue
        points = [nodes[t] for t in poly["exterior_node_tokens"] if t in nodes]
        if len(points) < 3:
            continue
        pixels = [((x - x0) / resolution, (y1 - y) / resolution) for x, y in points]
        draw.polygon(pixels, fill=1)
    return np.array(image, dtype=np.uint8)


def official_mask(sc, extent, shape):
    x0, x1, y0, y1 = extent
    image = Image.open(os.path.join(sc.root, sc.map_file)).convert("L")
    width, height = image.size
    box = (int(round(x0 / 0.1)), int(round(height - 1 - y1 * 10)),
           int(round(x1 / 0.1)), int(round(height - 1 - y0 * 10)))
    crop = image.crop(box).resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(crop) > 128


def render_preview(masks, extent, xy, out_path, title):
    x0, x1, y0, y1 = extent
    height, width = masks["road"].shape
    canvas = np.ones((height, width, 3), dtype=float)
    for layer in ["drivable", "walkway", "carpark", "road", "lane", "connector",
                  "crossing", "stop_line"]:
        color = np.array(matplotlib.colors.to_rgb(COLORS[layer]))
        mask = masks[layer] > 0
        if layer in ("lane", "connector"):
            canvas[mask] = 0.45 * canvas[mask] + 0.55 * color
        else:
            canvas[mask] = color

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, extent=(x0, x1, y0, y1), origin="upper", interpolation="nearest")
    ax.plot(xy[:, 0], xy[:, 1], color="#ff1f4b", linewidth=2.2, zorder=10,
            path_effects=[pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()])
    ax.scatter(*xy[0], c="lime", edgecolor="black", s=80, zorder=11)
    ax.scatter(*xy[-1], c="red", edgecolor="black", s=80, marker="X", zorder=11)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("global x (m)")
    ax.set_ylabel("global y (m)")
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=COLORS[layer], linewidth=6, label=layer)
               for layer in LAYERS]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_compare(official, road_mask, extent, iou, out_path, title):
    x0, x1, y0, y1 = extent
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(official, cmap="gray", extent=(x0, x1, y0, y1), origin="upper")
    axes[0].set_title("official PNG (white = drivable)")
    axes[1].imshow(road_mask, cmap="gray", extent=(x0, x1, y0, y1), origin="upper")
    axes[1].set_title("rasterized JSON road_segment")
    axes[2].imshow(official, cmap="gray", extent=(x0, x1, y0, y1), origin="upper")
    axes[2].imshow(np.ma.masked_where(~road_mask, road_mask), cmap=ListedColormap(["red"]),
                   alpha=0.45, extent=(x0, x1, y0, y1), origin="upper")
    axes[2].set_title("overlay | IoU = %.3f" % iou)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("global x (m)")
        ax.set_ylabel("global y (m)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join(project, "data", "nuscenes_full"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default="scene-0061")
    parser.add_argument("--resolution", type=float, default=0.2, help="metres per pixel")
    parser.add_argument("--margin", type=float, default=80.0)
    parser.add_argument("--out", default=os.path.join(project, "outputs",
                                                      "datasets_visualize_results",
                                                      "map_rasterization"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sc = SceneData(args.root, args.version, args.scene)
    xy = trajectory_xy(sc)
    vector_path = os.path.join(sc.root, "expansion", sc.log["location"] + ".json")
    data, nodes, polygons = load_vector(vector_path)

    x0 = max(0.0, float(xy[:, 0].min()) - args.margin)
    x1 = float(xy[:, 0].max()) + args.margin
    y0 = max(0.0, float(xy[:, 1].min()) - args.margin)
    y1 = float(xy[:, 1].max()) + args.margin
    span = 220.0
    if x1 - x0 < span:
        cx = (x0 + x1) / 2.0
        x0, x1 = cx - span / 2.0, cx + span / 2.0
    if y1 - y0 < span:
        cy = (y0 + y1) / 2.0
        y0, y1 = cy - span / 2.0, cy + span / 2.0
    extent = (max(0.0, x0), x1, max(0.0, y0), y1)

    masks = {layer: rasterize(layer_tokens(data, layer), nodes, polygons,
                              extent, args.resolution) for layer in LAYERS}
    official = official_mask(sc, extent, masks["road"].shape)
    iou = float((masks["road"] & official).sum() / (masks["road"] | official).sum())

    npz_path = os.path.join(args.out, sc.name + "_raster_layers.npz")
    np.savez_compressed(npz_path, **masks, extent=np.array(extent),
                        resolution=np.array(args.resolution), iou=np.array(iou))

    render_preview(masks, extent, xy,
                   os.path.join(args.out, sc.name + "_raster_preview.png"),
                   "%s | %s | rasterized vector HD map @ %.2f m/px"
                   % (sc.name, sc.log["location"], args.resolution))
    render_compare(official, masks["road"], extent, iou,
                   os.path.join(args.out, sc.name + "_raster_vs_official.png"),
                   "%s | road-layer rasterization vs official PNG" % sc.name)

    print("scene:", sc.name, "| location:", sc.log["location"])
    print("extent: x[%.1f, %.1f] y[%.1f, %.1f] m" % extent)
    print("resolution: %.2f m/px  size: %s" % (args.resolution, masks["road"].shape))
    print("road-layer IoU vs official PNG: %.3f" % iou)
    for layer in LAYERS:
        print("  %-9s %.1f%% pixels" % (layer, 100.0 * masks[layer].mean()))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
