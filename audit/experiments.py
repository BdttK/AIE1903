#!/usr/bin/env python3
"""Degradation sweep: per-frame perception and safety-envelope metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import SceneData  # noqa: E402
from audit.core import (  # noqa: E402
    Grid, boxes_to_masks, build_hazard_set, build_layers, degrade,
    hazard_recall, map_drivable, perception_metrics,
)

SCENE_META = {
    "scene-0061": {"city": "One North", "time": "day"},
    "scene-0103": {"city": "Seaport", "time": "day"},
    "scene-0553": {"city": "Seaport", "time": "day"},
    "scene-0655": {"city": "Seaport", "time": "day"},
    "scene-0757": {"city": "Seaport", "time": "day"},
    "scene-0796": {"city": "Queenstown", "time": "day"},
    "scene-0916": {"city": "Queenstown", "time": "day"},
    "scene-1077": {"city": "Holland Village", "time": "night"},
    "scene-1094": {"city": "Holland Village", "time": "night"},
    "scene-1100": {"city": "Holland Village", "time": "night"},
}

DEGRADATIONS = {
    "clean": ("clean", 0.0),
    "range40": ("range", 40.0),
    "range30": ("range", 30.0),
    "range20": ("range", 20.0),
    "range10": ("range", 10.0),
    "drop25": ("drop", 0.25),
    "drop50": ("drop", 0.50),
    "drop75": ("drop", 0.75),
    "noise005": ("noise", 0.05),
    "noise01": ("noise", 0.10),
    "noise02": ("noise", 0.20),
    "combo20_50": ("combo", (20.0, 0.50)),
}

DEFAULT_SCENES = ["scene-0061", "scene-0103", "scene-1077", "scene-1094"]
DEFAULT_DEGRADATIONS = [
    "clean", "range40", "range30", "range20", "range10",
    "drop25", "drop50", "drop75", "noise005", "noise01", "noise02",
]


def parse_name(name):
    if name not in DEGRADATIONS:
        raise SystemExit("Unknown degradation %r. Choose from: %s"
                         % (name, ", ".join(DEGRADATIONS)))
    return DEGRADATIONS[name]


def seeds_for(kind, seeds):
    return [0] if kind in ("clean", "range") else list(range(seeds))


def run_scene(scene_name, degradations, seeds, frame_step, grid,
              root, version, verbose=True):
    sc = SceneData(root, version, scene_name)
    meta = SCENE_META.get(scene_name, {"city": "", "time": ""})
    frames = list(range(0, len(sc.sample_tokens), frame_step))
    rows, hazard_rows = [], []
    t0 = time.time()



#1. 读点云 + 真值框（车体系）
#2. 算与退化无关的四件事：框栅格化、地图采样、eval_mask、包络目标集
#3. for 每档退化 × 该档的 seeds：
#     degrade 点云 → 重建三层栅格 → 算感知指标 + 包络指标 → 追加一行
#   逐目标展开：覆盖率、是否检出 → 追加 hazard 行
#4. 每 10 帧打印进度
    for n, fi in enumerate(frames):
        token = sc.sample_tokens[fi]
        _, points_ego = sc.lidar_ego(token)
        boxes = sc.boxes_ego(token)
        box_masks, gt_union = boxes_to_masks(boxes, grid)
        map_free, map_valid = map_drivable(sc, token, grid)
        X, Y = grid.centers()
        eval_mask = (np.hypot(X, Y) <= 50.0) & (map_free | gt_union)
        gt_free = map_free & ~gt_union
        hazards, _ = build_hazard_set(sc, token, grid)

        for name in degradations:
            kind, param = parse_name(name)
            for seed in seeds_for(kind, seeds):
                rng = np.random.default_rng(seed)
                pts = degrade(points_ego, kind, param, rng)
                layers = build_layers(pts, grid)
                pm = perception_metrics(layers, gt_union, box_masks, gt_free, eval_mask)
                hr = hazard_recall(hazards, layers["occupied"])
                rows.append({
                    "scene": scene_name, "city": meta["city"], "time": meta["time"],
                    "frame": fi, "token": token, "degradation": name,
                    "kind": kind, "param": str(param), "seed": seed,
                    "n_points": int(len(pts)),
                    "free_iou": pm["free_iou"],
                    "false_free_rate": pm["false_free_rate"],
                    "unknown_ratio": pm["unknown_ratio"],
                    "obstacle_recall": pm["obstacle_recall"],
                    "mean_box_coverage": pm["mean_box_coverage"],
                    "n_boxes": pm["n_boxes"],
                    "envelope_recall": hr["recall"],
                    "envelope_fsr": hr["false_safe_rate"],
                    "n_hazards": hr["n_hazards"],
                    "n_urgent": hr["n_urgent"],
                    "n_conflict": hr["n_conflict"],
                    "mean_hazard_coverage": hr["mean_coverage"],
                })
                if hazards:
                    occ = layers["occupied"]
                    for h in hazards:
                        m = h["mask"]
                        n = int(m.sum())
                        hit = int((m & occ).sum())
                        cov = hit / max(n, 1)
                        hazard_rows.append({
                            "scene": scene_name, "time": meta["time"],
                            "frame": fi, "degradation": name, "seed": seed,
                            "instance": h["instance"], "category": h["category"],
                            "s": h["s"], "d": h["d"], "ttc": h["ttc"],
                            "coverage": cov, "detected": hit >= max(1, 0.05 * n),
                        })
        if verbose and (n + 1) % 10 == 0:
            print("  %s: %d/%d frames (%.1fs)"
                  % (scene_name, n + 1, len(frames), time.time() - t0), flush=True)

    return rows, hazard_rows


def _frame_means(per_frame):
    df = pd.DataFrame(per_frame)
    return df.groupby(["scene", "city", "time", "degradation", "frame"],
                      as_index=False).agg(
        free_iou=("free_iou", "mean"),
        false_free_rate=("false_free_rate", "mean"),
        unknown_ratio=("unknown_ratio", "mean"),
        obstacle_recall=("obstacle_recall", "mean"),
        envelope_recall=("envelope_recall", "mean"),
        envelope_fsr=("envelope_fsr", "mean"),
        n_hazards=("n_hazards", "mean"),
        n_points=("n_points", "mean"),
    )


def summarize(per_frame):
    df = _frame_means(per_frame)
    agg = df.groupby(["scene", "time", "degradation"], dropna=False).agg(
        n=("frame", "count"),
        free_iou_mean=("free_iou", "mean"),
        free_iou_std=("free_iou", "std"),
        ffr_mean=("false_free_rate", "mean"),
        ffr_std=("false_free_rate", "std"),
        unknown_mean=("unknown_ratio", "mean"),
        obstacle_recall_mean=("obstacle_recall", "mean"),
        envelope_recall_mean=("envelope_recall", "mean"),
        envelope_recall_std=("envelope_recall", "std"),
        envelope_fsr_mean=("envelope_fsr", "mean"),
        n_hazards_mean=("n_hazards", "mean"),
        n_points_mean=("n_points", "mean"),
    ).reset_index()
    agg["free_iou_ci"] = 1.96 * agg["free_iou_std"] / np.sqrt(agg["n"])
    agg["envelope_recall_ci"] = 1.96 * agg["envelope_recall_std"] / np.sqrt(agg["n"])
    return agg


def summarize_by_degradation(per_frame):
    df = _frame_means(per_frame)
    agg = df.groupby(["time", "degradation"]).agg(
        n=("frame", "count"),
        free_iou_mean=("free_iou", "mean"),
        free_iou_std=("free_iou", "std"),
        ffr_mean=("false_free_rate", "mean"),
        ffr_std=("false_free_rate", "std"),
        envelope_recall_mean=("envelope_recall", "mean"),
        envelope_recall_std=("envelope_recall", "std"),
        envelope_fsr_mean=("envelope_fsr", "mean"),
    ).reset_index()
    agg["free_iou_ci"] = 1.96 * agg["free_iou_std"] / np.sqrt(agg["n"])
    agg["envelope_recall_ci"] = 1.96 * agg["envelope_recall_std"] / np.sqrt(agg["n"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--degradations", nargs="+", default=DEFAULT_DEGRADATIONS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/audit_results")
    ap.add_argument("--list-degradations", action="store_true")
    args = ap.parse_args()
    if args.list_degradations:
        for k, v in DEGRADATIONS.items():
            print(k, v)
        return

    grid = Grid(res=args.res)
    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    all_rows, all_hazards = [], []
    for scene in args.scenes:
        rows, hrows = run_scene(scene, args.degradations, args.seeds,
                                args.frame_step, grid, args.root, args.version)
        all_rows.extend(rows)
        all_hazards.extend(hrows)

    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.out, "metrics", "per_frame.csv"), index=False)
    if all_hazards:
        pd.DataFrame(all_hazards).to_csv(
            os.path.join(args.out, "metrics", "hazards.csv"), index=False)
    summarize(all_rows).to_csv(
        os.path.join(args.out, "metrics", "summary_by_scene.csv"), index=False)
    summarize_by_degradation(all_rows).to_csv(
        os.path.join(args.out, "metrics", "summary_by_degradation.csv"), index=False)
    with open(os.path.join(args.out, "metrics", "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("wrote", os.path.join(args.out, "metrics"))


if __name__ == "__main__":
    main()
