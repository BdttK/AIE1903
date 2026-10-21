#!/usr/bin/env python3
"""One-command runner for the midterm audit.

Example:
  python -m audit.run --frame-step 1 --out outputs/audit_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audit.core import Grid  # noqa: E402
from audit.experiments import (  # noqa: E402
    DEFAULT_DEGRADATIONS, DEFAULT_SCENES, parse_name, run_scene,
    summarize, summarize_by_degradation,
)
import audit.map_audit as audit_map  # noqa: E402
import audit.inject as audit_inject  # noqa: E402
import audit.visualize as vis  # noqa: E402

# 1. 退化扫描：4 场景 × 160 关键帧（--frame-step 1）× 11 档退化 × 3 seeds → per_frame.csv、hazards.csv、两个 summary（run.py:60-75）
# 2. 主图：退化曲线、昼夜对比、召回-false-free 权衡、距离/类别剖面、失效画廊（run.py:78-89）
# 3. 地图审计：4 场景的 A/B/C 差异、归因、全局持续性栅格与图（run.py:92-104）
# 4. 危险注入：3 个帧距 × 4 档退化 × 4 个注入距离（run.py:107-132）

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--degradations", nargs="+", default=DEFAULT_DEGRADATIONS)
    ap.add_argument("--seeds", type=int, default=3)       # 3
    ap.add_argument("--frame-step", type=int, default=1)  # 1
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/audit_results")  # outputs/audit_results
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--skip-inject", action="store_true")
    ap.add_argument("--skip-gallery", action="store_true")
    ap.add_argument("--gallery-degradation", default="range10")
    ap.add_argument("--inject-frames", nargs="+", type=int, default=[0, 10, 20])
    ap.add_argument("--inject-distances", nargs="+", type=float,
                    default=[10, 15, 20, 25])
    ap.add_argument("--inject-degradations", nargs="+",
                    default=["clean", "range20", "range10", "drop50"])
    ap.add_argument("--inject-category", default="pedestrian",
                    choices=list(audit_inject.ACTOR_SIZES))
    args = ap.parse_args()

    grid = Grid(res=args.res)  # BEV 栅格坐标系定义
    met = os.path.join(args.out, "metrics")
    fig = os.path.join(args.out, "figures")
    os.makedirs(met, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    with open(os.path.join(met, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1. degradation sweep
    rows, hazard_rows = [], []
    for scene in args.scenes:
        r, h = run_scene(scene, args.degradations, args.seeds,
                         args.frame_step, grid, args.root, args.version)
        rows.extend(r)
        hazard_rows.extend(h)
    per_frame = pd.DataFrame(rows)
    per_frame.to_csv(os.path.join(met, "per_frame.csv"), index=False)
    if hazard_rows:
        pd.DataFrame(hazard_rows).to_csv(os.path.join(met, "hazards.csv"),
                                         index=False)
    summary_scene = summarize(rows)
    summary_deg = summarize_by_degradation(rows)
    summary_scene.to_csv(os.path.join(met, "summary_by_scene.csv"), index=False)
    summary_deg.to_csv(os.path.join(met, "summary_by_degradation.csv"), index=False)
    print("metrics:", len(per_frame), "rows")

    # 2. main figures
    vis.plot_curves(summary_scene, summary_deg, fig)
    vis.plot_tradeoff(summary_scene, fig)
    hazards_csv = os.path.join(met, "hazards.csv")
    if os.path.exists(hazards_csv):
        vis.plot_distance_profile(hazards_csv, fig)
        vis.plot_category_profile(hazards_csv, fig)
    if not args.skip_gallery:
        gal = args.gallery_degradation
        if gal not in args.degradations:
            gal = args.degradations[-1]
        vis.failure_gallery(os.path.join(met, "per_frame.csv"),
                            args.root, args.version, grid, fig, gal)

    # 3. map audit
    if not args.skip_map:
        map_rows = []
        for scene in args.scenes:
            map_rows.extend(audit_map.run_scene(
                scene, args.root, args.version, args.frame_step, grid, met))
            demo = min(10, 39)
            audit_map.plot_frame(scene, args.root, args.version, demo, grid,
                                 os.path.join(fig, "map_discrepancy_%s.png" % scene))
            audit_map.plot_persistence(
                os.path.join(met, "map_global_%s.npz" % scene),
                os.path.join(fig, "map_persistence_%s.png" % scene))
        pd.DataFrame(map_rows).to_csv(
            os.path.join(met, "map_metrics.csv"), index=False)

    # 4. hazard injection
    if not args.skip_inject:
        inj_rows = []
        for scene in args.scenes:
            from utils.visualize_nuscenes_mini_scene import SceneData
            sc = SceneData(args.root, args.version, scene)
            for frame in args.inject_frames:
                if frame >= len(sc.sample_tokens):
                    continue
                for name in args.inject_degradations:
                    kind, param = parse_name(name)
                    nseed = 1 if kind in ("clean", "range") else args.seeds
                    for distance in args.inject_distances:
                        for seed in range(nseed):
                            out, _, _, _ = audit_inject.run_case(
                                sc, frame, name, distance, 0.0,
                                args.inject_category, kind, param, seed, grid)
                            out.pop("_path", None)
                            inj_rows.append(out)
            audit_inject.plot_case(
                sc, args.inject_frames[0], args.inject_degradations[-1],
                max(args.inject_distances) - 10, args.inject_category,
                *parse_name(args.inject_degradations[-1]), 0, grid,
                os.path.join(fig, "injection_%s.png" % scene))
        inj = pd.DataFrame(inj_rows)
        inj.to_csv(os.path.join(met, "injection_metrics.csv"), index=False)
        audit_inject.plot_summary(inj, os.path.join(fig, "injection_summary.png"))

    print("done:", args.out)


if __name__ == "__main__":
    main()
