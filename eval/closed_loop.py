#!/usr/bin/env python3
"""Closed-loop safety audit: perception, prediction, planning, control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualize_nuscenes_mini_scene import SceneData, quat_to_rot  # noqa: E402
from eval.core import degrade, obb_overlap, rectangle_corners  # noqa: E402
from eval.experiments import DEFAULT_DEGRADATIONS, DEFAULT_SCENES, parse_name  # noqa: E402
from eval.predict import IMMPredictor  # noqa: E402
from eval.planning import FrenetPlanner, RiskAStarPlanner  # noqa: E402
from eval.control import MPPIController, PurePursuitController  # noqa: E402
from eval.safety import SafetyMonitor  # noqa: E402
from eval.sim import (  # noqa: E402
    DynamicBicycle, EgoState, ReferencePath, VehicleParams,
)
from eval.world import SceneReplay, WorldModel  # noqa: E402

PLANNERS = {"frenet": FrenetPlanner, "astar": RiskAStarPlanner}
CONTROLLERS = {"mppi": MPPIController, "pp": PurePursuitController}
COMBOS = ["frenet+mppi", "frenet+pp", "astar+mppi", "astar+pp"]


def _segment_distance(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-9 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_distance(a, b):
    if obb_overlap(a, b):
        return 0.0
    best = np.inf
    for p in a:
        for i in range(len(b)):
            best = min(best, _segment_distance(p, b[i], b[(i + 1) % len(b)]))
    for p in b:
        for i in range(len(a)):
            best = min(best, _segment_distance(p, a[i], a[(i + 1) % len(a)]))
    return float(best)


def ego_corners(ego, p: VehicleParams):
    c, s = np.cos(ego.yaw), np.sin(ego.yaw)
    local = np.array([[p.length / 2, p.width / 2], [p.length / 2, -p.width / 2],
                      [-p.length / 2, -p.width / 2], [-p.length / 2, p.width / 2]])
    return local @ np.array([[c, -s], [s, c]]).T + ego.xy


def _box_points(center, size, base_z=0.0, spacing=0.25):
    hx, hy = size[0] / 2.0, size[1] / 2.0
    xs = np.arange(-hx, hx + 1e-6, spacing)
    ys = np.arange(-hy, hy + 1e-6, spacing)
    pts = []
    for dz in (0.4, 0.9, 1.4):
        for x in xs:
            for y in ys:
                pts.append([center[0] + x, center[1] + y, base_z + dz])
    return np.asarray(pts, dtype=float)


def _forecasts(agents, t, imm, max_dist=40.0, ego_xy=None):
    out = []
    for a in agents:
        if a.get("track") is None:
            continue
        if ego_xy is not None and np.linalg.norm(a["xy"] - ego_xy) > max_dist:
            continue
        th, xy, yh, _ = a["track"].history(t)
        if len(th) < 2:
            continue
        out.append({"track": a["track"], "xy": a["xy"], "v": a["v"],
                    "forecast": imm.forecast(th, xy, yh, horizon=3.0, dt_pred=0.2)})
    return out


def _shift_plan(plan, dt):
    plan.t = np.maximum.accumulate(plan.t - dt)
    return plan


def _plan_arrays(plan, max_pts=90):
    n = min(len(plan.xy), max_pts)
    x = np.full(max_pts, np.nan)
    y = np.full(max_pts, np.nan)
    x[:n] = plan.xy[:n, 0]
    y[:n] = plan.xy[:n, 1]
    return x, y


@dataclass
class LoopConfig:
    max_seconds: float = 25.0
    plan_every: int = 2
    dt: float = 0.1
    goal_margin: float = 3.0
    seed: int = 0
    capture: bool = False
    safety: str = "aware"
    inject_distance: float | None = None
    inject_size: tuple = (1.2, 1.2, 1.5)
    start_frame: int = 0


def run_scenario(scene_name, degradation, planner_name, controller_name,
                 root, version, cfg: LoopConfig | None = None):
    cfg = cfg or LoopConfig()
    kind, param = parse_name(degradation)
    sc = SceneData(root, version, scene_name)
    replay = SceneReplay(sc)
    world = WorldModel(sc, replay)
    start = int(np.clip(cfg.start_frame, 0, len(replay.tokens) - 2))
    t0 = float(replay.times[start])
    ref = ReferencePath(replay.real_xy[start:])
    p = VehicleParams()

    ego = EgoState(float(replay.real_xy[start, 0]), float(replay.real_xy[start, 1]),
                   float(replay.real_yaw[start]),
                   float(np.clip(replay.real_v[start], 0.0, 12.0)))
    plant = DynamicBicycle(ego.x, ego.y, ego.yaw, ego.v, p)
    planner = PLANNERS[planner_name](
        params=p, unknown_weight=1.0 if cfg.safety == "aware" else 0.0)
    controller = CONTROLLERS[controller_name](params=p)
    monitor = SafetyMonitor(use_unknown=(cfg.safety == "aware"))
    imm = IMMPredictor()
    rng = np.random.default_rng(cfg.seed)

    t_end = min(cfg.max_seconds, float(replay.times[-1]) - t0 - 0.5)
    steps = int(round(t_end / cfg.dt))
    plan = None
    records = []

    log = {k: [] for k in ["t", "x", "y", "yaw", "v", "accel", "steer", "mode",
                           "gap", "ttc", "d", "free_d", "safe_dist", "unk_ratio",
                           "plan_x", "plan_y", "lat_acc", "pred_rmse"]}
    collision = False
    collision_speed = None
    reached = False
    mrm_count = 0
    prev_mode = "CRUISE"
    s_start = float(ref.project(ego.xy)[0])
    inject_xy = (ref.point(s_start + cfg.inject_distance)
                 if cfg.inject_distance is not None else None)
    distance = 0.0
    prev_xy = ego.xy.copy()
    pred_errors = []

    for k in range(steps):
        t_abs = t0 + (k + 1) * cfg.dt
        idx = replay.sweep_near(plant.state.xy)
        pts = replay.sweep_points(idx)
        if inject_xy is not None:
            sw = replay.sweeps[idx]
            rot = quat_to_rot(sw["pose"]["rotation"])
            trans = np.asarray(sw["pose"]["translation"], dtype=float)
            ped = _box_points(inject_xy, cfg.inject_size, base_z=float(trans[2]))
            pts = np.vstack([pts, (ped - trans) @ rot])
        pts = degrade(pts, kind, param, rng)
        world.update(plant.state.xy, pts)
        agents = replay.agents_at(t_abs)
        if inject_xy is not None:
            agents = list(agents) + [{
                "track": None, "xy": inject_xy, "v": 0.0,
                "corners": rectangle_corners(inject_xy, 0.0,
                                             cfg.inject_size[0], cfg.inject_size[1]),
            }]
        forecasts = _forecasts(agents, t_abs, imm, ego_xy=plant.state.xy)

        if plan is None or k % cfg.plan_every == 0:
            plan = planner.plan(plant.state, ref, world, forecasts)
        else:
            plan = _shift_plan(plan, cfg.dt)

        accel, steer = controller.step(plant.state, plan, world)
        safety = monitor.update(plant.state, world, forecasts, t_abs, ref=ref)
        if safety.v_cap is not None:
            accel = min(accel, float(np.clip(0.9 * (safety.v_cap - plant.state.v),
                                             -p.max_decel, p.max_accel)))
        plant.step(accel, steer)

        corners = ego_corners(plant.state, p)
        gap = np.inf
        for a in agents:
            d = polygon_distance(corners, a["corners"])
            gap = min(gap, d)
            if d <= 0.0:
                collision = True
                if collision_speed is None:
                    collision_speed = float(plant.state.v)
        if collision:
            gap = 0.0

        s_now, d_now = ref.project(plant.state.xy)
        distance += float(np.linalg.norm(plant.state.xy - prev_xy))
        prev_xy = plant.state.xy.copy()
        lat_acc = abs(plant.state.v * (plant.state.yaw - ego.yaw) / cfg.dt) \
            if k > 0 else 0.0

        pred_rmse = np.nan
        if forecasts:
            errs = []
            for f in forecasts:
                tr = f["track"]
                if t_abs + f["forecast"].t[-1] > tr.t[-1]:
                    continue
                pred = f["forecast"].mean()
                actual = np.array([tr.at(t_abs + tau)[0] for tau in f["forecast"].t])
                errs.append(np.linalg.norm(actual - pred, axis=1).mean())
            if errs:
                pred_rmse = float(np.mean(errs))
                pred_errors.append(pred_rmse)

        mode = safety.mode
        if mode == "MRM" and prev_mode != "MRM":
            mrm_count += 1
        prev_mode = mode

        px, py = _plan_arrays(plan)
        values = [t_abs, plant.state.x, plant.state.y, plant.state.yaw,
                  plant.state.v, accel, steer, mode, gap, safety.ttc,
                  d_now, safety.free_dist, safety.safe_dist, safety.unknown_ratio,
                  px, py, lat_acc, pred_rmse]
        for key, val in zip(log, values):
            log[key].append(val)

        if cfg.capture:
            records.append({
                "t": t_abs, "ego": plant.state.copy(), "plan": plan,
                "forecasts": forecasts, "safety": safety, "agents": agents,
                "layers": {k2: v.copy() for k2, v in world.layers.items()},
                "grid": world.grid, "map_free": world.map_free,
            })

        ego = plant.state.copy()
        if collision:
            break
        if s_now >= ref.length - cfg.goal_margin:
            reached = True
            break

    n = len(log["t"])
    modes = np.array(log["mode"], dtype=object)
    speeds = np.array(log["v"])
    metrics = {
        "scene": scene_name, "degradation": degradation,
        "planner": planner_name, "controller": controller_name,
        "safety": cfg.safety,
        "inject_distance": cfg.inject_distance,
        "start_frame": cfg.start_frame,
        "success": bool(reached and not collision),
        "collision": bool(collision),
        "collision_moving": bool(collision and (collision_speed or 0.0) >= 0.8),
        "collision_at_stop": bool(collision and (collision_speed or 0.0) < 0.8),
        "collision_speed": float(collision_speed) if collision_speed is not None else np.nan,
        "time": float(log["t"][-1]) if n else 0.0,
        "distance": distance,
        "mean_speed": float(speeds.mean()) if n else 0.0,
        "min_gap": float(np.min(log["gap"])) if n else np.nan,
        "min_ttc": float(np.min(log["ttc"])) if n else np.nan,
        "mrm_count": mrm_count,
        "mrm_time": float(np.sum(modes == "MRM") * cfg.dt),
        "caution_time": float(np.sum(modes == "CAUTION") * cfg.dt),
        "max_lat_acc": float(np.max(log["lat_acc"])) if n else 0.0,
        "cross_track_rms": float(np.sqrt(np.mean(np.square(log["d"])))) if n else np.nan,
        "pred_rmse": float(np.mean(pred_errors)) if pred_errors else np.nan,
        "unknown_ratio": float(np.mean(log["unk_ratio"])) if n else np.nan,
        "n_steps": n,
        "goal_progress": float(ref.project(plant.state.xy)[0] - s_start),
        "route_length": float(ref.length - s_start),
        "progress_ratio": float(np.clip((ref.project(plant.state.xy)[0] - s_start)
                                        / max(ref.length - s_start, 1e-3), 0.0, 1.0)),
    }
    arrays = {k: (np.array(v, dtype="U16") if k == "mode" else np.asarray(v))
              for k, v in log.items()}
    arrays["safety_mode"] = arrays.pop("mode")
    arrays["route_xy"] = ref.xy
    return metrics, arrays, records


def _parse_combos(names):
    out = []
    for n in names:
        if n not in COMBOS:
            raise SystemExit("Unknown combo %r. Choose from %s" % (n, ", ".join(COMBOS)))
        planner, controller = n.split("+")
        out.append((planner, controller))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--degradations", nargs="+",
                    default=["clean", "range20", "range10", "drop75", "noise01"])
    ap.add_argument("--combos", nargs="+", default=COMBOS)
    ap.add_argument("--safety", nargs="+", default=["aware"],
                    choices=["aware", "blind"])
    ap.add_argument("--root", default="data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="outputs/eval_results/closed_loop")
    ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--plan-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject-distance", type=float, default=None,
                    help="static box distance ahead on the route (metres)")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--demo", default=None,
                    help="scene:degradation:planner+controller to render")
    args = ap.parse_args()

    cfg = LoopConfig(max_seconds=args.max_seconds, plan_every=args.plan_every,
                     seed=args.seed, inject_distance=args.inject_distance,
                     start_frame=args.start_frame)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "steps"), exist_ok=True)

    if args.demo:
        scene, deg, combo = args.demo.split(":")
        planner, controller = combo.split("+")
        cfg.capture = True
        cfg.safety = args.safety[0]
        metrics, arrays, records = run_scenario(scene, deg, planner, controller,
                                                args.root, args.version, cfg)
        from eval.visualize_loop import render_scenario
        tag = f"{scene}_{deg}_{planner}_{controller}"
        if cfg.safety != "aware":
            tag += f"_{cfg.safety}"
        render_scenario(records, metrics, arrays,
                        os.path.join(args.out, "demo", tag))
        print(json.dumps(metrics, indent=2))
        return

    rows = []
    for safety in args.safety:
        run_cfg = LoopConfig(max_seconds=args.max_seconds,
                             plan_every=args.plan_every, seed=args.seed,
                             safety=safety, inject_distance=args.inject_distance,
                             start_frame=args.start_frame)
        suffix = "" if safety == "aware" else f"_{safety}"
        for scene in args.scenes:
            for deg in args.degradations:
                for planner, controller in _parse_combos(args.combos):
                    metrics, arrays, _ = run_scenario(scene, deg, planner, controller,
                                                      args.root, args.version, run_cfg)
                    rows.append(metrics)
                    np.savez(os.path.join(args.out, "steps",
                                          f"{scene}_{deg}_{planner}_{controller}{suffix}.npz"),
                             **arrays)
                    print("%-12s %-8s %-12s %-5s success=%d coll=%d mrm=%d "
                          "gap=%.2f ttc=%.2f"
                          % (scene, deg, f"{planner}+{controller}", safety,
                             metrics["success"], metrics["collision"],
                             metrics["mrm_count"], metrics["min_gap"],
                             metrics["min_ttc"]), flush=True)

    path = os.path.join(args.out, "metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
