#!/usr/bin/env python3
"""Merge a per-scene rerun into the main matrix and annotate collisions."""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd


def _collision_speed(steps, row):
    path = os.path.join(steps, f"{row.scene}_{row.degradation}_"
                               f"{row.planner}_{row.controller}.npz")
    if not os.path.exists(path):
        return np.nan
    d = np.load(path, allow_pickle=True)
    gap = np.asarray(d["gap"], dtype=float)
    if gap.min() > 0.01:
        return np.nan
    return float(d["v"][int(np.argmin(gap))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", required=True, help="main run directory")
    ap.add_argument("--sub", required=True, help="rerun directory to merge")
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args()

    main = pd.read_csv(os.path.join(args.main, "metrics.csv"))
    sub = pd.read_csv(os.path.join(args.sub, "metrics.csv"))
    merged = pd.concat(
        [main[~main["scene"].isin(set(sub["scene"]))], sub], ignore_index=True)

    steps_main = os.path.join(args.main, "steps")
    steps_sub = os.path.join(args.sub, "steps")
    for f in glob.glob(os.path.join(steps_sub, "*.npz")):
        shutil.copyfile(f, os.path.join(steps_main, os.path.basename(f)))

    if "collision_speed" not in merged:
        merged["collision_speed"] = np.nan
    for i, row in merged.iterrows():
        if pd.isna(merged.at[i, "collision_speed"]):
            merged.at[i, "collision_speed"] = _collision_speed(steps_main, row)
    merged["collision_moving"] = (merged["collision_speed"] >= 0.8).astype(int)
    merged["collision_at_stop"] = (
        merged["collision_speed"].notna() & (merged["collision_speed"] < 0.8)
    ).astype(int)

    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(merged)} rows")


if __name__ == "__main__":
    main()
