#!/usr/bin/env python3
"""Run the Stone Soup tracking baseline on one SkyGym episode (CLI).

Thin wrapper around skygym.stone_soup (the standard bridge since v0.2.0):

    SkyGymEnv.step()  ->  corrupted detections (radar / EO / RF)
                      ->  Stone Soup Detections (per-det measurement models)
                      ->  EKF + GNN2D MultiTargetTracker
                      ->  graded against the hidden witness GT

Two modes:
    --sensors radar   radar az/el/range only
    --sensors fusion  radar + EO (0.08 deg bearing, stereo range when
                      finite) + RF (az-only, 4 deg)

Examples:
    python examples/run_stonesoup.py --sensors radar  --start-km 1.2
    python examples/run_stonesoup.py --sensors fusion --start-km 1.2
    python examples/run_stonesoup.py --sensors fusion --start-km 4.0 --noise 2.0 --clutter 2.5

Requires: pip install stonesoup   (or pip install -e '.[tracking]')
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from skygym.config import EnvCfg, ScenarioCfg
from skygym.env import SkyGymEnv
from skygym.stone_soup import run_episode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensors", choices=["radar", "fusion"], default="fusion")
    p.add_argument("--scenario", default="approach")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--start-km", type=float, default=1.2)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--clutter", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--no-eo-range", action="store_true",
                   help="feed EO as bearing-only even when stereo range exists")
    p.add_argument("--out", default="results/stonesoup")
    args = p.parse_args()

    r = args.start_km * 1000.0
    sc = ScenarioCfg(
        name=args.scenario, duration_s=args.duration, seed=args.seed,
        start_min=(-1.08 * r, 0.75 * r, 60.0),
        start_max=(-0.75 * r, 1.08 * r, 140.0),
        tx_on=True, noise_scale=args.noise, clutter_scale=args.clutter)
    env = SkyGymEnv(EnvCfg())
    summary, rows, n_frames = run_episode(
        env, seed=args.seed, options={"scenario_cfg": sc},
        mode=args.sensors, eo_with_range=not args.no_eo_range)

    tag = f"{args.scenario}_d{args.duration:.0f}s_r{args.start_km:.1f}_" \
          f"n{args.noise:g}_c{args.clutter:g}_{args.sensors}"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, f"{tag}_tracks.csv"), "w") as f:
        f.write("t_s,track_id,est_e_m,est_n_m,est_u_m,est_vel_e,est_vel_n,"
                "est_vel_u,pos_err_m,az_err_deg,el_err_deg,range_err_m\n")
        for row in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                             for v in row) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/{tag}_summary.json and {tag}_tracks.csv "
          f"({len(rows)} rows / {n_frames} frames)")


if __name__ == "__main__":
    main()
