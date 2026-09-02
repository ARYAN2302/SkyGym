#!/usr/bin/env python3
"""Batch-evaluate the Stone Soup tracking baseline over many episodes.

Usage:
    python examples/evaluate.py --episodes 12 --scenario approach
    python examples/evaluate.py --episodes 8 --mode radar
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skygym.config import EnvCfg
from skygym.env import SkyGymEnv
from skygym.metrics import run_tracker_on_episode, aggregate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--scenario", default=None,
                    help="hover|approach|orbit|waypoint_cruise|serpentine|egress")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--mode", default="fusion", choices=["fusion", "radar"])
    ap.add_argument("--no-eo-range", action="store_true",
                    help="feed EO as bearing-only even when stereo range exists")
    ap.add_argument("--seed-offset", type=int, default=9_000_000,
                    help="default seeds from the TEST split range")
    args = ap.parse_args()

    env = SkyGymEnv(EnvCfg())
    results = []
    for i in range(args.episodes):
        seed = args.seed_offset + i
        res = run_tracker_on_episode(
            env, seed=seed,
            options={"scenario": args.scenario, "duration_s": args.duration},
            mode=args.mode, eo_with_range=not args.no_eo_range)
        res["seed"] = seed
        results.append(res)
        sc_name = args.scenario if args.scenario else "random"
        rmse = res.get("position_rmse_m")
        cont = res.get("track_continuity")
        ida = res.get("id_accuracy")
        print(f"ep{i:03d} seed={seed} scenario={sc_name:>15} mode={args.mode:>6} "
              f"init={'Y' if res['track_initiated'] else 'N'} "
              f"rmse={rmse if rmse is not None else float('nan'):7.1f}m "
              f"cont={cont if cont is not None else float('nan'):.2f} "
              f"id={ida if ida is not None else float('nan'):.2f}")
    agg = aggregate(results)
    print("\n== aggregate ==")
    print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
