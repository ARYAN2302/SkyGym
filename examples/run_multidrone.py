#!/usr/bin/env python3
"""S5 benchmark: track a fleet of drones with the Stone Soup baseline.

Runs multi-drone episodes (crossing / merging trajectories + clutter),
tracks all sensors with the standard baseline, grades each tick with a
global Hungarian track-to-truth assignment, and writes:

    <out>/multidrone_tracks.csv    per-tick track rows (target-assigned)
    <out>/multidrone_summary.json  per-target metrics + identity switches

Example:
    python examples/run_multidrone.py --n 3 --episodes 4 --duration 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skygym.config import EnvCfg  # noqa: E402
from skygym.multidrone import MultiDroneEnv  # noqa: E402
from skygym.stone_soup import run_episode_multi  # noqa: E402

CSV_COLS = ["t", "track_id", "tgt_idx", "e_m", "n_m", "u_m",
            "ve_mps", "vn_mps", "vu_mps",
            "pos_err_m", "az_err_deg", "el_err_deg", "range_err_m",
            "nearest_truth_m"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="drones per episode")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--mode", choices=("fusion", "radar"), default="fusion")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--clutter", type=float, default=1.0)
    ap.add_argument("--mix", default=None,
                    help="comma-separated behaviours, e.g. approach,approach,approach")
    ap.add_argument("--radius", default=None,
                    help="spawn band 'min,max' metres from the site")
    ap.add_argument("--out", default=os.path.join("results", "multidrone"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = MultiDroneEnv(EnvCfg(
        max_dets_per_sensor=max(24, 12 * args.n)))

    csv_path = os.path.join(args.out, "multidrone_tracks.csv")
    summaries = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + CSV_COLS)
        for ep in range(args.episodes):
            seed = args.seed + ep
            t0 = time.time()
            summary, rows, n_frames = run_episode_multi(
                env, seed,
                options={"n_drones": args.n, "duration_s": args.duration,
                         "noise_scale": args.noise,
                         "clutter_scale": args.clutter,
                         "mix": (args.mix.split(",") if args.mix else None),
                         "start_radius": (tuple(float(x) for x in
                                                args.radius.split(","))
                                          if args.radius else None)},
                mode=args.mode)
            for r in rows:
                w.writerow([ep] + [("" if v is None or
                                    (isinstance(v, float) and not np.isfinite(v))
                                    else round(v, 3) if isinstance(v, float)
                                    else v) for v in r])
            summaries.append(summary)
            pt = summary["per_target"]
            print(f"ep {ep} seed {seed}: mean tracked {summary['mean_tracked_pct']}% | "
                  f"ID switches {summary['identity_switches']} | "
                  f"false tracks {summary['n_false_confirmed_tracks']} | "
                  f"RMSE {[p.get('position_rmse_m', '-') for p in pt]} | "
                  f"({time.time() - t0:.1f}s, {n_frames} frames)")

    with open(os.path.join(args.out, "multidrone_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"config": vars(args), "episodes": summaries}, f, indent=2)

    # fleet-level rollup
    mean_tracked = float(np.mean([s["mean_tracked_pct"] for s in summaries]))
    switches = int(sum(s["identity_switches"] for s in summaries))
    false_tr = int(sum(s["n_false_confirmed_tracks"] for s in summaries))
    rmses = [p["position_rmse_m"] for s in summaries for p in s["per_target"]
             if "position_rmse_m" in p]
    print(f"\n== Fleet rollup ({args.episodes} episodes x {args.n} drones, "
          f"{args.mode}) ==")
    print(f"mean tracked: {mean_tracked:.1f}%  | identity switches: {switches}"
          f"  | false confirmed tracks: {false_tr}"
          f"  | median RMSE: {np.median(rmses) if rmses else '-'} m")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
