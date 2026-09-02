"""End-to-end demo: fly one scenario, record detections, run the EKF, report.

Usage:
    python examples/demo.py [--scenario serpentine] [--seed 42] [--out output]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from skygym.config import EnvCfg
from skygym.env import SkyGymEnv
from skygym.metrics import run_tracker_on_episode
from skygym.wrappers import DetectionRecorder, QAChecker, DistributionMonitor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="serpentine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = SkyGymEnv(EnvCfg())

    # 1) tracker eval pass
    res = run_tracker_on_episode(env, seed=args.seed,
                                 options={"scenario": args.scenario,
                                          "duration_s": args.duration})
    print("== tracker eval vs witness ==")
    print(json.dumps(res, indent=2, default=str))

    # 2) recorded + QA'd rollout with coverage stats
    qa_env = QAChecker(SkyGymEnv(EnvCfg()))
    rec = DetectionRecorder(qa_env, os.path.join(args.out, "demo_detections.jsonl"))
    mon = DistributionMonitor(rec)
    obs, info = mon.reset(seed=args.seed,
                          options={"scenario": args.scenario,
                                   "duration_s": args.duration})
    scenario_name = info["scenario_cfg"]["name"]
    done, steps, dets_seen = False, 0, 0
    while not done:
        obs, _, te, tr, info = mon.step(None)
        dets_seen += sum(o["n"] for o in obs.values())
        done = te or tr
        steps += 1
    rec.close()
    mon.save_report(os.path.join(args.out, "demo_distribution.json"),
                    plots_dir=args.out)
    print(f"\n== rollout ==")
    print(f"scenario={scenario_name}  steps={steps}  "
          f"detections={dets_seen}  episode={info['gt']['episode_id']}")
    print(f"recorded -> {os.path.join(args.out, 'demo_detections.jsonl')}")
    print(f"QA violations: {qa_env.summary()['violations']}")
    rep = mon.report()
    for s in ("radar", "eo", "rf"):
        r = rep["range_stats"][s]
        if r.get("n"):
            print(f"  {s:6s}: {rep['detections'][s]:5d} dets | "
                  f"range {r['min']/1000:.1f}-{r['max']/1000:.1f} km "
                  f"(mean {r['mean']/1000:.1f})")


if __name__ == "__main__":
    main()
