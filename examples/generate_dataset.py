"""Generate a labelled recon dataset (one command -> validated dataset).

Usage:
    python examples/generate_dataset.py --episodes 60 --split train --out output/ds_train
    python examples/generate_dataset.py --episodes 20 --split test  --out output/ds_test --grid
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skygym.scenarios import build_dataset_with_qa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", default="output/dataset")
    args = ap.parse_args()

    manifest = build_dataset_with_qa(
        out_dir=args.out, episodes=args.episodes, split=args.split,
        seed_offset=args.seed_offset, duration_s=args.duration)
    print(f"dataset written -> {args.out}")
    print(f"episodes: {len(manifest['episodes'])}  split: {manifest['split']}")
    print(f"QA violations: {manifest['qa']['violations']}")
    d = manifest["distribution"]
    print(f"detections: radar={d['detections']['radar']} "
          f"eo={d['detections']['eo']} rf={d['detections']['rf']}")
    for s in ("radar", "eo", "rf"):
        rs = d["range_stats"][s]
        if rs.get("n"):
            print(f"  {s:6s} range km: mean={rs['mean']/1000:.2f} "
                  f"[{rs['min']/1000:.2f}, {rs['max']/1000:.2f}]")
    print("manifest.json + distribution_report.json + coverage plots included")


if __name__ == "__main__":
    main()
