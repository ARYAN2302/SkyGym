#!/usr/bin/env python3
"""Build the Stage-1 corpus: episodes -> per-frame model records.

For every seed: roll the env, track with the Stone Soup fusion baseline,
and store per-frame (0.1 s grid) records containing ONLY legitimate inputs
(tracker state + observation summaries - never truth-derived features) plus
truth targets for supervision.

Track-selection rule (truth-free, identical at train and test time):
among live tracks, the most mature (most cumulative updates) wins.
Frames where that track is grossly off (>500 m from truth, ~clutter
hijack / warm-up) are flagged `off` and excluded from TRAINING losses;
test metrics are reported unfiltered and with the flag applied.

Usage:
    python3 build_stage1_corpus.py --smoke          # 2 episodes, inspect
    python3 build_stage1_corpus.py                  # full 320/80/80 run
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Project root: override with SKYGYM_PROJECT_ROOT (see STAGE1_RUNBOOK.md §1)
PROJECT = os.environ.get("SKYGYM_PROJECT_ROOT", "/home/z/my-project")
sys.path.insert(0, os.path.join(PROJECT, "skygym_repo"))

import numpy as np

from skygym.config import EnvCfg, ScenarioCfg
from skygym.env import SkyGymEnv
from skygym.stone_soup import (T0, build_detections, build_tracker, roll_frames)
from skygym.world import ang_diff_deg, cartesian_to_spherical, spherical_to_cartesian

OUT_DIR = Path(PROJECT) / "download" / "stage1" / "corpus"
CLASSES = ["quad", "fixed_wing", "bird"]
MIX = ["approach", "orbit", "serpentine", "hover", "waypoint_cruise", "egress"]
DT = 0.1
N_FRAMES = 200
OFF_M = 500.0            # 'grossly off' flag threshold

SPLITS = {"train": (1_000_000, 320), "val": (8_000_000, 80), "test": (9_000_000, 80)}

FEATURE_NAMES = (
    [f"est_{a}" for a in ("e", "n", "u")] +
    [f"vel_{a}" for a in ("e", "n", "u")] +
    ["track_len", "stale"] +
    [f"radar_{k}" for k in ("n", "snr_mean", "snr_max", "dist_min",
                            "p_quad", "p_fixed", "p_bird", "p_unknown")] +
    [f"eo_{k}" for k in ("n", "px_mean", "px_max", "dist_min",
                         "p_quad", "p_fixed", "p_bird", "p_unknown")] +
    [f"rf_{k}" for k in ("n", "az_diff",
                         "p_quad", "p_fixed", "p_bird", "p_unknown")] +
    ["track_valid"])


def episode_cfg(seed: int) -> ScenarioCfg:
    """Fully seed-determined episode configuration (class-balanced mix)."""
    rng = np.random.default_rng(seed)
    cls = CLASSES[seed % 3]                      # round-robin balance
    behav = MIX[rng.integers(0, len(MIX))]
    return ScenarioCfg(
        name=behav, duration_s=20.0, seed=seed, true_class=cls,
        tx_on=bool(rng.random() < 0.8),
        noise_scale=float(rng.uniform(0.5, 2.0)),
        clutter_scale=float(rng.uniform(0.5, 2.5)))


def _sensor_block(rows, est_pos, site, kind):
    """Legit-input summary of one sensor's det rows vs the track estimate."""
    n = len(rows)
    out = [0.0] * 8 if kind != "rf" else [0.0] * 6
    if n == 0:
        return out
    posts = np.zeros(4)
    dists, extra = [], []
    for r in rows:
        posts += np.array(r[7:11])
        az, el, rng_m = float(r[0]), float(r[1]), float(r[2])
        if kind == "radar":
            extra.append(float(r[4]) / 30.0)                       # snr
        elif kind == "eo":
            px = float(r[5])
            if np.isfinite(px):
                extra.append(min(px, 10.0) / 5.0)
        else:  # rf: angular agreement with the track bearing
            if est_pos is not None:
                az_tr = math.degrees(math.atan2(est_pos[0], est_pos[1]))
                extra.append(abs(ang_diff_deg(az, az_tr)) / 45.0)
        if est_pos is not None and np.isfinite(el) and np.isfinite(rng_m):
            p = spherical_to_cartesian(az, el, rng_m) + site
            dists.append(float(np.linalg.norm(p - est_pos)) / 1000.0)
    out[0] = min(n, 24) / 24.0
    base = 1 if kind == "rf" else 3   # rf: [n, az_diff, p×4] | radar/eo: [n, extra, extra_max, dist, p×4]
    if extra:
        out[1] = float(np.mean(extra))
        if kind != "rf":
            out[2] = float(np.max(extra))
    if dists:
        out[base] = float(np.min(dists))
    for j in range(4):
        out[base + 1 + j] = posts[j] / n
    return out


def build_episode(seed: int) -> dict:
    sc = episode_cfg(seed)
    env = SkyGymEnv(EnvCfg())
    frames, env = roll_frames(env, seed=seed, options={"scenario_cfg": sc})
    detector, per_sensor, _ = build_detections(frames, env, mode="fusion")
    tracker = build_tracker()
    tracker.detector = detector

    K = len(frames)
    site = np.zeros(3)
    est_pos = np.full((K, 3), np.nan)
    est_vel = np.full((K, 3), np.nan)
    track_len = np.zeros(K)
    stale = np.full(K, np.nan)
    last_tick_t = None

    for time, tracks in tracker:                       # truth-free selection
        t_rel = (time - T0).total_seconds()
        k = int(np.clip(round(t_rel / DT), 0, K - 1))
        last_tick_t = t_rel
        if not tracks:
            continue
        best = max(tracks, key=lambda tr: (len(tr.states), str(tr.id)))
        sv = np.asarray(best.state_vector).flatten()
        est_pos[k] = sv[[0, 2, 4]]
        est_vel[k] = sv[[1, 3, 5]]
        track_len[k] = len(best.states)

    feats = np.zeros((K, len(FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros(K, dtype=np.float32)
    off = np.zeros(K, dtype=np.float32)
    truth_pos = np.array([f["gt"]["pos"] for f in frames], dtype=np.float32)
    truth_vel = np.array([f["gt"]["vel"] for f in frames], dtype=np.float32)

    for k, f in enumerate(frames):
        t = f["gt"]["t"]
        has_tr = np.isfinite(est_pos[k]).all()
        valid[k] = 1.0 if has_tr else 0.0
        ep = est_pos[k] if has_tr else None
        if has_tr:
            if last_tick_t is not None:
                stale[k] = min(max(t - last_tick_t, 0.0), 2.0)
            if np.linalg.norm(est_pos[k] - truth_pos[k]) > OFF_M:
                off[k] = 1.0
        row = feats[k]
        row[0:3] = est_pos[k] / 1000.0 if has_tr else 0.0
        row[3:6] = est_vel[k] / 30.0 if has_tr else 0.0
        row[6] = track_len[k] / N_FRAMES
        row[7] = stale[k] if np.isfinite(stale[k]) else 2.0
        obs = f["obs"]
        row[8:16] = _sensor_block(obs["radar"]["dets"][:obs["radar"]["n"]], ep, site, "radar")
        row[16:24] = _sensor_block(obs["eo"]["dets"][:obs["eo"]["n"]], ep, site, "eo")
        row[24:30] = _sensor_block(obs["rf"]["dets"][:obs["rf"]["n"]], ep, site, "rf")
        row[30] = valid[k]

    return {
        "seed": seed, "scenario": sc.name, "true_class": sc.true_class,
        "class_idx": CLASSES.index(sc.true_class), "tx_on": sc.tx_on,
        "noise_scale": sc.noise_scale, "clutter_scale": sc.clutter_scale,
        "K": K,
        "est_pos": est_pos.tolist(), "est_vel": est_vel.tolist(),
        "track_len": track_len.tolist(),
        "feats": np.round(feats, 5).tolist(),
        "track_valid": valid.tolist(), "off": off.tolist(),
        "truth_pos": np.round(truth_pos, 3).tolist(),
        "truth_vel": np.round(truth_vel, 3).tolist(),
    }


def _work(args):
    split, seed = args
    t0 = time.perf_counter()
    rec = build_episode(seed)
    return split, seed, rec, time.perf_counter() - t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", action="append", default=[],
                    help="split:start:count (repeatable); default = full run")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    jobs, plan = [], {}
    if args.chunk:
        for spec in args.chunk:
            split, start, count = spec.split(":")
            base = SPLITS[split][0]
            plan.setdefault(split, []).extend(base + i
                                              for i in range(int(start),
                                                             int(start) + int(count)))
    else:
        for split, (base, n) in SPLITS.items():
            n = min(n, 1) if args.smoke else n
            plan[split] = [base + i for i in range(n)]
    for split, seeds in plan.items():
        jobs += [(split, s) for s in seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    handles = {s: open(OUT_DIR / f"{s}.jsonl", mode) for s in plan}
    counts = {s: 0 for s in SPLITS}
    t0 = time.perf_counter()
    with Pool(2) as pool:
        for i, (split, seed, rec, dt) in enumerate(
                pool.imap_unordered(_work, jobs, chunksize=1)):
            handles[split].write(json.dumps(rec) + "\n")
            handles[split].flush()
            counts[split] += 1
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                el = time.perf_counter() - t0
                print(f"[{i+1}/{len(jobs)}] {el:6.0f}s elapsed  "
                      f"({el/(i+1):.1f}s/ep)  { {k: v for k, v in counts.items() if v} }",
                      flush=True)
    for h in handles.values():
        h.close()
    meta = {"feature_names": FEATURE_NAMES, "dt": DT, "n_frames": N_FRAMES,
            "off_m": OFF_M, "classes": CLASSES,
            "splits_seed_base": {s: b for s, (b, _) in SPLITS.items()},
            "last_chunk": {s: counts[s] for s in counts if counts[s]},
            "hours": (time.perf_counter() - t0) / 3600}
    (OUT_DIR / "corpus_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", json.dumps(counts), f"{time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
