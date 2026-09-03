#!/usr/bin/env python3.13
"""Benchmark: time env roll + Stone Soup run per episode (sizes the corpus)."""
import os
import sys
import time

# Project root: override with SKYGYM_PROJECT_ROOT (see STAGE1_RUNBOOK.md §1)
sys.path.insert(0, os.path.join(
    os.environ.get("SKYGYM_PROJECT_ROOT", "/home/z/my-project"), "skygym_repo"))
import numpy as np

from skygym.config import EnvCfg, ScenarioCfg
from skygym.env import SkyGymEnv
from skygym.stone_soup import run_episode

MIX = ["approach", "orbit", "serpentine", "hover", "waypoint_cruise", "egress"]
CLASSES = ["quad", "fixed_wing", "bird"]

t_all = time.perf_counter()
for i in range(4):
    seed = 777_000 + i
    t0 = time.perf_counter()
    sc = ScenarioCfg(
        name=MIX[i % len(MIX)], duration_s=20.0, seed=seed,
        true_class=CLASSES[i % len(CLASSES)], tx_on=bool(i % 4),
        noise_scale=float(np.random.default_rng(seed).uniform(0.5, 2.0)),
        clutter_scale=float(np.random.default_rng(seed).uniform(0.5, 2.5)))
    env = SkyGymEnv(EnvCfg())
    summary, rows, n_frames = run_episode(
        env, seed=seed, options={"scenario_cfg": sc}, mode="fusion")
    t1 = time.perf_counter()
    print(f"ep{i} {MIX[i % len(MIX)]:16s} {CLASSES[i % 3]:10s} "
          f"{t1 - t0:5.2f}s  tracked={summary['pct_of_episode_tracked']:5.1f}%  "
          f"rows={len(rows)}  dets={sum(summary['dets_fed'].values())}")
print(f"total {time.perf_counter() - t_all:.1f}s for 4 episodes")
