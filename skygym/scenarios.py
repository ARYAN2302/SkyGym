"""Scenario library + dataset builder.

Volume comes from two mechanisms (both enforced here):
  GRID coverage on the axes that matter  (scenario type x noise x clutter)
  RANDOM diversity on everything else    (start, speed, heading, jitter)
with explicit seed-range splits for train/val/test from day one.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import ScenarioCfg
from .env import SkyGymEnv
from .wrappers import DetectionRecorder, QAChecker, DistributionMonitor

SEED_RANGES = {
    "train": (0, 8_000_000),
    "val": (8_000_000, 9_000_000),
    "test": (9_000_000, 10_000_000),
}

SCENARIO_NAMES = ("hover", "approach", "orbit", "waypoint_cruise",
                  "serpentine", "egress")

NOISE_LEVELS = {"low": 0.5, "mid": 1.0, "high": 2.0}
CLUTTER_LEVELS = {"low": 0.3, "mid": 1.0, "high": 2.5}


def sample_scenario(rng: np.random.Generator, name: str | None = None,
                    noise_scale: float = 1.0, clutter_scale: float = 1.0,
                    true_class: str = "quad", duration_s: float | None = None,
                    tx_on: bool | None = None) -> ScenarioCfg:
    """Random scenario draw (the 'random diversity' mechanism)."""
    name = name or SCENARIO_NAMES[int(rng.integers(len(SCENARIO_NAMES)))]
    duration = duration_s if duration_s is not None else float(rng.uniform(45.0, 90.0))
    return ScenarioCfg(
        name=name,
        seed=int(rng.integers(0, 1 << 31 - 1)),
        duration_s=duration,
        true_class=true_class,
        tx_on=bool(rng.random() < 0.8) if tx_on is None else tx_on,
        speed_min=float(rng.uniform(6.0, 12.0)),
        speed_max=float(rng.uniform(20.0, 30.0)),
        orbit_radius_m=float(rng.uniform(400.0, 1200.0)),
        weave_amplitude_mps=float(rng.uniform(3.0, 9.0)),
        weave_period_s=float(rng.uniform(5.0, 12.0)),
        noise_scale=float(noise_scale),
        clutter_scale=float(clutter_scale),
    )


def grid_scenarios(n_per_cell: int = 2, duration_s: float = 60.0) -> list[ScenarioCfg]:
    """Deterministic coverage grid: scenario x noise x clutter."""
    out: list[ScenarioCfg] = []
    for sname in SCENARIO_NAMES:
        for nl_name, nl in NOISE_LEVELS.items():
            for cl_name, cl in CLUTTER_LEVELS.items():
                for k in range(n_per_cell):
                    out.append(ScenarioCfg(
                        name=sname, seed=1000 * k + hash((sname, nl_name, cl_name)) % 997,
                        duration_s=duration_s,
                        noise_scale=nl, clutter_scale=cl))
    return out


def build_dataset(out_dir: str, episodes: int = 40, split: str = "train",
                  seed_offset: int = 0, duration_s: float = 60.0,
                  use_grid: bool = False, env_cfg=None) -> dict:
    """Generate a labelled detection dataset: JSONL per episode + manifest.

    Seed hygiene: split ranges are disjoint by construction (SEED_RANGES),
    so train/val/test never share a scenario seed.
    """
    os.makedirs(out_dir, exist_ok=True)
    lo, _hi = SEED_RANGES[split]
    env = SkyGymEnv(env_cfg)
    env = QAChecker(env)
    monitor = None

    plans: list[tuple[int, dict | None]] = []
    if use_grid:
        for i, sc in enumerate(grid_scenarios(duration_s=duration_s)):
            plans.append((lo + seed_offset + i, {"scenario_cfg": sc}))
        episodes = len(plans)
    else:
        for i in range(episodes):
            plans.append((lo + seed_offset + i,
                          {"duration_s": duration_s,
                           "noise_scale": list(NOISE_LEVELS.values())[i % 3],
                           "clutter_scale": list(CLUTTER_LEVELS.values())[i % 3]}))

    manifest = {"split": split, "episodes": [], "n_planned": len(plans)}
    rec_paths = []
    for i, (seed, options) in enumerate(plans):
        ep_dir = os.path.join(out_dir, f"ep_{i:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        rec_path = os.path.join(ep_dir, "detections.jsonl")
        # rewrap: QAChecker(env) -> recorder on top
        inner = env.env if hasattr(env, "env") else env
        rec = DetectionRecorder(inner, rec_path)
        try:
            obs, info = rec.reset(seed=seed, options=options)
            sc_meta = info["scenario_cfg"]
            done = False
            while not done:
                obs, _, term, trunc, info = rec.step(None)
                done = term or trunc
        finally:
            rec.close()
        ep_meta = {
            "index": i, "seed": seed,
            "episode_id": info["gt"]["episode_id"] if "gt" in info else None,
            "scenario": sc_meta["name"],
            "noise_scale": sc_meta["noise_scale"],
            "clutter_scale": sc_meta["clutter_scale"],
            "path": os.path.relpath(rec_path, out_dir),
            "n_steps": info["gt"]["steps"] if "gt" in info else None,
        }
        manifest["episodes"].append(ep_meta)
        rec_paths.append(rec_path)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_dataset_with_qa(out_dir: str, episodes: int = 40, split: str = "train",
                          seed_offset: int = 0, duration_s: float = 60.0,
                          plots: bool = True, env_cfg=None) -> dict:
    """Dataset builder + distribution audit in one pass (single rollout per ep)."""
    os.makedirs(out_dir, exist_ok=True)
    lo, _hi = SEED_RANGES[split]
    base = SkyGymEnv(env_cfg)
    qa = QAChecker(base)
    monitor = DistributionMonitor(qa)
    env_cfg_dict = {"flight": None}

    episodes_meta = []
    for i in range(episodes):
        seed = lo + seed_offset + i
        options = {"duration_s": duration_s,
                   "noise_scale": list(NOISE_LEVELS.values())[i % 3],
                   "clutter_scale": list(CLUTTER_LEVELS.values())[i % 3]}
        ep_dir = os.path.join(out_dir, f"ep_{i:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        rec_env = DetectionRecorder(monitor, os.path.join(ep_dir, "detections.jsonl"))
        try:
            obs, info = rec_env.reset(seed=seed, options=options)
            sc_meta = info["scenario_cfg"]
            done = False
            while not done:
                obs, _, term, trunc, info = rec_env.step(None)
                done = term or trunc
        finally:
            rec_env.close()
        episodes_meta.append({
            "index": i, "seed": seed,
            "episode_id": info["gt"]["episode_id"],
            "scenario": sc_meta["name"],
            "noise_scale": sc_meta["noise_scale"],
            "clutter_scale": sc_meta["clutter_scale"],
            "steps": info["gt"]["steps"],
        })

    qa_rep = qa.summary()
    mon_rep = monitor.save_report(
        os.path.join(out_dir, "distribution_report.json"),
        plots_dir=os.path.join(out_dir, "plots") if plots else None)
    manifest = {"split": split, "episodes": episodes_meta,
                "qa": qa_rep, "distribution": {k: v for k, v in mon_rep.items()}}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest
