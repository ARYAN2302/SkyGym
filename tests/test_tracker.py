"""Tracker validation: the consumer must be able to eat the corruption."""
import numpy as np
import pytest

from skygym.config import EnvCfg, RadarCfg, EOCfg, RFCfg, SensorRig
from skygym.env import SkyGymEnv
from skygym.metrics import run_tracker_on_episode, aggregate
from skygym.wrappers import QAChecker, DistributionMonitor


def _env(clutter: float = 1.0, noise: float = 1.0) -> SkyGymEnv:
    cfg = EnvCfg()
    cfg.rig.radar.clutter_rate_per_scan = clutter
    return SkyGymEnv(cfg)


def test_clean_scene_tracking_quality():
    """No clutter, low noise: EKF must track tightly (honesty check)."""
    cfg = EnvCfg()
    cfg.rig.radar.clutter_rate_per_scan = 0.0
    env = SkyGymEnv(cfg)
    res = run_tracker_on_episode(env, seed=42,
                                 options={"scenario": "orbit",
                                          "duration_s": 60.0,
                                          "noise_scale": 0.5,
                                          "clutter_scale": 0.0})
    assert res["track_initiated"], res
    assert res["pos_rmse_m"] < 60.0, res
    assert res["track_continuity"] > 0.7, res


def test_cluttered_fusion_tracking():
    """Clutter ON: tracker must still initiate and reject false contacts."""
    cfg = EnvCfg()
    cfg.rig.radar.clutter_rate_per_scan = 1.5
    env = SkyGymEnv(cfg)
    results = []
    for seed in (101, 202, 303):
        r = run_tracker_on_episode(env, seed=seed,
                                   options={"scenario": "approach",
                                            "duration_s": 60.0,
                                            "noise_scale": 1.0,
                                            "clutter_scale": 1.0})
        results.append(r)
    agg = aggregate(results)
    assert agg["initiation_rate"] >= 0.66, agg
    # RMSE median must be sane under clutter
    assert agg["pos_rmse_m"]["median"] < 150.0, agg


def test_serpentine_evasive_track():
    """The 2g weaver from the intercept physics must still be trackable."""
    env = _env()
    res = run_tracker_on_episode(env, seed=77,
                                 options={"scenario": "serpentine",
                                          "duration_s": 60.0,
                                          "noise_scale": 1.0,
                                          "clutter_scale": 0.5})
    assert res["track_initiated"], res
    assert res["pos_rmse_m"] < 200.0, res


def test_qa_wrapper_catches_contract():
    """QA wrapper passes on healthy pipeline (and would raise otherwise)."""
    env = SkyGymEnv()
    qa = QAChecker(env)
    qa.reset(seed=9, options={"duration_s": 15.0})
    done, steps = False, 0
    while not done and steps < 200:
        _, _, te, tr, _ = qa.step(None)
        done = te or tr
        steps += 1
    assert qa.summary()["violations"] == 0


def test_distribution_monitor_coverage():
    env = SkyGymEnv()
    mon = DistributionMonitor(QAChecker(env))
    mon.reset(seed=13, options={"duration_s": 30.0})
    done, steps = False, 0
    while not done and steps < 400:
        _, _, te, tr, _ = mon.step(None)
        done = te or tr
        steps += 1
    rep = mon.report()
    assert rep["steps"] > 100
    assert rep["detections"]["radar"] > 50
    assert rep["detections"]["eo"] > 50
    # RF only if tx_on; radar+eo must have coverage regardless
    assert rep["range_stats"]["radar"]["n"] > 0
