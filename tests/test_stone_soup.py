"""Validation gate: the Stone Soup baseline must track and stay honest.

Runs one short approach episode through the full bridge
(env -> detections -> EKF/GNN -> grading) and asserts the consumer
contract: track initiated, continuity sane, position error bounded,
ID readout sane. Skipped when stonesoup is not installed.
"""
import pytest

ss = pytest.importorskip("stonesoup", reason="stonesoup extra not installed")

import numpy as np  # noqa: E402

from skygym.config import EnvCfg, ScenarioCfg  # noqa: E402
from skygym.env import SkyGymEnv  # noqa: E402
from skygym import stone_soup  # noqa: E402


def _make_env():
    sc = ScenarioCfg(
        name="approach", duration_s=10.0, seed=20260902,
        start_min=(-1296.0, 900.0, 60.0), start_max=(-900.0, 1296.0, 140.0),
        tx_on=True, noise_scale=1.0, clutter_scale=1.0)
    env = SkyGymEnv(EnvCfg())
    return env, {"scenario_cfg": sc}


def test_convention_selfcheck():
    ebr, eb, brg = stone_soup.verified_mappings()
    assert list(ebr) == [2, 0, 4] and list(eb) == [2, 0, 4]
    assert list(brg) == [2, 0]


def test_bridge_tracks_approach_episode():
    env, options = _make_env()
    summary, rows, n_frames = stone_soup.run_episode(
        env, seed=20260902, options=options, mode="fusion")
    assert n_frames >= 100
    assert summary["n_confirmed_tracks"] >= 1
    assert summary["pct_of_episode_tracked"] > 60.0
    assert summary["position_rmse_m"] < 60.0
    assert summary["mean_abs_az_err_deg"] < 0.5
    assert summary["id_accuracy_on_target_track"] is None or \
        summary["id_accuracy_on_target_track"] > 0.5


def test_metrics_adapter_contract():
    from skygym.metrics import run_tracker_on_episode, aggregate

    env, options = _make_env()
    res = run_tracker_on_episode(env, seed=20260902, options=options)
    assert res["track_initiated"] and res["n_steps"] >= 100
    assert res["position_rmse_m"] < 60.0
    agg = aggregate([res, res])
    assert agg["episodes"] == 2
    assert "position_rmse_m" in agg and agg["initiation_rate"] == 1.0


def test_radar_only_mode_smokes():
    env, options = _make_env()
    summary, rows, _ = stone_soup.run_episode(
        env, seed=20260902, options=options, mode="radar")
    assert summary["dets_fed"]["radar"] > 0
    assert summary["dets_fed"]["eo"] == 0 and summary["dets_fed"]["rf"] == 0
