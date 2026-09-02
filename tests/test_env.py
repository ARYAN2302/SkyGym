"""Env contract: API shapes, seed reproducibility, GT separation, bounds."""
import numpy as np
import pytest

import skygym
from skygym.config import EnvCfg, ScenarioCfg
from skygym.env import SkyGymEnv


def test_reset_step_shapes():
    env = SkyGymEnv()
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"radar", "eo", "rf"}
    for s, d in obs.items():
        assert d["dets"].shape == (24, 11)
        assert 0 <= d["n"] <= 24
    assert "gt" in info
    gt = info["gt"]
    assert gt["pos"].shape == (3,) and gt["vel"].shape == (3,)
    obs2, r, term, trunc, _ = env.step(None)
    assert not term and not trunc
    assert isinstance(r, float)


def test_seed_reproducibility():
    env = SkyGymEnv()
    a = [env.reset(seed=123)]
    seq_a = []
    done = False
    while not done:
        o, _, te, tr, _ = env.step(None)
        seq_a.append((o["radar"]["dets"].copy(), o["radar"]["n"]))
        done = te or tr
        if len(seq_a) > 50:
            break
    env2 = SkyGymEnv()
    _ = env2.reset(seed=123)
    i = 0
    done = False
    while not done and i < 50:
        o2, _, te, tr, _ = env2.step(None)
        assert np.array_equal(seq_a[i][0], o2["radar"]["dets"], equal_nan=True)
        assert seq_a[i][1] == o2["radar"]["n"]
        i += 1
        done = te or tr
    assert i > 10


def test_different_seeds_diverge():
    env = SkyGymEnv()
    o1, i1 = env.reset(seed=1)
    gt1 = i1["gt"]["pos"].copy()
    n1 = sum(obs["n"] for obs in o1.values())
    o2, i2 = env.reset(seed=2)
    n2 = sum(obs["n"] for obs in o2.values())
    assert (n1, n2) != (0, 0)
    # trajectories must differ (different start boxes)
    assert not np.allclose(i2["gt"]["pos"], gt1)


def test_no_gt_leakage_in_obs():
    env = SkyGymEnv()
    obs, info = env.reset(seed=7)
    for forbidden in ("gt", "labels", "truth", "witness"):
        assert forbidden not in obs
        assert forbidden not in info["gt"] or forbidden == "gt"
    for k in range(30):
        obs, _, te, tr, info = env.step(None)
        for forbidden in ("gt", "labels", "truth", "witness"):
            assert forbidden not in obs
        if te or tr:
            break


def test_control_mode_accepts_action():
    cfg = EnvCfg(mode="control")
    env = SkyGymEnv(cfg)
    env.reset(seed=5, options={"scenario": "hover"})
    p0 = env._state.pos.copy()
    for _ in range(20):
        obs, _, te, tr, info = env.step(np.array([4.0, 0.0, 0.0], dtype=np.float32))
    assert not np.allclose(p0, env._state.pos)
    # action clip: way over limit should not fling the drone to absurd speed
    assert np.linalg.norm(env._state.vel) <= env.cfg.flight.vmax_mps + 1e-6


def test_scenario_override_via_options():
    env = SkyGymEnv()
    obs, info = env.reset(seed=3, options={"scenario": "orbit",
                                           "duration_s": 20.0,
                                           "noise_scale": 0.5,
                                           "clutter_scale": 0.3})
    assert info["scenario_cfg"]["name"] == "orbit"
    assert info["scenario_cfg"]["noise_scale"] == 0.5
    done, steps = False, 0
    while not done and steps < 400:
        obs, _, te, tr, info = env.step(None)
        done = te or tr
        steps += 1
    assert done
    assert info["gt"]["t"] >= 20.0 - 0.2


def test_episode_terminates_within_duration():
    env = SkyGymEnv()
    env.reset(seed=11, options={"duration_s": 10.0})
    done, steps = False, 0
    while not done and steps < 500:
        _, _, te, tr, _ = env.step(None)
        done = te or tr
        steps += 1
    assert done and steps <= 120
