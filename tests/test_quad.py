"""Angle-mode quad controller tests (control mode only).

Guarantees:
- sticks are commands (attitude lags, never jumps)
- tilt translates to acceleration along the yaw heading (0 = North)
- yaw integrates at a bounded rate
- climb is a rate command (release -> level off)
- legacy 3-float ENU accel actions keep working
- data mode is byte-for-byte unaffected (no attitude in gt)
- MultiDroneEnv possession: only the possessed drone leaves autopilot
"""
import math

import numpy as np
import pytest

from skygym.config import EnvCfg, QuadCfg
from skygym.env import SkyGymEnv
from skygym.flight import QuadAttitude, heading_from_vel, sticks_to_accel
from skygym.multidrone import MultiDroneEnv


def _heading_deg(dE, dN):
    return math.degrees(math.atan2(dE, dN)) % 360.0


def test_attitude_lags_never_jumps():
    att = QuadAttitude(yaw_deg=0.0)
    quad = QuadCfg()
    from skygym.config import FlightCfg
    cfg_f = FlightCfg()
    sticks = [1.0, 0.0, 0.0, 0.0]
    vel = np.zeros(3)
    tilts = [sticks_to_accel(att, sticks, vel, cfg_f, quad) for _ in range(3)]
    # tilt grows smoothly toward the 22 deg cap: accel at tick 1 << tick 3
    assert np.linalg.norm(tilts[0][:2]) < np.linalg.norm(tilts[2][:2])
    assert att.tilt_fwd_deg <= quad.tilt_max_deg + 1e-6
    # accel bound respected
    for a in tilts:
        assert np.linalg.norm(a) <= cfg_f.amax_mps2 + 1e-6


def test_pitch_accelerates_along_heading():
    env = SkyGymEnv(EnvCfg(mode="control"))
    obs, info = env.reset(seed=7, options={"scenario": "approach",
                                           "duration_s": 30})
    v0 = info["gt"]["vel"]
    hdg = heading_from_vel(v0)
    p0 = info["gt"]["pos"].copy()
    for _ in range(10):
        obs, _, te, tr, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    p1 = info["gt"]["pos"]
    moved = _heading_deg(p1[0] - p0[0], p1[1] - p0[1])
    assert abs((moved - hdg + 180) % 360 - 180) < 15.0   # moved along heading
    assert info["gt"]["attitude"]["pitch_deg"] > 15.0    # nosed down


def test_tilt_accel_rotates_with_yaw():
    """Unit check: yaw=90 (east) + full pitch -> accel points EAST."""
    att = QuadAttitude(yaw_deg=90.0)
    quad = QuadCfg()
    from skygym.config import FlightCfg
    cfg_f = FlightCfg()
    for _ in range(30):                       # let tilt reach the cap
        a = sticks_to_accel(att, [1.0, 0.0, 0.0, 0.0], np.zeros(3), cfg_f, quad)
    assert a[0] > 3.5 and abs(a[1]) < 0.5     # east, not north


def test_yaw_then_pitch_from_hover():
    """From rest, displacement after yaw+pitch follows the quad's yaw."""
    env = SkyGymEnv(EnvCfg(mode="control"))
    env.reset(seed=3, options={"scenario": "hover", "duration_s": 60})
    for _ in range(20):                       # ~2 s yaw right from rest
        env.step(np.array([0.0, 0.0, 1.0, 0.0]))
    yaw = env._quad.yaw_deg
    p0 = env._state.pos.copy()
    for _ in range(20):                       # 2 s full pitch
        obs, _, te, tr, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    p1 = env._state.pos
    moved = _heading_deg(p1[0] - p0[0], p1[1] - p0[1])
    assert abs((moved - yaw + 180) % 360 - 180) < 25.0


def test_climb_is_rate_command():
    env = SkyGymEnv(EnvCfg(mode="control"))
    env.reset(seed=5, options={"scenario": "approach", "duration_s": 60})
    for _ in range(25):                                   # 2.5 s full climb
        obs, _, te, tr, info = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
    assert info["gt"]["vel"][2] > 2.5                     # near vz_max
    for _ in range(15):                                   # release
        obs, _, te, tr, info = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
    assert abs(info["gt"]["vel"][2]) < 1.0                # levelled off


def test_legacy_3vec_action_still_works():
    env = SkyGymEnv(EnvCfg(mode="control"))
    obs, info = env.reset(seed=7, options={"scenario": "approach",
                                           "duration_s": 20})
    obs, _, te, tr, info = env.step(np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(info["gt"]["pos"]).all()
    obs, _, te, tr, info = env.step(np.array([0.0, 0.0, 4.0]))
    assert info["gt"]["vel"][2] > 0.0                     # pushed up


def test_data_mode_has_no_attitude():
    env = SkyGymEnv(EnvCfg(mode="data"))
    obs, info = env.reset(seed=7, options={"scenario": "approach",
                                           "duration_s": 10})
    obs, _, te, tr, info = env.step(None)
    assert "attitude" not in info["gt"]
    assert env.action_space.shape == (3,)


def test_multidrone_possession_routes_to_one_drone():
    m = MultiDroneEnv(EnvCfg(mode="control", max_dets_per_sensor=96))
    obs, info = m.reset(seed=7, options={"n_drones": 3, "duration_s": 30})
    p_all = [tg["pos"].copy() for tg in info["gt"]["targets"]]
    for _ in range(10):
        obs, _, te, tr, info = m.step(np.array([1.0, 0.0, 0.0, 0.0]),
                                      control_idx=2)
    moved = [np.linalg.norm(info["gt"]["targets"][k]["pos"] - p_all[k])
             for k in range(3)]
    assert "attitude" in info["gt"]["targets"][2]
    # drone 0 is the default possessed drone (attitude pre-created at reset);
    # drone 1 stays a pure autopilot and must not report attitude
    assert "attitude" in info["gt"]["targets"][0]
    assert "attitude" not in info["gt"]["targets"][1]
    # manual drone accelerates promptly (attitude commanded); autopilot keeps
    # its smooth velocity tracking - both move, none teleport
    assert max(moved) < 60.0
    assert min(moved) > 0.0


def test_multidrone_data_mode_ignores_control_idx():
    m = MultiDroneEnv(EnvCfg(mode="data", max_dets_per_sensor=96))
    m.reset(seed=7, options={"n_drones": 3, "duration_s": 10})
    obs, _, te, tr, info = m.step(np.array([1.0, 1.0, 1.0, 1.0]),
                                  control_idx=1)
    for tg in info["gt"]["targets"]:
        assert "attitude" not in tg


def test_heading_from_vel():
    v = np.array([0.0, 10.0, 0.0])          # north
    assert abs(heading_from_vel(v) - 0.0) < 1e-9
    v = np.array([10.0, 0.0, 0.0])          # east
    assert abs(heading_from_vel(v) - 90.0) < 1e-9
    v = np.array([0.0, 0.0, 0.0])           # slow -> default
    assert heading_from_vel(v, default_deg=42.0) == 42.0
