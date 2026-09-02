"""Flight model: physical feasibility limits enforced."""
import numpy as np

from skygym.config import FlightCfg
from skygym.flight import FlightState, step_plant, clip_accel, serpentine, approach


def test_accel_clip():
    cfg = FlightCfg(amax_mps2=4.0)
    a = clip_accel(np.array([100.0, 0.0, 0.0]), cfg)
    assert abs(np.linalg.norm(a) - 4.0) < 1e-5


def test_speed_limit_and_floor():
    cfg = FlightCfg(vmax_mps=30.0, z_floor_m=2.0)
    st = FlightState(np.array([0.0, 0.0, 50.0]), np.array([0.0, 0.0, 0.0]))
    for _ in range(3000):
        step_plant(st, np.array([10.0, 0.0, -50.0]), cfg)  # slam down+forward
        assert np.linalg.norm(st.vel) <= 30.0 + 1e-6
        assert st.pos[2] >= 2.0 - 1e-9
    # and it should have converged to forward motion, not crashed
    assert st.pos[2] <= cfg.z_ceiling_m


def test_serpentine_stays_bounded():
    cfg = FlightCfg()
    st = FlightState(np.array([0.0, 0.0, 100.0]), np.array([20.0, 0.0, 0.0]))
    params = {"speed": 20.0, "forward_dir": np.array([1.0, 0.0, 0.0]),
              "weave_period": 8.0, "weave_amplitude": 6.0}
    speeds = []
    for _ in range(5000):
        a = serpentine(st, params, cfg)
        step_plant(st, a, cfg)
        speeds.append(np.linalg.norm(st.vel))
    assert np.mean(speeds) < cfg.vmax_mps
    assert st.pos[2] >= cfg.z_floor_m


def test_approach_arrives():
    cfg = FlightCfg()
    st = FlightState(np.array([2000.0, 0.0, 150.0]), np.zeros(3))
    params = {"target_enu": np.array([0.0, 0.0, 60.0]), "speed": 15.0}
    for _ in range(20000):
        a = approach(st, params, cfg)
        step_plant(st, a, cfg)
        if np.linalg.norm(st.pos - params["target_enu"]) < 80.0:
            break
    assert np.linalg.norm(st.pos - params["target_enu"]) < 120.0
