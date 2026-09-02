"""S5 multi-drone tests: env contract, determinism, recorder, tracking."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skygym.config import EnvCfg
from skygym.multidrone import MultiDroneEnv, sample_fleet
from skygym.sensors.base import Sensor


def test_poll_multi_single_target_matches_poll():
    """poll(pos) must be exactly poll_multi([pos]) for any sensor setup."""
    class Probe(Sensor):
        name = "probe"

        def _observe(self, t_meas, drone_pos, meta):
            from skygym.sensors.base import Detection
            return Detection(sensor=self.name, t_meas=t_meas,
                             az_deg=float(drone_pos[0]) % 360.0, el_deg=5.0,
                             range_m=float(drone_pos[1]), cls={})

    class _Cfg:
        rate_hz = 10.0

    s1 = Probe.__new__(Probe)
    Sensor.__init__(s1, _Cfg(), np.random.default_rng(3))
    s2 = Probe.__new__(Probe)
    Sensor.__init__(s2, _Cfg(), np.random.default_rng(3))
    pos, meta = np.array([100.0, 200.0, 50.0]), {"x": 1}
    out_single = []
    for t in (0.0, 0.1, 0.2, 0.35):
        out_single.extend(s1.poll(t, pos, meta))
    out_multi = []
    for t in (0.0, 0.1, 0.2, 0.35):
        out_multi.extend(s2.poll_multi(t, [pos], [meta]))
    assert len(out_single) == len(out_multi) > 0
    assert all(a.az_deg == b.az_deg and a.t_meas == b.t_meas
               for a, b in zip(out_single, out_multi))


def test_poll_multi_observes_all_targets_per_scan():
    """One due scan must yield one detection attempt per target."""
    class Always(Sensor):
        name = "always"

        def _observe(self, t_meas, drone_pos, meta):
            from skygym.sensors.base import Detection
            return Detection(sensor=self.name, t_meas=t_meas,
                             az_deg=float(meta["tag"]), el_deg=0.0,
                             range_m=1.0, cls={})

    class _Cfg:
        rate_hz = 10.0

    s = Always.__new__(Always)
    Sensor.__init__(s, _Cfg(), np.random.default_rng(0))
    dets = s.poll_multi(0.15, [np.zeros(3)] * 3,
                        [{"tag": i} for i in range(3)])
    # scans at t=0.0 and t=0.1 are due within 0.15 (rate 10 Hz)
    assert len(dets) == 6
    assert sorted(d.az_deg for d in dets) == [0.0] * 2 + [1.0] * 2 + [2.0] * 2


def test_fleet_sampling_deterministic_and_valid():
    f1 = sample_fleet(np.random.default_rng(7), 3)
    f2 = sample_fleet(np.random.default_rng(7), 3)
    assert [s.to_dict() for s in f1] == [s.to_dict() for s in f2]
    assert len(f1) == 3
    for sc in f1:
        assert sc.name in ("approach", "orbit", "serpentine", "hover",
                           "waypoint_cruise", "egress")
        assert sc.true_class in ("quad", "fixed_wing")
        # sector spawn: distinct azimuths, plausible radii
        assert np.hypot(sc.start_min[0], sc.start_min[1]) > 200.0
    with pytest.raises(ValueError):
        sample_fleet(np.random.default_rng(0), 9)


def test_multidrone_env_contract_and_determinism():
    env = MultiDroneEnv(EnvCfg(max_dets_per_sensor=36))
    obs, info = env.reset(seed=123, options={"n_drones": 3, "duration_s": 2.0})
    assert info["gt"]["n_targets"] == 3
    assert len(info["gt"]["targets"]) == 3
    assert set(obs) == {"radar", "eo", "rf"}
    for k in range(5):
        obs, _, term, trunc, info = env.step(None)
    # every target moved and carries its own truth block
    for tg in info["gt"]["targets"]:
        assert len(tg["pos"]) == 3 and len(tg["vel"]) == 3
        assert tg["true_class"] in ("quad", "fixed_wing")
        assert tg["behaviour"] in ("approach", "orbit", "serpentine",
                                   "hover", "waypoint_cruise", "egress")
    # same seed -> identical streams (NaN-aware compare)
    env2 = MultiDroneEnv(EnvCfg(max_dets_per_sensor=36))
    env2.reset(seed=123, options={"n_drones": 3, "duration_s": 2.0})
    for _ in range(5):
        obs2, _, _, _, _ = env2.step(None)
    for s in ("radar", "eo", "rf"):
        n1, n2 = obs[s]["n"], obs2[s]["n"]
        assert n1 == n2
        assert np.allclose(obs[s]["dets"][:n1], obs2[s]["dets"][:n2],
                           equal_nan=True)


def test_multidrone_more_targets_more_detections():
    """A fleet must produce strictly more radar target dets than one drone."""
    def mean_radar(n, seed=5):
        env = MultiDroneEnv(EnvCfg(max_dets_per_sensor=48))
        env.reset(seed=seed, options={"n_drones": n, "duration_s": 4.0,
                                      "clutter_scale": 0.0})
        tot = 0
        for _ in range(20):
            obs, _, _, _, info = env.step(None)
            tot += info["gt"]["n_dets"]["radar"]
        return tot

    assert mean_radar(3) > mean_radar(1)


def test_recorder_records_per_target_labels():
    from skygym.wrappers import DetectionRecorder
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ep.jsonl")
        env = MultiDroneEnv(EnvCfg(max_dets_per_sensor=24))
        rec = DetectionRecorder(env, path)
        rec.reset(seed=9, options={"n_drones": 2, "duration_s": 1.0})
        for _ in range(5):
            rec.step(None)
        rec.close()
        with open(path, encoding="utf-8") as f:
            line = json.loads(f.readline())
        labels = line["labels"]
        assert len(labels["targets"]) == 2
        assert {"idx", "pos", "vel", "true_class", "behaviour"} \
            <= set(labels["targets"][0])


def test_run_episode_multi_grades_all_targets():
    from skygym.stone_soup import run_episode_multi
    pytest.importorskip("stonesoup")
    pytest.importorskip("scipy")
    env = MultiDroneEnv(EnvCfg(max_dets_per_sensor=36))
    summary, rows, n_frames = run_episode_multi(
        env, 20260902, options={"n_drones": 3, "duration_s": 8.0},
        mode="fusion")
    assert summary["n_targets"] == 3
    assert len(summary["per_target"]) == 3
    assert n_frames >= 80
    for p in summary["per_target"]:
        assert 0.0 <= p["tracked_pct"] <= 100.0
        assert p["tracked_pct"] > 50.0        # near fleet, fusion: must hold
        assert p["n_tracks_used"] <= 3        # no pathological identity churn
    assert summary["identity_switches"] <= 2
    # row schema: [t, tid, tgt, e, n, u, ve, vn, vu, err, az_e, el_e, r_e, near]
    assert len(rows[0]) == 14
