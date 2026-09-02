"""SkyGymEnv - the Gymnasium-native recon playground.

The twist vs every other drone gym:
    obs  = what the SENSORS report about the target (corrupted detections)
    info = the hidden ground truth ("witness channel") - for labels/eval only

Two modes through one API:
    mode="data"    action=None  -> scripted autopilot flies, env records
    mode="control" action=Box(3) -> you drive (accel command), sensors report
"""
from __future__ import annotations

import uuid
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvCfg, ScenarioCfg, CLASSES, SensorRig, FlightCfg
from .flight import FlightState, step_plant, build_params, BEHAVIOURS
from .sensors import SENSORS, Detection
from . import world


def _dets_to_padded(dets: list[Detection], max_dets: int, dt_row_len: int) -> np.ndarray:
    arr = np.full((max_dets, dt_row_len), np.nan, dtype=np.float32)
    for i, d in enumerate(dets[:max_dets]):
        arr[i, :] = np.asarray(d.as_row(), dtype=np.float32)
    return arr


class SkyGymEnv(gym.Env):
    """One drone vs one sensor site. Detection-level recon gym."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: EnvCfg | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or EnvCfg()
        self.render_mode = render_mode
        self.row_len = 11  # see sensors.base.Detection.as_row
        self.max_dets = self.cfg.max_dets_per_sensor

        n_cls = len(CLASSES)
        self.observation_space = spaces.Dict({
            s: spaces.Dict({
                "dets": spaces.Box(-np.inf, np.inf,
                                   shape=(self.max_dets, self.row_len), dtype=np.float32),
                "n": spaces.Discrete(self.max_dets + 1),
            })
            for s in ("radar", "eo", "rf")
        })
        self.action_space = spaces.Box(
            low=-self.cfg.flight.amax_mps2, high=self.cfg.flight.amax_mps2,
            shape=(3,), dtype=np.float32)

        self._scenario: ScenarioCfg | None = None
        self._state: FlightState | None = None
        self._params: dict = {}
        self._sensors: dict[str, Any] = {}
        self._rng: np.random.Generator | None = None
        self._episode_id: str | None = None
        self._traj_hist: list[np.ndarray] = []
        self._steps = 0

    # ------------------------------------------------------------------ #
    def _make_obs(self, sensor_dets: dict[str, list[Detection]]) -> dict:
        obs = {}
        for name, dets in sensor_dets.items():
            obs[name] = {
                "dets": _dets_to_padded(dets, self.max_dets, self.row_len),
                "n": int(min(len(dets), self.max_dets)),
            }
        return obs

    def _gt_info(self, sensor_dets: dict[str, list[Detection]]) -> dict:
        """Witness channel: hidden ground truth. Consumers must not train on it
        as INPUT; it exists for labels and evaluation."""
        st = self._state
        site = np.asarray(self.cfg.rig.site_enu, dtype=float)
        az, el, rng_m = world.cartesian_to_spherical(st.pos - site)
        return {
            "t": st.t,
            "pos": st.pos.copy(),
            "vel": st.vel.copy(),
            "az_deg": az,
            "el_deg": el,
            "range_m": rng_m,
            "true_class": self._scenario.true_class,
            "tx_on": self._scenario.tx_on,
            "n_dets": {k: len(v) for k, v in sensor_dets.items()},
            "scenario": self._scenario.name,
            "episode_id": self._episode_id,
            "steps": self._steps,
        }

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        opts = dict(options or {})
        self._rng = np.random.default_rng(seed)
        self._episode_id = f"{int(self._rng.integers(0, 2**62)):016x}"[:12]

        # scenario: explicit override > options["scenario"] > default sample
        if "scenario_cfg" in opts and isinstance(opts["scenario_cfg"], ScenarioCfg):
            sc = opts["scenario_cfg"]
            sc.seed = seed if seed is not None else sc.seed
        else:
            from .scenarios import sample_scenario  # local import: avoid cycle
            sc = sample_scenario(
                self._rng,
                name=opts.get("scenario"),
                noise_scale=opts.get("noise_scale", 1.0),
                clutter_scale=opts.get("clutter_scale", 1.0),
                true_class=opts.get("true_class", "quad"),
                duration_s=opts.get("duration_s"),
                tx_on=opts.get("tx_on"),
            )
        self._scenario = sc

        # flight state + behaviour params
        self._state = FlightState(np.zeros(3), np.zeros(3))
        self._params = build_params(sc, self._state, self._rng)

        # sensors (fresh rngs derived from the master seed for stream isolation)
        rig: SensorRig = self.cfg.rig
        self._sensors = {
            "radar": SENSORS["radar"](rig.radar,
                                      np.random.default_rng(self._rng.integers(1 << 62)),
                                      noise_scale=sc.noise_scale,
                                      clutter_scale=sc.clutter_scale),
            "eo": SENSORS["eo"](rig.eo,
                                np.random.default_rng(self._rng.integers(1 << 62))),
            "rf": SENSORS["rf"](rig.rf,
                                np.random.default_rng(self._rng.integers(1 << 62))),
        }
        for s in self._sensors.values():
            s.reset()

        self._traj_hist = [self._state.pos.copy()]
        self._steps = 0

        dets = {k: list(s.poll(0.0, self._state.pos, self._drone_meta()))
                for k, s in self._sensors.items()}
        obs = self._make_obs(dets)
        info = {"gt": self._gt_info(dets), "scenario_cfg": sc.to_dict(),
                "rig": self.cfg.rig.to_dict(), "mode": self.cfg.mode}
        return obs, info

    def _drone_meta(self) -> dict:
        return {
            "site_enu": self.cfg.rig.site_enu,
            "true_class": self._scenario.true_class,
            "tx_on": self._scenario.tx_on,
            "noise_scale": self._scenario.noise_scale,
        }

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray | None = None):
        cfg: FlightCfg = self.cfg.flight
        sc: ScenarioCfg = self._scenario

        # --- control selection -----------------------------------------
        if action is None or self.cfg.mode == "data":
            behaviour = BEHAVIOURS[sc.name]
            a_cmd = behaviour(self._state, self._params, cfg)
        else:
            a_cmd = np.clip(np.asarray(action, dtype=float),
                            -cfg.amax_mps2, cfg.amax_mps2)

        # --- physics: substeps until the next env tick ------------------
        n_sub = max(1, int(round(self.cfg.dt / cfg.dt_phys)))
        for _ in range(n_sub):
            step_plant(self._state, a_cmd, cfg)

        self._steps += 1

        # --- sensors fire at their own rates ----------------------------
        sensor_dets = {k: list(s.poll(self._state.t, self._state.pos,
                                      self._drone_meta()))
                       for k, s in self._sensors.items()}
        self._traj_hist.append(self._state.pos.copy())

        # --- episode end conditions -------------------------------------
        pos = self._state.pos
        out_of_bounds = (abs(pos[0]) > 6000 or abs(pos[1]) > 6000
                         or pos[2] < 0.5 or pos[2] > 900)
        terminated = bool(out_of_bounds)
        truncated = bool(self._state.t >= sc.duration_s - 1e-9)

        reward = 0.0  # data mode: no reward; control-mode shaping comes later
        obs = self._make_obs(sensor_dets)
        info = {"gt": self._gt_info(sensor_dets),
                "out_of_bounds": out_of_bounds}
        if self.render_mode == "rgb_array":
            info["rgb"] = self.render()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):  # pragma: no cover - visual only
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        traj = np.asarray(self._traj_hist)
        fig = plt.figure(figsize=(11, 4.2))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(traj[:, 0] / 1000, traj[:, 1] / 1000, traj[:, 2], lw=1.2)
        ax1.scatter([0], [0], [0], c="r", marker="^", s=60, label="sensor site")
        ax1.set_xlabel("E km"); ax1.set_ylabel("N km"); ax1.set_zlabel("z m")
        ax1.set_title(f"Truth  |  {self._scenario.name}  t={self._state.t:.1f}s")
        ax1.legend()
        ax2 = fig.add_subplot(1, 2, 2)
        st = self._state
        site = np.asarray(self.cfg.rig.site_enu, dtype=float)
        az, el, r = world.cartesian_to_spherical(st.pos - site)
        ax2.scatter([az], [el], c="g", s=90, label="TRUE (witness)", marker="x")
        for name, color in (("radar", "tab:red"), ("eo", "tab:blue"), ("rf", "tab:orange")):
            n = self._sensors[name].cfg.rate_hz and None
        ax2.set_xlabel("azimuth deg"); ax2.set_ylabel("elevation deg")
        ax2.set_title("Bearing space (green = witness)")
        ax2.grid(alpha=0.3); ax2.legend()
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf

    def close(self):
        pass
