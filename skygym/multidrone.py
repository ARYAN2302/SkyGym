"""S5 - MultiDroneEnv: N drones vs one sensor site.

The single-target env teaches "track a drone". The multi-drone env teaches
"separate three drones that cross, merge and occlude each other inside the
clutter" - the regime where global assignment (Stone Soup GNN2D) visibly
beats greedy association and where learned trackers have room to shine.

Design notes
------------
- Same Gymnasium contract as SkyGymEnv: obs = corrupted detections per
  sensor, info["gt"] = witness. The ONLY difference: gt carries a `targets`
  list (one entry per drone) and the sensors observe the whole fleet at
  every scan via Sensor.poll_multi().
- Detections remain anonymous. No detection is tagged with the drone that
  produced it - recovering that correspondence IS the benchmark task.
- Fleet spawn: drones are seeded in distinct azimuth sectors around the
  site with behaviour-dependent radii, so approach trajectories converge on
  the protected asset (merging), orbits sweep through each other's sectors
  (crossing), and serpentine weavers cut across radar beams.
- Fully deterministic given (seed, options) - same seed, same fleet.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvCfg, ScenarioCfg
from .env import _dets_to_padded  # shared padded-obs helper
from .flight import (FlightState, step_plant, build_params, BEHAVIOURS,
                     QuadAttitude, sticks_to_accel, heading_from_vel)
from .sensors import SENSORS
from . import world

DEFAULT_MIX = ("approach", "orbit", "serpentine", "hover", "waypoint_cruise",
               "egress")

# behaviour -> (radius_min_m, radius_max_m) spawn band from the site
SPAWN_BAND = {
    "approach": (900.0, 1500.0),   # converges on the asset -> merging
    "orbit": (500.0, 1000.0),      # sweeps across sectors -> crossing
    "serpentine": (800.0, 1300.0),  # cuts across beams
    "hover": (500.0, 1200.0),
    "waypoint_cruise": (600.0, 1200.0),
    "egress": (300.0, 700.0),      # flies out -> leaves sensor coverage
}


def sample_fleet(rng: np.random.Generator, n_drones: int = 3,
                 mix: list[str] | None = None,
                 classes: list[str] | None = None,
                 tx_prob: float = 0.85,
                 start_radius: tuple[float, float] | None = None,
                 alt_band: tuple[float, float] = (40.0, 160.0),
                 noise_scale: float = 1.0, clutter_scale: float = 1.0,
                 duration_s: float = 20.0) -> list[ScenarioCfg]:
    """Sample a deterministic fleet of per-drone ScenarioCfgs.

    Sector spawn: drone k gets azimuth sector ~360deg*k/N (+/- jitter);
    radius drawn from the behaviour's spawn band (or `start_radius`).
    """
    if n_drones < 1 or n_drones > 8:
        raise ValueError("n_drones must be in [1, 8]")
    mix = list(mix) if mix else []
    if not mix:
        # interesting default: two ingressing + one orbiting + one weaving
        base = ["approach", "approach", "orbit", "serpentine",
                "hover", "waypoint_cruise", "egress", "hover"]
        mix = base[:n_drones]
        rng.shuffle(mix)
    while len(mix) < n_drones:
        mix.append(DEFAULT_MIX[int(rng.integers(len(DEFAULT_MIX)))])
    if classes is None:
        classes = ["quad" if rng.random() < 0.7 else "fixed_wing"
                   for _ in range(n_drones)]

    fleet: list[ScenarioCfg] = []
    for k in range(n_drones):
        name = mix[k % len(mix)]
        az = (360.0 * k / n_drones) + float(rng.uniform(-18.0, 18.0))
        lo, hi = start_radius or SPAWN_BAND.get(name, (600.0, 1400.0))
        r = float(rng.uniform(lo, hi))
        a = np.radians(az)
        px, py = r * np.sin(a), r * np.cos(a)
        pz = float(rng.uniform(*alt_band))
        speed_lo, speed_hi = (18.0, 30.0) if classes[k] == "fixed_wing" \
            else (6.0, 18.0)
        # tight start box around the sector point (build_params samples from it)
        jit = 30.0
        sc = ScenarioCfg(
            name=name,
            seed=int(rng.integers(0, 1 << 31)),
            duration_s=duration_s,
            start_min=(px - jit, py - jit, max(20.0, pz - jit)),
            start_max=(px + jit, py + jit, pz + jit),
            speed_min=speed_lo, speed_max=speed_hi,
            true_class=classes[k],
            tx_on=bool(rng.random() < tx_prob),
            target_enu=(0.0, 0.0, 60.0),
            orbit_radius_m=float(rng.uniform(300.0, 700.0)),
            weave_amplitude_mps=float(rng.uniform(4.0, 9.0)),
            weave_period_s=float(rng.uniform(5.0, 10.0)),
            noise_scale=noise_scale,
            clutter_scale=clutter_scale,
        )
        fleet.append(sc)
    return fleet


class MultiDroneEnv(gym.Env):
    """N drones vs one sensor site. Detection-level multi-target recon gym.

    action (optional): accel command for drone 0 only (the fleet keeps its
    autopilot). Data mode (action=None) is the primary usage: every drone
    flies its scripted behaviour while the rig observes all of them.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, cfg: EnvCfg | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or EnvCfg()
        self.render_mode = render_mode
        self.row_len = 11
        self.max_dets = self.cfg.max_dets_per_sensor

        self.observation_space = spaces.Dict({
            s: spaces.Dict({
                "dets": spaces.Box(-np.inf, np.inf,
                                   shape=(self.max_dets, self.row_len),
                                   dtype=np.float32),
                "n": spaces.Discrete(self.max_dets + 1),
            })
            for s in ("radar", "eo", "rf")
        })
        self.action_space = spaces.Box(
            low=-self.cfg.flight.amax_mps2, high=self.cfg.flight.amax_mps2,
            shape=(3,), dtype=np.float32)

        self._fleet: list[ScenarioCfg] = []
        self._states: list[FlightState] = []
        self._params: list[dict] = []
        self._sensors: dict[str, Any] = {}
        self._rng: np.random.Generator | None = None
        self._episode_id: str | None = None
        self._traj_hist: list[list[np.ndarray]] = []
        self._steps = 0
        self._oob: list[bool] = []
        self._quads: dict[int, QuadAttitude] = {}   # per-drone attitude (control)
        self._control_idx = -1

    # ------------------------------------------------------------------ #
    def set_behaviour(self, k: int, name: str) -> dict:
        """Switch drone k's autopilot behaviour MID-EPISODE (interactive use).

        Builds fresh behaviour parameters from the drone's CURRENT state so
        the transition is continuous (no teleport): hover holds here,
        orbit uses the current range to the asset as radius, serpentine
        weaves along the current heading, waypoint_cruise samples a fresh
        route from the site. The drone keeps flying under BEHAVIOURS[name]
        from the next step().

        Playground/benchmark interactive feature only - the scripted
        single-behaviour data generator (data mode without this call) is
        unchanged and stays fully deterministic per seed.
        """
        if not (0 <= k < len(self._states)):
            raise IndexError(f"drone index {k} out of range "
                             f"(fleet of {len(self._states)})")
        if name not in BEHAVIOURS:
            raise ValueError(f"unknown behaviour '{name}' - "
                             f"choose from {sorted(BEHAVIOURS)}")
        st, sc = self._states[k], self._fleet[k]
        sc.name = name
        speed = float(np.clip(np.linalg.norm(st.vel[[0, 1]]),
                              max(sc.speed_min, 1.0), sc.speed_max))
        pr: dict = {"speed": speed}
        if name == "hover":
            pr["hold_pos"] = st.pos.copy()
        elif name == "approach":
            pr["target_enu"] = (np.asarray(sc.target_enu, dtype=float)
                                + np.array([0.0, 0.0, 30.0]))
        elif name == "orbit":
            center = np.asarray(sc.target_enu, dtype=float)
            rel = (st.pos - center)[[0, 1]]
            pr["center_enu"] = center
            pr["radius"] = float(np.clip(np.linalg.norm(rel) + 1e-6,
                                         150.0, 1200.0))
            pr["alt"] = float(st.pos[2])
        elif name == "waypoint_cruise":
            pr["waypoints"] = _sample_waypoints(st.pos, self._rng)
            pr["wp_idx"] = 0
        elif name == "serpentine":
            fwd = st.vel.copy()
            fwd[2] = 0.0
            n = np.linalg.norm(fwd)
            fwd = fwd / n if n > 1.0 else np.array([0.0, 1.0, 0.0])
            pr["forward_dir"] = fwd
            pr["weave_period"] = float(sc.weave_period_s)
            pr["weave_amplitude"] = float(sc.weave_amplitude_mps)
        self._params[k] = pr
        return {"idx": k, "behaviour": name,
                "speed_mps": round(speed, 2),
                "params": {kk: (np.asarray(vv).tolist()
                                if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in pr.items()}}

    # ------------------------------------------------------------------ #
    def _make_obs(self, sensor_dets: dict[str, list]):
        return {name: {"dets": _dets_to_padded(dets, self.max_dets,
                                               self.row_len),
                       "n": int(min(len(dets), self.max_dets))}
                for name, dets in sensor_dets.items()}

    def _drone_meta(self, k: int) -> dict:
        sc = self._fleet[k]
        return {
            "site_enu": self.cfg.rig.site_enu,
            "true_class": sc.true_class,
            "tx_on": sc.tx_on,
            "noise_scale": sc.noise_scale,
        }

    def _gt_info(self, sensor_dets: dict[str, list]) -> dict:
        site = np.asarray(self.cfg.rig.site_enu, dtype=float)
        targets = []
        for k, (st, sc) in enumerate(zip(self._states, self._fleet)):
            az, el, rng_m = world.cartesian_to_spherical(st.pos - site)
            tg = {
                "idx": k,
                "pos": st.pos.copy(),
                "vel": st.vel.copy(),
                "az_deg": az, "el_deg": el, "range_m": rng_m,
                "true_class": sc.true_class,
                "tx_on": sc.tx_on,
                "behaviour": sc.name,
                "oob": self._oob[k],
            }
            if k in self._quads:            # manually-flown drones report attitude
                tg["attitude"] = self._quads[k].as_dict()
            targets.append(tg)
        st0 = self._states[0]
        return {
            "t": st0.t,
            "n_targets": len(self._states),
            "targets": targets,
            # drone-0 convenience fields (back-compat with single-target tooling)
            "pos": st0.pos.copy(),
            "vel": st0.vel.copy(),
            "az_deg": targets[0]["az_deg"],
            "el_deg": targets[0]["el_deg"],
            "range_m": targets[0]["range_m"],
            "true_class": self._fleet[0].true_class,
            "tx_on": self._fleet[0].tx_on,
            "noise_scale": self._fleet[0].noise_scale,
            "n_dets": {k: len(v) for k, v in sensor_dets.items()},
            "scenario": "multi",
            "episode_id": self._episode_id,
            "steps": self._steps,
        }

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        opts = dict(options or {})
        self._rng = np.random.default_rng(seed)
        self._episode_id = f"{int(self._rng.integers(0, 2**62)):016x}"[:12]

        n = int(opts.get("n_drones", 3))
        self._fleet = sample_fleet(
            self._rng, n_drones=n,
            mix=opts.get("mix"),
            classes=opts.get("classes"),
            tx_prob=float(opts.get("tx_prob", 0.85)),
            start_radius=opts.get("start_radius"),
            noise_scale=float(opts.get("noise_scale", 1.0)),
            clutter_scale=float(opts.get("clutter_scale", 1.0)),
            duration_s=float(opts.get("duration_s", 20.0)),
        )
        if "scenario_cfg" in opts and opts["scenario_cfg"] is not None:
            # explicit per-drone override: list of ScenarioCfg
            scs = opts["scenario_cfg"]
            self._fleet = list(scs)

        self._states = [FlightState(np.zeros(3), np.zeros(3))
                        for _ in self._fleet]
        self._params = [build_params(sc, st, self._rng)
                        for sc, st in zip(self._fleet, self._states)]

        rig = self.cfg.rig
        self._sensors = {
            "radar": SENSORS["radar"](
                rig.radar, np.random.default_rng(self._rng.integers(1 << 62)),
                noise_scale=self._fleet[0].noise_scale,
                clutter_scale=self._fleet[0].clutter_scale),
            "eo": SENSORS["eo"](
                rig.eo, np.random.default_rng(self._rng.integers(1 << 62))),
            "rf": SENSORS["rf"](
                rig.rf, np.random.default_rng(self._rng.integers(1 << 62))),
        }
        for s in self._sensors.values():
            s.reset()

        self._oob = [False] * len(self._states)
        self._traj_hist = [[st.pos.copy()] for st in self._states]
        self._steps = 0
        self._quads = {}          # fresh attitude per episode
        self._control_idx = -1
        if self.cfg.mode == "control":   # the default possessed drone reports
            self._quads[0] = QuadAttitude(   # attitude from the first frame
                heading_from_vel(self._states[0].vel))

        positions = [st.pos for st in self._states]
        metas = [self._drone_meta(k) for k in range(len(self._states))]
        dets = {k: list(s.poll_multi(0.0, positions, metas))
                for k, s in self._sensors.items()}
        obs = self._make_obs(dets)
        info = {"gt": self._gt_info(dets),
                # compat summary for single-target tooling (recorder/QA/dataset)
                "scenario_cfg": {
                    "name": "multi", "n_drones": len(self._fleet),
                    "duration_s": self._fleet[0].duration_s,
                    "noise_scale": self._fleet[0].noise_scale,
                    "clutter_scale": self._fleet[0].clutter_scale,
                },
                "scenario_cfgs": [sc.to_dict() for sc in self._fleet],
                "rig": self.cfg.rig.to_dict(), "mode": self.cfg.mode}
        return obs, info

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray | None = None,
             control_idx: int = 0):
        """Advance the fleet one env tick.

        action: None (data mode / full autopilot) OR
          - shape (4,): pilot sticks [pitch, roll, yaw-rate, climb] in [-1,1]
            (angle-mode quad, applied to drone `control_idx`; attitude is
            integrated per physics substep)
          - shape (3,): legacy ENU accel command for drone `control_idx`
        All other drones keep their scripted behaviour.
        """
        cfg = self.cfg.flight
        stick_mode = False
        act = None
        if (action is not None and self.cfg.mode == "control"
                and 0 <= int(control_idx) < len(self._states)):
            act = np.asarray(action, dtype=float)
            stick_mode = act.shape == (4,)
            self._control_idx = int(control_idx)
            if stick_mode and self._control_idx not in self._quads:
                st = self._states[self._control_idx]
                self._quads[self._control_idx] = QuadAttitude(
                    heading_from_vel(st.vel))
        else:
            self._control_idx = -1

        for k, (st, pr) in enumerate(zip(self._states, self._params)):
            sc = self._fleet[k]
            if k == self._control_idx and act is not None:
                n_sub = max(1, int(round(self.cfg.dt / cfg.dt_phys)))
                if stick_mode:
                    quad = self._quads[self._control_idx]
                    for _ in range(n_sub):
                        a_cmd = sticks_to_accel(quad, act, st.vel,
                                                cfg, self.cfg.quad)
                        step_plant(st, a_cmd, cfg)
                    continue
                a_cmd = np.clip(act.reshape(-1)[:3],
                                -cfg.amax_mps2, cfg.amax_mps2)
            else:
                behaviour = BEHAVIOURS[sc.name]
                a_cmd = behaviour(st, pr, cfg)
            n_sub = max(1, int(round(self.cfg.dt / cfg.dt_phys)))
            for _ in range(n_sub):
                step_plant(st, a_cmd, cfg)
            # per-drone out-of-bounds (frozen once flagged: no resurrection)
            p = st.pos
            self._oob[k] = self._oob[k] or bool(
                abs(p[0]) > 6000 or abs(p[1]) > 6000
                or p[2] < 0.5 or p[2] > 900)

        self._steps += 1

        positions = [st.pos for st in self._states]
        metas = [self._drone_meta(k) for k in range(len(self._states))]
        sensor_dets = {k: list(s.poll_multi(self._states[0].t, positions, metas))
                       for k, s in self._sensors.items()}
        for k, st in enumerate(self._states):
            self._traj_hist[k].append(st.pos.copy())

        terminated = bool(all(self._oob))
        truncated = bool(self._states[0].t >= self._fleet[0].duration_s - 1e-9)

        obs = self._make_obs(sensor_dets)
        info = {"gt": self._gt_info(sensor_dets),
                "out_of_bounds": all(self._oob)}
        if self.render_mode == "rgb_array":
            info["rgb"] = self.render()
        return obs, 0.0, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):  # pragma: no cover - visual only
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        fig = plt.figure(figsize=(11, 4.2))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self._traj_hist), 1)))
        for k, hist in enumerate(self._traj_hist):
            tr = np.asarray(hist)
            ax1.plot(tr[:, 0] / 1000, tr[:, 1] / 1000, tr[:, 2], lw=1.2,
                     color=colors[k], label=f"D{k + 1}:{self._fleet[k].name}")
        ax1.scatter([0], [0], [0], c="r", marker="^", s=60, label="site")
        ax1.set_xlabel("E km"); ax1.set_ylabel("N km"); ax1.set_zlabel("z m")
        ax1.set_title(f"Truth (fleet of {len(self._states)})  "
                      f"t={self._states[0].t:.1f}s")
        ax1.legend(fontsize=7)
        ax2 = fig.add_subplot(1, 2, 2)
        site = np.asarray(self.cfg.rig.site_enu, dtype=float)
        for k, st in enumerate(self._states):
            az, el, _ = world.cartesian_to_spherical(st.pos - site)
            ax2.scatter([az], [el], s=70, marker="x", color=colors[k],
                        label=f"D{k + 1}")
        ax2.set_xlabel("azimuth deg"); ax2.set_ylabel("elevation deg")
        ax2.set_title("Bearing space (witness)")
        ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf

    def close(self):
        pass
