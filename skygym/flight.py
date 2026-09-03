"""3-DOF point-mass flight model and scripted autopilot behaviours.

The plant is deliberately simple (fast, vectorisable) but kinematically
honest: acceleration limits, quadratic drag, speed limits and a ground
floor are enforced, so no generated trajectory contains impossible motion.

Autopilot behaviours (control laws) are what a scenario 'flies':
    hover / approach / orbit / waypoint_cruise / serpentine / egress
They convert behaviour parameters into acceleration commands for the plant.
"""
from __future__ import annotations

import numpy as np

from .config import FlightCfg

EPS = 1e-6


class FlightState:
    """Kinematic state of one drone."""

    __slots__ = ("pos", "vel", "t")

    def __init__(self, pos: np.ndarray, vel: np.ndarray, t: float = 0.0):
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.asarray(vel, dtype=float).copy()
        self.t = float(t)

    def copy(self) -> "FlightState":
        return FlightState(self.pos, self.vel, self.t)


def clip_accel(a_cmd: np.ndarray, cfg: FlightCfg) -> np.ndarray:
    n = np.linalg.norm(a_cmd)
    if n > cfg.amax_mps2:
        return a_cmd * (cfg.amax_mps2 / (n + EPS))
    return a_cmd


def step_plant(state: FlightState, a_cmd: np.ndarray, cfg: FlightCfg) -> None:
    """Integrate one physics substep in place (semi-implicit Euler)."""
    a = clip_accel(np.asarray(a_cmd, dtype=float), cfg)
    a = a - cfg.drag_k * np.linalg.norm(state.vel) * state.vel  # quadratic drag
    state.vel = state.vel + a * cfg.dt_phys
    speed = np.linalg.norm(state.vel)
    if speed > cfg.vmax_mps:
        state.vel *= cfg.vmax_mps / speed
    state.pos = state.pos + state.vel * cfg.dt_phys
    state.t += cfg.dt_phys
    # soft ground floor: push up, kill downward speed
    if state.pos[2] < cfg.z_floor_m:
        state.pos[2] = cfg.z_floor_m
        if state.vel[2] < 0:
            state.vel[2] = 0.0
    if state.pos[2] > cfg.z_ceiling_m:
        state.pos[2] = cfg.z_ceiling_m
        if state.vel[2] > 0:
            state.vel[2] = 0.0


# --------------------------------------------------------------------------
# Angle-mode quadcopter controller (control mode only - data mode untouched)
# --------------------------------------------------------------------------

G_MPS2 = 9.81


class QuadAttitude:
    """Attitude state of one manually-flown quad (angle-mode stabilisation).

    The pilot's sticks are commands, not direct accelerations:
      pitch (+1 forward) / roll (+1 right)  -> tilt command (deg, capped)
      yaw (+1 clockwise)                    -> yaw-rate command (deg/s)
      climb (+1 up)                         -> climb-rate command (m/s)
    Tilt and yaw rate follow their commands through a first-order lag (the
    vehicle's rotational dynamics); horizontal acceleration is g*tan(tilt)
    rotated into ENU by yaw; vertical is a climb-rate servo. The returned
    ENU accel is fed to the SAME step_plant, so drag / speed limits /
    floors are shared with data mode and the trajectory stays physically
    honest.
    """

    __slots__ = ("yaw_deg", "tilt_fwd_deg", "tilt_right_deg", "yaw_rate_dps")

    def __init__(self, yaw_deg: float = 0.0):
        self.yaw_deg = float(yaw_deg) % 360.0
        self.tilt_fwd_deg = 0.0
        self.tilt_right_deg = 0.0
        self.yaw_rate_dps = 0.0

    def as_dict(self) -> dict:
        return {"yaw_deg": round(self.yaw_deg, 2),
                "pitch_deg": round(self.tilt_fwd_deg, 2),
                "roll_deg": round(self.tilt_right_deg, 2)}


def heading_from_vel(vel: np.ndarray, default_deg: float = 0.0) -> float:
    """Azimuth of the velocity vector (deg, 0 = North, clockwise) or default."""
    e, n = float(vel[0]), float(vel[1])
    if e * e + n * n < 4.0:        # slower than 2 m/s: keep the default
        return default_deg
    return float(np.degrees(np.arctan2(e, n))) % 360.0


def sticks_to_accel(att: QuadAttitude, sticks, vel: np.ndarray,
                    cfg_f: FlightCfg, quad) -> np.ndarray:
    """One controller tick at dt_phys granularity is NOT needed: the caller
    integrates per physics substep with dt = cfg_f.dt_phys. Returns the ENU
    acceleration command for step_plant."""
    dt = cfg_f.dt_phys
    pitch = float(np.clip(sticks[0], -1.0, 1.0)) * quad.tilt_max_deg
    roll = float(np.clip(sticks[1], -1.0, 1.0)) * quad.tilt_max_deg
    yrate = float(np.clip(sticks[2], -1.0, 1.0)) * quad.yaw_rate_max_dps
    climb = float(np.clip(sticks[3], -1.0, 1.0)) * quad.vz_max_mps

    # first-order lag toward the commanded attitude
    a_t = dt / max(quad.tau_tilt_s, 1e-3)
    att.tilt_fwd_deg += (pitch - att.tilt_fwd_deg) * a_t
    att.tilt_right_deg += (roll - att.tilt_right_deg) * a_t
    att.yaw_rate_dps += (yrate - att.yaw_rate_dps) * (dt / max(quad.tau_yaw_s, 1e-3))
    att.yaw_deg = (att.yaw_deg + att.yaw_rate_dps * dt) % 360.0

    # tilt -> horizontal acceleration, rotated into ENU by yaw (0 = North)
    yaw = np.radians(att.yaw_deg)
    fwd = np.array([np.sin(yaw), np.cos(yaw), 0.0])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    a_h = (np.tan(np.radians(att.tilt_fwd_deg)) * fwd
           + np.tan(np.radians(att.tilt_right_deg)) * right) * G_MPS2

    # climb-rate servo (Mode-2 throttle feel: release -> level off)
    a_z = float(np.clip(quad.kp_z * (climb - float(vel[2])),
                        -cfg_f.amax_mps2, cfg_f.amax_mps2))
    return clip_accel(a_h + np.array([0.0, 0.0, a_z]), cfg_f)


# --------------------------------------------------------------------------
# Autopilot behaviours: f(state, params, cfg) -> accel command
# --------------------------------------------------------------------------

def _velocity_tracking(state: FlightState, v_des: np.ndarray, cfg: FlightCfg,
                       kp: float = 1.2) -> np.ndarray:
    """Simple velocity servo: a = kp * (v_des - v)."""
    return clip_accel(kp * (v_des - state.vel), cfg)


def hover(state: FlightState, params: dict, cfg: FlightCfg) -> np.ndarray:
    """PD hold of the initial position."""
    hold = np.asarray(params["hold_pos"], dtype=float)
    a = 1.2 * (hold - state.pos) - 1.6 * state.vel
    return clip_accel(a, cfg)


def approach(state: FlightState, params: dict, cfg: FlightCfg) -> np.ndarray:
    """Constant-speed ingress toward a protected asset."""
    tgt = np.asarray(params["target_enu"], dtype=float)
    speed = params["speed"]
    to_t = tgt - state.pos
    dist = np.linalg.norm(to_t)
    if dist < 25.0:  # loiter on arrival instead of hitting the asset
        return hover(state, {"hold_pos": tgt + np.array([0, 0, 40.0])}, cfg)
    v_des = to_t / dist * speed
    return _velocity_tracking(state, v_des, cfg)


def egress(state: FlightState, params: dict, cfg: FlightCfg) -> np.ndarray:
    """Fly directly away from the site (radial escape)."""
    speed = params["speed"]
    away = state.pos.copy()
    away[2] = 0.0
    n = np.linalg.norm(away)
    if n < 1.0:
        away = np.array([1.0, 0.0, 0.0])
        n = 1.0
    v_des = away / n * speed
    v_des[2] = 0.0
    return _velocity_tracking(state, v_des, cfg)


def orbit(state: FlightState, params: dict, cfg: FlightCfg) -> np.ndarray:
    """Circle a centre point at fixed radius and speed."""
    center = np.asarray(params["center_enu"], dtype=float)
    radius = params["radius"]
    speed = params["speed"]
    rel = state.pos - center
    rel_h = rel[[0, 1]]
    r_now = np.linalg.norm(rel_h) + EPS
    # radial correction (proportional) + tangential unit vector
    radial = rel_h / r_now
    tangent = np.array([-radial[1], radial[0]])
    omega = speed / max(radius, 10.0)
    v_h_des = tangent * speed + radial * ((radius - r_now) * 0.2)
    v_des = np.array([v_h_des[0], v_h_des[1], 0.0])
    # gentle altitude hold at entry altitude
    v_des[2] = 0.4 * (params.get("alt", state.pos[2]) - state.pos[2])
    _ = omega  # kept for readability / future use
    return _velocity_tracking(state, v_des, cfg)


def waypoint_cruise(state: FlightState, params: dict, cfg: FlightCfg) -> np.ndarray:
    """Track a list of waypoints in order, advance within capture radius."""
    wps = params["waypoints"]
    idx = int(params.get("wp_idx", 0))
    speed = params["speed"]
    tgt = np.asarray(wps[min(idx, len(wps) - 1)], dtype=float)
    to_t = tgt - state.pos
    if np.linalg.norm(to_t) < 60.0 and idx < len(wps) - 1:
        params["wp_idx"] = idx + 1
        tgt = np.asarray(wps[idx + 1], dtype=float)
        to_t = tgt - state.pos
    v_des = to_t / (np.linalg.norm(to_t) + EPS) * speed
    return _velocity_tracking(state, v_des, cfg)


def serpentine(state: FlightState, params: dict, cfg: FlightCfg) -> np.ndarray:
    """Forward flight with lateral weave (the 2 g evader from the physics study).

    v_des = speed * (f_hat * cos(theta) + s_hat * sin(theta)),
    theta(t) = (2 pi / T) * integral ... implemented as phase accumulated
    from weave_period and weave_amplitude.
    """
    speed = params["speed"]
    period = max(params.get("weave_period", 8.0), 2.0)
    amp = params.get("weave_amplitude", 6.0)
    fwd = params.get("forward_dir", None)
    if fwd is None:
        fwd = state.vel.copy()
        fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1.0 else np.array([0.0, 1.0, 0.0])
    else:
        fwd = np.asarray(fwd, dtype=float)
        fwd = fwd / (np.linalg.norm(fwd) + EPS)
    side = np.cross(np.array([0.0, 0.0, 1.0]), fwd)  # horizontal right-hand side
    phase = 2.0 * np.pi * state.t / period
    v_lateral = amp * np.cos(phase)          # oscillating lateral velocity
    v_forward = np.sqrt(max(speed**2 - v_lateral**2, 0.25 * speed**2))
    v_des = fwd * v_forward + side * v_lateral
    return _velocity_tracking(state, v_des, cfg)


BEHAVIOURS = {
    "hover": hover,
    "approach": approach,
    "orbit": orbit,
    "waypoint_cruise": waypoint_cruise,
    "serpentine": serpentine,
    "egress": egress,
}


def build_params(scenario, state: FlightState, rng: np.random.Generator) -> dict:
    """Sample behaviour parameters for a scenario (uses scenario start box)."""
    name = scenario.name
    speed = rng.uniform(scenario.speed_min, scenario.speed_max)
    lo = np.asarray(scenario.start_min, dtype=float)
    hi = np.asarray(scenario.start_max, dtype=float)
    start = rng.uniform(lo, hi)
    if name == "hover":
        state.pos = start
        state.vel[:] = 0.0
        return {"hold_pos": start.copy(), "speed": 0.0}
    if name == "approach":
        state.pos = start
        tgt = np.asarray(scenario.target_enu, dtype=float) + np.array([0, 0, 30.0])
        state.vel[:] = _initial_velocity_toward(start, tgt, speed)
        return {"target_enu": tgt, "speed": speed}
    if name == "egress":
        state.pos = start
        state.vel[:] = _initial_velocity_toward(start, np.array([0.0, 0.0, start[2]]), -speed)
        return {"speed": speed}
    if name == "orbit":
        state.pos = start
        center = np.asarray(scenario.target_enu, dtype=float)
        state.vel[:] = _initial_tangential(start, center, speed)
        return {"center_enu": center, "radius": scenario.orbit_radius_m,
                "speed": speed, "alt": start[2]}
    if name == "waypoint_cruise":
        state.pos = start
        wps = _sample_waypoints(start, rng)
        state.vel[:] = _initial_velocity_toward(start, np.asarray(wps[0]), speed)
        return {"waypoints": wps, "wp_idx": 0, "speed": speed}
    if name == "serpentine":
        state.pos = start
        tgt = np.asarray(scenario.target_enu, dtype=float)
        f = _initial_velocity_toward(start, tgt, speed)
        state.vel[:] = f
        return {"speed": speed, "forward_dir": f,
                "weave_period": scenario.weave_period_s,
                "weave_amplitude": scenario.weave_amplitude_mps}
    raise ValueError(f"unknown scenario '{name}'")


def _initial_velocity_toward(p0: np.ndarray, p1: np.ndarray, speed: float) -> np.ndarray:
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    n = np.linalg.norm(d) + EPS
    v = d / n * abs(speed)
    v[2] *= 0.3  # shallow climb-out
    return v


def _initial_tangential(p: np.ndarray, center: np.ndarray, speed: float) -> np.ndarray:
    rel_h = (np.asarray(p, dtype=float) - np.asarray(center, dtype=float))[[0, 1]]
    n = np.linalg.norm(rel_h) + EPS
    t = np.array([-rel_h[1], rel_h[0]]) / n
    return np.array([t[0], t[1], 0.0]) * speed


def _sample_waypoints(start: np.ndarray, rng: np.random.Generator, n: int = 4) -> list:
    wps = []
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(n):
        ang += rng.uniform(0.4, 1.6)
        r = rng.uniform(400.0, 2200.0)
        wps.append([r * np.sin(ang), r * np.cos(ang), rng.uniform(50.0, 220.0)])
    return wps
