"""Configuration dataclasses for SkyGym.

Every stochastic parameter lives here so that scenario sampling, QA checks
and downstream analysis all read the *same* specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# Canonical threat classes. Every sensor emits posteriors over these.
CLASSES: tuple[str, ...] = ("quad", "fixed_wing", "bird", "unknown")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Physical reference sizes / RCS used by the sensor models.
DRONE_SIZE_M = {"quad": 0.5, "fixed_wing": 1.8, "bird": 0.4, "unknown": 0.5}
RCS_M2 = {"quad": 0.05, "fixed_wing": 0.5, "bird": 0.02, "unknown": 0.05}


@dataclass
class FlightCfg:
    """3-DOF point-mass plant limits (small quadcopter / light fixed-wing)."""
    vmax_mps: float = 30.0          # max speed
    amax_mps2: float = 4.0          # max commanded acceleration (body-agnostic)
    drag_k: float = 0.015           # quadratic drag coefficient  a = -k |v| v
    z_floor_m: float = 2.0          # soft ground floor
    z_ceiling_m: float = 600.0      # airspace ceiling
    dt_phys: float = 0.02           # physics integration step (s)


@dataclass
class RadarCfg:
    """3D surveillance radar channel (detection-level model)."""
    rate_hz: float = 10.0
    latency_s: float = 0.06
    max_range_m: float = 8000.0
    # SNR model: SNR(r, rcs) = snr0_db + 10log10(rcs/rcs_ref) - 40log10(r/r_ref)
    # Calibrated: quad (0.05 m^2) -> Pd ~0.93 @ 2 km, ~0.04 @ 4 km;
    #             bird (0.02 m^2) -> Pd ~0.23 @ 2 km (fades fast).
    snr0_db: float = 32.0           # SNR at reference range for reference RCS
    rcs_ref_m2: float = 0.1
    ref_range_km: float = 1.0
    # Detection probability: logistic in SNR (Albersheim-like, Pfa ~ 1e-6)
    pd_mid_snr_db: float = 13.0     # Pd = 0.5 here
    pd_slope_db: float = 2.5
    # Range-dependent noise: sigma_r = c0 + c2 * r_km^2  (SNR falls as 1/r^4)
    range_sigma_base_m: float = 5.0
    range_sigma_per_km2_m: float = 3.0
    # Beamwidth-limited angle noise, grows linearly with range
    ang_sigma_base_deg: float = 0.6
    ang_sigma_per_km_deg: float = 0.5
    # Clutter: Poisson false contacts per scan (birds / ground / multipath)
    clutter_rate_per_scan: float = 0.8
    clutter_el_min_deg: float = 0.2
    clutter_el_max_deg: float = 8.0


@dataclass
class EOCfg:
    """EO/IR gimbal camera channel (detection-level model).

    Default mode is 'slaved': after cueing (radar/RF), the gimbal tracks the
    target, so boresight stays on target. With slaved=False the camera has a
    fixed boresight (North) and detections only occur inside the FOV.
    """
    rate_hz: float = 30.0
    latency_s: float = 0.03
    fov_deg: tuple[float, float] = (60.0, 40.0)   # horizontal, vertical
    res: tuple[int, int] = (1920, 1080)
    ang_sigma_deg: float = 0.08     # pixel-quantised bearing accuracy
    p50_pixel: float = 4.0          # angular size (px) giving Pd = 0.5
    pix_slope: float = 1.5
    max_range_m: float = 5000.0
    slaved: bool = True
    # Stereo ranging quality: sigma_r = r * (base + per_km2 * r_km^2)
    range_mode: Literal["none", "stereo"] = "stereo"
    range_sigma_base: float = 0.03
    range_sigma_per_km2: float = 0.05


@dataclass
class RFCfg:
    """RF direction-finding channel (az-only bearing + protocol fingerprint)."""
    rate_hz: float = 5.0
    latency_s: float = 0.10
    az_sigma_deg: float = 4.0
    pd_tx: float = 0.97             # detection prob when transmitter is on
    max_range_m: float = 6000.0
    # Per-scenario: is the drone's control link radiating?
    # (set on the scenario; rf.tx_on is the default for sampling)


@dataclass
class SensorRig:
    """The full recon suite + site position."""
    radar: RadarCfg = field(default_factory=RadarCfg)
    eo: EOCfg = field(default_factory=EOCfg)
    rf: RFCfg = field(default_factory=RFCfg)
    site_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioCfg:
    """One scenario = flight behaviour + start box + environment knobs."""
    name: str = "approach"
    seed: int = 0
    duration_s: float = 60.0
    # start box (ENU, m) - sampled uniformly
    start_min: tuple[float, float, float] = (-2500.0, -2500.0, 40.0)
    start_max: tuple[float, float, float] = (2500.0, 2500.0, 180.0)
    speed_min: float = 8.0
    speed_max: float = 24.0
    true_class: str = "quad"
    tx_on: bool = True
    # behaviour parameters
    target_enu: tuple[float, float, float] = (0.0, 0.0, 60.0)  # protected asset
    orbit_radius_m: float = 700.0
    weave_amplitude_mps: float = 6.0   # lateral weave velocity amplitude
    weave_period_s: float = 8.0
    # noise level multiplier applied to all sensor sigmas (grid axis)
    noise_scale: float = 1.0
    # clutter scale multiplier (grid axis)
    clutter_scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnvCfg:
    """Gym environment top-level config."""
    flight: FlightCfg = field(default_factory=FlightCfg)
    rig: SensorRig = field(default_factory=SensorRig)
    dt: float = 0.1                  # env step (s) - radar ticks 1:1 at 10 Hz
    max_dets_per_sensor: int = 24
    mode: Literal["data", "control"] = "data"
    sensor_dropout_seed: int | None = None
