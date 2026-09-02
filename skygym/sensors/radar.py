"""Radar channel: az/el/range + Pd gating + Poisson clutter + micro-Doppler ID.

Physics baked in:
- SNR falls as 1/r^4 -> Pd collapses with range and small RCS.
- Range noise sigma ~ r^2 (measurement precision follows SNR).
- Angle noise sigma grows with range (fixed beamwidth).
- Clutter is a Poisson point process in az/el/range (birds, ground).
"""
from __future__ import annotations

import numpy as np

from ..config import RadarCfg, RCS_M2
from ..world import cartesian_to_spherical
from .base import Detection, Sensor, class_posterior, confidence_from_snr


class RadarSensor(Sensor):
    name = "radar"

    def __init__(self, cfg: RadarCfg, rng: np.random.Generator,
                 noise_scale: float = 1.0, clutter_scale: float = 1.0):
        super().__init__(cfg, rng)
        self.noise_scale = float(noise_scale)
        self.clutter_scale = float(clutter_scale)

    # ------------------------------------------------------------------
    def snr_db(self, range_m: float, rcs_m2: float) -> float:
        c = self.cfg
        return (c.snr0_db
                + 10.0 * np.log10(max(rcs_m2, 1e-4) / c.rcs_ref_m2)
                - 40.0 * np.log10(max(range_m, 1.0) / (c.ref_range_km * 1000.0)))

    def pd(self, snr_db: float) -> float:
        c = self.cfg
        return float(1.0 / (1.0 + np.exp(-(snr_db - c.pd_mid_snr_db) / c.pd_slope_db)))

    def range_sigma(self, range_m: float) -> float:
        c = self.cfg
        return (c.range_sigma_base_m + c.range_sigma_per_km2_m * (range_m / 1000.0) ** 2) \
            * self.noise_scale

    def ang_sigma(self, range_m: float) -> float:
        c = self.cfg
        return (c.ang_sigma_base_deg + c.ang_sigma_per_km_deg * (range_m / 1000.0)) \
            * self.noise_scale

    # ------------------------------------------------------------------
    def poll_multi(self, t_now, positions, metas) -> list[Detection]:
        """Multi-target poll with per-scan Poisson clutter.

        Bug fix (kept from v0.1.1): clutter is a Poisson process PER SCAN
        (birds/ground/multipath enter the beam whether or not any target is
        detected or even in range) - once per consumed scan, independent of
        the number of targets, so the false-contact rate per scan is
        unchanged when the fleet grows.
        """
        t_prev_next = self._next_t
        dets = super().poll_multi(t_now, positions, metas)
        # clutter is drawn once per consumed scan so the Poisson rate stays
        # per-scan regardless of how many scans fired since the last poll.
        n_scans = max(0, int(round((self._next_t - t_prev_next) * self.cfg.rate_hz)))
        for k in range(n_scans):
            dets.extend(self._clutter(t_prev_next + k / self.cfg.rate_hz))
        return dets

    def _observe(self, t_meas: float, drone_pos: np.ndarray, meta: dict) -> Detection | None:
        c = self.cfg
        site = np.asarray(meta["site_enu"], dtype=float)
        rel = drone_pos - site
        az, el, rng_m = cartesian_to_spherical(rel)
        if rng_m > c.max_range_m:
            return None  # no target det; clutter handled per-scan in poll()

        rcs = RCS_M2.get(meta.get("true_class", "quad"), 0.05)
        snr = self.snr_db(rng_m, rcs)
        if self.rng.random() >= self.pd(snr):
            return None  # missed detection - the tracker must bridge the gap

        sa = self.ang_sigma(rng_m)
        sr = self.range_sigma(rng_m)
        az_n = (az + self.rng.normal(0.0, sa)) % 360.0
        el_n = float(np.clip(el + self.rng.normal(0.0, sa), -5.0, 89.0))
        r_n = float(max(rng_m + self.rng.normal(0.0, sr), 10.0))
        conf = confidence_from_snr(snr)
        cls = class_posterior(self.rng, meta.get("true_class", "quad"), conf,
                              confuse_with="bird")
        return Detection(sensor=self.name, t_meas=t_meas - c.latency_s,
                         az_deg=az_n, el_deg=el_n, range_m=r_n, cls=cls,
                         snr_db=float(snr))

    # ------------------------------------------------------------------
    def _clutter(self, t_meas: float) -> list[Detection]:
        """Poisson false contacts (evaluated every poll even on target miss)."""
        c = self.cfg
        n = self.rng.poisson(c.clutter_rate_per_scan * self.clutter_scale)
        out = []
        for _ in range(n):
            az = self.rng.uniform(0.0, 360.0)
            el = self.rng.uniform(c.clutter_el_min_deg, c.clutter_el_max_deg)
            r = float(self.rng.uniform(200.0, c.max_range_m))
            sa = self.ang_sigma(r)
            sr = self.range_sigma(r)
            az_n = (az + self.rng.normal(0.0, sa)) % 360.0
            el_n = float(np.clip(el + self.rng.normal(0.0, sa), -5.0, 89.0))
            r_n = float(max(r + self.rng.normal(0.0, sr), 10.0))
            true_cl = "bird" if self.rng.random() < 0.8 else "unknown"
            cls = class_posterior(self.rng, true_cl, self.rng.uniform(0.5, 0.85),
                                  confuse_with="quad")
            out.append(Detection(sensor=self.name, t_meas=t_meas - c.latency_s,
                                 az_deg=az_n, el_deg=el_n, range_m=r_n, cls=cls,
                                 from_clutter=True,
                                 snr_db=float(self.rng.uniform(8.0, 14.0))))
        return out
