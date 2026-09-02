"""RF direction-finding channel: az-only bearing + protocol fingerprint ID.

Physics baked in:
- Only hears the drone when its control link is transmitting (tx_on).
- Bearing-only (sigma ~ 4 deg); NO elevation, NO range from one station.
- Best threat-ID of the suite: protocol fingerprinting, near-clean when on.
"""
from __future__ import annotations

import numpy as np

from ..config import RFCfg
from ..world import cartesian_to_spherical, ang_diff_deg
from .base import Detection, Sensor, class_posterior


class RFSensor(Sensor):
    name = "rf"

    def _observe(self, t_meas: float, drone_pos: np.ndarray, meta: dict) -> Detection | None:
        c = self.cfg
        if not meta.get("tx_on", True):
            return None  # silent drone: RF is deaf

        site = np.asarray(meta["site_enu"], dtype=float)
        rel = drone_pos - site
        az, _el, rng_m = cartesian_to_spherical(rel)
        if rng_m > c.max_range_m:
            return None

        if self.rng.random() >= c.pd_tx:
            return None

        sig = c.az_sigma_deg * meta.get("noise_scale", 1.0)
        az_n = (az + self.rng.normal(0.0, sig)) % 360.0

        # protocol fingerprint: drone classes are near-certain, bird ~ excluded
        true_class = meta.get("true_class", "quad")
        if true_class == "bird":
            return None  # birds do not emit control links
        cls = class_posterior(self.rng, true_class, 0.95, confuse_with="unknown")
        return Detection(sensor=self.name, t_meas=t_meas - c.latency_s,
                         az_deg=az_n, el_deg=float("nan"), range_m=float("nan"),
                         cls=cls)
