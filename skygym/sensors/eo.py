"""EO/IR camera channel: pixel bearing (tight az/el) + degraded range/ID.

Physics baked in:
- Detection requires the target to subtend enough pixels (Pd vs pixel size).
- Bearing accuracy is pixel-quantised -> far tighter than radar.
- Classification confidence collapses as the target shrinks to a blob.
- Range only via (noisy) stereo model, or none for mono.
"""
from __future__ import annotations

import numpy as np

from ..config import EOCfg, DRONE_SIZE_M
from ..world import cartesian_to_spherical, ang_diff_deg
from .base import Detection, Sensor, class_posterior


class EOSensor(Sensor):
    name = "eo"

    def _observe(self, t_meas: float, drone_pos: np.ndarray, meta: dict) -> Detection | None:
        c = self.cfg
        site = np.asarray(meta["site_enu"], dtype=float)
        rel = drone_pos - site
        az, el, rng_m = cartesian_to_spherical(rel)
        if rng_m > c.max_range_m:
            return None

        # pixel footprint: target diameter -> px using horizontal FOV
        size = DRONE_SIZE_M.get(meta.get("true_class", "quad"), 0.5)
        px_per_rad = c.res[0] / (2.0 * np.tan(np.radians(c.fov_deg[0]) / 2.0))
        px = float(size / max(rng_m, 1.0) * px_per_rad)

        if not c.slaved:
            # fixed boresight (North, 10 deg up): target must be inside the FOV
            off_az = abs(ang_diff_deg(az, 0.0))
            off_el = abs(el - 10.0)
            if off_az > c.fov_deg[0] / 2.0 or off_el > c.fov_deg[1] / 2.0:
                return None

        # detection probability vs angular size
        pd = 1.0 / (1.0 + np.exp(-(px - c.p50_pixel) / c.pix_slope))
        if self.rng.random() >= pd:
            return None

        az_n = (az + self.rng.normal(0.0, c.ang_sigma_deg * meta.get("noise_scale", 1.0))) % 360.0
        el_n = float(np.clip(
            el + self.rng.normal(0.0, c.ang_sigma_deg * meta.get("noise_scale", 1.0)),
            -5.0, 89.0))

        if c.range_mode == "stereo":
            r_km2 = (rng_m / 1000.0) ** 2
            sig_r = rng_m * (c.range_sigma_base + c.range_sigma_per_km2 * r_km2) \
                * meta.get("noise_scale", 1.0)
            r_n = float(max(rng_m + self.rng.normal(0.0, sig_r), 10.0))
        else:
            r_n = float("nan")

        # image classification confidence collapses with pixel size
        conf = float(np.clip(0.35 + 0.6 / (1.0 + np.exp(-(px - 6.0) / 2.0)), 0.3, 0.95))
        confuse = "bird" if meta.get("true_class", "quad") == "bird" else "bird"
        cls = class_posterior(self.rng, meta.get("true_class", "quad"), conf,
                              confuse_with=confuse)
        return Detection(sensor=self.name, t_meas=t_meas - c.latency_s,
                         az_deg=az_n, el_deg=el_n, range_m=r_n, cls=cls,
                         pixel_size_px=px)
