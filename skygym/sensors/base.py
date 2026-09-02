"""Sensor base: Detection record, threat-ID confusion, rate gating."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CLASSES


@dataclass
class Detection:
    """One reported contact. az/el/range are the sensor's corrupted view.

    range_m / el_deg may be NaN for sensors that do not measure them
    (RF has no elevation and no range; EO mono has no range).
    `cls` is a posterior over CLASSES - NEVER a hard ground-truth label.
    """
    sensor: str
    t_meas: float              # measurement timestamp (env time - latency)
    az_deg: float
    el_deg: float              # NaN if not measured
    range_m: float             # NaN if not measured
    cls: dict[str, float]
    from_clutter: bool = False
    snr_db: float = float("nan")
    pixel_size_px: float = float("nan")

    def as_row(self) -> list[float]:
        """Flat row for padded observation arrays."""
        p = [self.cls.get(c, 0.0) for c in CLASSES]
        return [
            self.az_deg,
            self.el_deg,
            self.range_m,
            float(self.from_clutter),   # NOTE: consumer-visible clutter flag;
                                        # real trackers must NOT rely on it.
            self.snr_db,
            self.pixel_size_px,
            self.t_meas,
            *p,
        ]


DET_ROW_LEN = 11  # az, el, r, clutter, snr, px, t_meas, 4 class probs


def class_posterior(
    rng: np.random.Generator,
    true_class: str,
    confidence: float,
    confuse_with: str = "bird",
) -> dict[str, float]:
    """Sample a class posterior from (true class, confidence).

    confidence in [0,1]: 1 -> all mass on the true class; 0 -> diffuse mass
    biased toward `confuse_with` (the classic drone/bird confusion).
    This is the ONLY place threat ID is generated - always corrupted.
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    p = {c: 0.0 for c in CLASSES}
    # diffuse part: mostly the confuser, remainder spread on others
    others = [c for c in CLASSES if c != true_class]
    w = np.array([1.0 if c == confuse_with else 0.35 for c in others])
    w = w / w.sum()
    for c, wgt in zip(others, w):
        p[c] += (1.0 - confidence) * wgt
    p[true_class] += confidence
    # Dirichlet jitter so posteriors are not degenerate repeats
    keys = list(p.keys())
    vals = np.array([p[k] for k in keys])
    vals = rng.dirichlet(np.maximum(vals, 1e-3) * 60.0)
    p = {k: float(v) for k, v in zip(keys, vals)}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


class Sensor:
    """Rate-gated base class. Subclasses implement _observe()."""

    name: str = "base"

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._next_t = 0.0

    def reset(self) -> None:
        self._next_t = 0.0

    def poll(self, t_now: float, drone_pos: np.ndarray, drone_meta: dict) -> list[Detection]:
        """Fire at the sensor's own rate; return detections for this step."""
        rate = self.cfg.rate_hz
        dets: list[Detection] = []
        while t_now + 1e-9 >= self._next_t:
            d = self._observe(self._next_t, drone_pos, drone_meta)
            if d is not None:
                dets.append(d)
            self._next_t += 1.0 / rate
        return dets

    def _observe(self, t_meas: float, drone_pos: np.ndarray, meta: dict):
        raise NotImplementedError


def confidence_from_snr(snr_db: float, mid_db: float = 16.0, width_db: float = 6.0,
                        c_min: float = 0.35, c_max: float = 0.93) -> float:
    """Map SNR to classification confidence (logistic)."""
    x = (snr_db - mid_db) / width_db
    sig = 1.0 / (1.0 + np.exp(-x))
    return float(c_min + (c_max - c_min) * sig)
