"""Baseline consumer: EKF track-while-scan with GNN association.

This is the proof-of-honesty consumer for Stage 0: it eats the corrupted
detection streams, estimates the target state, and is scored against the
hidden witness. It also prototypes the recon->intercept handoff contract
(track state + covariance, never raw truth).

Design:
- CV model EKF (6D state: pos+vel), white-noise-acceleration process noise.
- Greedy GNN association with Mahalanobis gating (chi2).
- Confirmed tracks: full 3D position updates (radar/EO) AND az-only bearing
  pseudo-measurements (RF).
- Tentative tracks: initiated on any 3D measurement, promoted by M-of-N.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CLASSES

CHI2_GATE_3D = 16.27   # 99.9% for 3 DOF
CHI2_GATE_1D = 10.83   # 99.9% for 1 DOF (az-only)


def cv_model_F(dt: float) -> np.ndarray:
    F = np.eye(6)
    for i in range(3):
        F[i, i + 3] = dt
    return F


def cv_model_Q(dt: float, q: float = 2.0) -> np.ndarray:
    """Discrete white-noise-acceleration model PSD q (m^2/s^3)."""
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = q * dt**3 / 3.0
        Q[i, i + 3] = q * dt**2 / 2.0
        Q[i + 3, i] = q * dt**2 / 2.0
        Q[i + 3, i + 3] = q * dt
    return Q


@dataclass
class Track:
    tid: int
    x: np.ndarray                     # [pos(3), vel(3)]
    P: np.ndarray
    hits: int = 1
    misses: int = 0
    class_logodds: np.ndarray = field(default_factory=lambda: np.zeros(len(CLASSES)))
    last_t: float = 0.0
    initiated: bool = False
    history: list = field(default_factory=list)   # (t, pos, vel) after confirmation


class TrackWhileScan:
    """Single-hypothesis track-while-scan (1 real target + clutter)."""

    def __init__(self, q: float = 2.0, confirm_m: int = 2, confirm_n: int = 3,
                 delete_after: int = 6):
        self.q = q
        self.confirm_m = confirm_m
        self.confirm_n = confirm_n
        self.delete_after = delete_after
        self.tracks: list[Track] = []       # confirmed
        self._pending: list[Track] = []     # tentative
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def _predict(self, trk: Track, t: float) -> None:
        dt = t - trk.last_t
        if dt <= 0:
            return
        F = cv_model_F(dt)
        trk.x = F @ trk.x
        trk.P = F @ trk.P @ F.T + cv_model_Q(dt, self.q)
        trk.last_t = t

    def _gate_md(self, trk: Track, m: dict) -> float | None:
        """Mahalanobis distance of measurement m to track (None = incompatible)."""
        if "pos" in m:
            H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
            y = m["pos"] - H @ trk.x
            S = H @ trk.P @ H.T + m["R"]
            return float(y.T @ np.linalg.solve(S + 1e-9 * np.eye(3), y))
        elif "az_unit" in m:
            p = trk.x[:3]
            horiz = p[0] ** 2 + p[1] ** 2
            if horiz < 1.0:
                return None
            az_pred = np.arctan2(p[0], p[1])
            az_meas = np.arctan2(m["az_unit"][0], m["az_unit"][1])
            y = float((az_meas - az_pred + np.pi) % (2 * np.pi) - np.pi)
            # H = [daz/dE, daz/dN, 0, 0, 0, 0] with az = atan2(E, N)
            H = np.zeros((1, 6))
            H[0, 0] = p[1] / horiz
            H[0, 1] = -p[0] / horiz
            S = float(H @ trk.P @ H.T) + m["R_az"]
            return y * y / S
        return None

    def _apply_update(self, trk: Track, m: dict) -> None:
        if "pos" in m:
            H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
            y = m["pos"] - H @ trk.x
            S = H @ trk.P @ H.T + m["R"]
            K = trk.P @ H.T @ np.linalg.solve(S + 1e-9 * np.eye(3), np.eye(3))
            trk.x = trk.x + K @ y
            I_KH = np.eye(6) - K @ H
            trk.P = I_KH @ trk.P @ I_KH.T + K @ m["R"] @ K.T
        else:  # az-only
            p = trk.x[:3]
            horiz = p[0] ** 2 + p[1] ** 2
            if horiz < 1.0:
                return
            az_pred = np.arctan2(p[0], p[1])
            az_meas = np.arctan2(m["az_unit"][0], m["az_unit"][1])
            y = np.array([float((az_meas - az_pred + np.pi) % (2 * np.pi) - np.pi)])
            H = np.zeros((1, 6))
            H[0, 0] = p[1] / horiz
            H[0, 1] = -p[0] / horiz
            S = float(H @ trk.P @ H.T) + m["R_az"]
            K = (trk.P @ H.T) / S
            trk.x = trk.x + (K @ y).ravel()
            I_KH = np.eye(6) - K @ H
            trk.P = I_KH @ trk.P @ I_KH.T + K * m["R_az"] @ K.T

    # ------------------------------------------------------------------ #
    def process_tick(self, t: float, measurements: list[dict]) -> list[Track]:
        """One scan. measurements: [{'pos': xyz, 'R': 3x3, 'cls': post} or
        {'az_unit': (E,N) unit, 'R_az': var, 'cls': post}] at common time t."""
        everything = self.tracks + self._pending
        for trk in everything:
            self._predict(trk, t)

        used = set()
        # confirmed tracks get first pick (they win associations)
        for trk in everything:
            best, best_md = None, np.inf
            for i, m in enumerate(measurements):
                if i in used:
                    continue
                md = self._gate_md(trk, m)
                if md is not None and md < best_md:
                    best, best_md = i, md
            gate = CHI2_GATE_3D if (best is not None and "pos" in measurements[best]) \
                else CHI2_GATE_1D
            if best is not None and best_md < gate:
                m = measurements[best]
                used.add(best)
                self._apply_update(trk, m)
                trk.hits += 1
                trk.misses = 0
                if m.get("cls"):
                    trk.class_logodds += np.log(np.clip(
                        [m["cls"].get(c, 1e-6) for c in CLASSES], 1e-6, None))
                if trk.initiated:
                    trk.history.append((t, trk.x[:3].copy(), trk.x[3:].copy()))
            else:
                trk.misses += 1

        # spawn tentative tracks on unclaimed 3D measurements
        for i, m in enumerate(measurements):
            if i in used or "pos" not in m:
                continue
            p0 = np.concatenate([m["pos"], np.zeros(3)])
            P0 = np.zeros((6, 6))
            P0[:3, :3] = m["R"]
            P0[3:, 3:] = np.eye(3) * 100.0   # unknown initial velocity
            self._pending.append(Track(tid=self._next_id, x=p0, P=P0, last_t=t))
            self._next_id += 1

        # promotion (M-of-N: 2 hits within ~3 scans) and cleanup
        promoted = []
        still_pending = []
        for pt in self._pending:
            if pt.hits >= self.confirm_m:
                pt.initiated = True
                promoted.append(pt)
            elif (pt.hits + pt.misses) > self.confirm_n + 2:
                continue  # died before confirmation
            else:
                still_pending.append(pt)
        self._pending = still_pending
        self.tracks.extend(promoted)
        self.tracks = [trk for trk in self.tracks if trk.misses <= self.delete_after]
        return self.tracks

    def confirmed_track(self) -> Track | None:
        if self.tracks:
            return max(self.tracks, key=lambda tr: tr.hits)
        return None
