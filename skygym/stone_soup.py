"""Stone Soup tracking bridge — SkyGym's standard tracking baseline.

Since v0.2.0 the reference consumer of SkyGym detections is Stone Soup
(Dstl's open-source tracking library). This module is the single bridge:

    SkyGymEnv frames (obs + hidden gt)
      -> per-detection Stone Soup measurement models
           radar  CartesianToElevationBearingRange (az, el, range)
           EO     CartesianToElevationBearingRange (az, el, stereo range)
                  or CartesianToElevationBearing   (az, el) when range is NaN
           RF     Cartesian2DToBearing              (az only)
      -> EKF (constant-velocity) + GNN2D association + 1 s delete
      -> tracks graded against the witness ground truth

Conventions (SkyGym compass az = atan2(E, N) vs Stone Soup bearing) are
verified numerically at every run via verified_mappings() — a wrong axis
mapping silently produces nonsense tracks, so it is never trusted.

Bearing-only sensors (RF always, EO without range) can update tracks but
never initiate them: a bearing cannot be inverted to a position.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    from stonesoup.types.detection import Detection as SSDetection
    from stonesoup.types.state import GaussianState, State
    from stonesoup.models.measurement.nonlinear import (
        CartesianToElevationBearingRange, CartesianToElevationBearing,
        Cartesian2DToBearing)
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel, ConstantVelocity)
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.updater.kalman import ExtendedKalmanUpdater
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.measures import Mahalanobis
    from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
    from stonesoup.deleter.time import UpdateTimeDeleter
    from stonesoup.initiator.simple import SinglePointMeasurementInitiator
    from stonesoup.tracker.simple import MultiTargetTracker
    STONESOUP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    STONESOUP_AVAILABLE = False

from .config import CLASSES
from .sensors.radar import RadarSensor
from .world import cartesian_to_spherical, ang_diff_deg, spherical_to_cartesian

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
MISSING_STONESOUP_MSG = (
    "Stone Soup is required for the tracking baseline: "
    "pip install stonesoup  (or pip install -e '.[tracking]')")


# ------------------------------------------------------------------ #
# Convention self-checks
# ------------------------------------------------------------------ #
def _check(model, st, expect, label):
    v = np.asarray(model.function(State(st), noise=False)).flatten()
    ok = all(abs(a - b) < 1e-6 for a, b in zip(v, expect))
    if not ok:
        raise RuntimeError(
            f"measurement-convention check FAILED for {label}: got {v}, "
            f"expected {expect}")
    return True


def verified_mappings():
    """Return (ebr_map, eb_map, brg_map) after numeric verification.

    Stone Soup state layout is INTERLEAVED: [E, vE, N, vN, U, vU], so the
    position indices are [2, 0, 4] for (N, E, U) — matching SkyGym's
    az = atan2(E, N), el = atan2(U, hypot(E, N)).
    """
    if not STONESOUP_AVAILABLE:
        raise ImportError(MISSING_STONESOUP_MSG)
    E, N, U = 500.0, 866.0254, 100.0
    st = np.array([E, 0.0, N, 0.0, U, 0.0])
    az, el = math.atan2(E, N), math.atan2(U, math.hypot(E, N))
    r = math.sqrt(E * E + N * N + U * U)

    ebr = CartesianToElevationBearingRange(
        ndim_state=6, mapping=np.array([2, 0, 4]),
        noise_covar=np.diag([1e-6, 1e-6, 1.0]))
    _check(ebr, st, [el, az, r], "CartesianToElevationBearingRange[2,0,4]")

    eb = CartesianToElevationBearing(
        ndim_state=6, mapping=np.array([2, 0, 4]),
        noise_covar=np.diag([1e-6, 1e-6]))
    _check(eb, st, [el, az], "CartesianToElevationBearing[2,0]")

    brg = Cartesian2DToBearing(
        ndim_state=6, mapping=np.array([2, 0]),
        noise_covar=np.diag([1e-6]))
    _check(brg, st, [az], "Cartesian2DToBearing[2,0]")

    return np.array([2, 0, 4]), np.array([2, 0, 4]), np.array([2, 0])


class RadarOnlyInitiator(SinglePointMeasurementInitiator):
    """Initiate tracks only from range-capable radar detections.

    Bearing-only measurements (RF az-only, EO without range) cannot be
    inverted to a position, so they update existing tracks but never
    spawn new ones. Sensors are identified via detection metadata.
    """

    def initiate(self, detections, timestamp, **kwargs):
        radar_dets = {d for d in detections
                      if d.metadata.get("sensor") == "radar"}
        return super().initiate(radar_dets, timestamp, **kwargs)


# ------------------------------------------------------------------ #
# Detections: SkyGym obs rows -> Stone Soup
# ------------------------------------------------------------------ #
def build_detections(frames, env, mode="fusion", eo_with_range=True):
    """Convert recorded obs rows into a Stone Soup detector iterator.

    frames: list of {"t", "gt", "obs"} as produced by roll_frames().
    mode:   "radar" (radar only) or "fusion" (radar + EO + RF).
    eo_with_range: feed EO stereo range (EBR) when finite instead of
            bearing-only (EB). Free accuracy when the cfg is stereo.

    Returns (detector_iterator, per_sensor_counts, all_dets_sorted).
    """
    ebr_map, eb_map, brg_map = verified_mappings()
    rig = env.cfg.rig
    radar = RadarSensor(rig.radar, np.random.default_rng(0))
    radar.noise_scale = frames[0]["gt"].get("noise_scale", 1.0)
    per_sensor = {"radar": 0, "eo": 0, "rf": 0}
    ss_dets = []

    for f in frames:
        noise = f["gt"].get("noise_scale", 1.0)
        obs = f["obs"]
        for sensor, keep in (("radar", True), ("eo", mode == "fusion"),
                             ("rf", mode == "fusion")):
            if not keep:
                continue
            n = obs[sensor]["n"]
            for row in obs[sensor]["dets"][:n]:
                az, el, rng_m = float(row[0]), float(row[1]), float(row[2])
                t_meas = float(row[6])
                ts = T0 + timedelta(seconds=t_meas)
                if sensor == "radar":
                    sa = math.radians(radar.ang_sigma(rng_m))
                    sr = radar.range_sigma(rng_m)
                    model = CartesianToElevationBearingRange(
                        ndim_state=6, mapping=ebr_map,
                        noise_covar=np.diag([sa ** 2, sa ** 2, sr ** 2]))
                    sv = np.array([math.radians(el), math.radians(az), rng_m])
                elif sensor == "eo":
                    sa = math.radians(rig.eo.ang_sigma_deg * noise)
                    use_r = (eo_with_range and np.isfinite(rng_m)
                             and rng_m > 0)
                    if use_r:
                        sr = (rng_m * (rig.eo.range_sigma_base
                                       + rig.eo.range_sigma_per_km2
                                       * (rng_m / 1000.0) ** 2) * noise)
                        model = CartesianToElevationBearingRange(
                            ndim_state=6, mapping=ebr_map,
                            noise_covar=np.diag([sa ** 2, sa ** 2, sr ** 2]))
                        sv = np.array([math.radians(el), math.radians(az),
                                       rng_m])
                    else:
                        model = CartesianToElevationBearing(
                            ndim_state=6, mapping=eb_map,
                            noise_covar=np.diag([sa ** 2, sa ** 2]))
                        sv = np.array([math.radians(el), math.radians(az)])
                else:  # rf: azimuth only (one station -> no el, no range)
                    sa = math.radians(rig.rf.az_sigma_deg * noise)
                    model = Cartesian2DToBearing(
                        ndim_state=6, mapping=brg_map,
                        noise_covar=np.diag([sa ** 2]))
                    sv = np.array([math.radians(az)])
                ss_dets.append((t_meas, SSDetection(
                    sv, timestamp=ts, measurement_model=model,
                    metadata={"sensor": sensor})))
                per_sensor[sensor] += 1

    ss_dets.sort(key=lambda x: x[0])
    times = sorted({round(t, 6) for t, _ in ss_dets})
    detector = iter([
        (T0 + timedelta(seconds=t),
         {d for tt, d in ss_dets if abs(tt - t) < 1e-6}) for t in times])
    return detector, per_sensor, ss_dets


def build_tracker(gate: float = 9.0, delete_s: float = 1.0,
                  q=(0.5, 0.5, 0.3)):
    """EKF(CV 3D) + GNN2D + single-point initiator + update-time deleter."""
    if not STONESOUP_AVAILABLE:
        raise ImportError(MISSING_STONESOUP_MSG)
    transition = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(q[0]), ConstantVelocity(q[1]), ConstantVelocity(q[2])])
    predictor = KalmanPredictor(transition)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(),
                                        missed_distance=gate)
    associator = GNNWith2DAssignment(hypothesiser)
    deleter = UpdateTimeDeleter(time_since_update=timedelta(seconds=delete_s))
    prior = GaussianState(np.zeros((6, 1)),
                          np.diag([100.0, 25.0, 100.0, 25.0, 100.0, 25.0]),
                          timestamp=T0)
    initiator = RadarOnlyInitiator(prior, measurement_model=None)
    return MultiTargetTracker(
        initiator=initiator, deleter=deleter, detector=None,
        data_associator=associator, updater=updater)


# ------------------------------------------------------------------ #
# Episode plumbing + grading
# ------------------------------------------------------------------ #
def roll_frames(env, seed: int, options: dict | None = None,
                duration_s: float | None = None):
    """Roll one episode with the autopilot; return [{"t","gt","obs"}, ...]."""
    options = dict(options or {})
    if duration_s is None:
        duration_s = float(options.get("duration_s", 20.0))
    obs, info = env.reset(seed=seed, options=options)
    frames = [{"t": info["gt"]["t"], "gt": info["gt"], "obs": obs}]
    for _ in range(int(duration_s / env.cfg.dt)):
        obs, _, term, trunc, info = env.step(None)
        frames.append({"t": info["gt"]["t"], "gt": info["gt"], "obs": obs})
        if term or trunc:
            break
    return frames, env


def run_episode(env, seed: int, options: dict | None = None,
                mode: str = "fusion", eo_with_range: bool = True):
    """Full baseline: roll an episode, track, grade vs witness.

    Returns (summary_dict, track_rows, n_frames). Track rows:
    [t, track_id, e, n, u, ve, vn, vu, pos_err, az_err, el_err, range_err].
    """
    frames, env = roll_frames(env, seed, options)
    detector, per_sensor, _ = build_detections(frames, env, mode=mode,
                                               eo_with_range=eo_with_range)
    tracker = build_tracker()
    # detector must be set after construction (kept immutable in build_tracker)
    tracker.detector = detector

    gt_t = np.array([f["t"] for f in frames])
    gt_pos = np.array([f["gt"]["pos"] for f in frames])
    gt_vel = np.array([f["gt"]["vel"] for f in frames])

    def gt_at(t):
        i = int(np.searchsorted(gt_t, t))
        i = int(np.clip(i, 1, len(gt_t) - 1))
        w = (t - gt_t[i - 1]) / (gt_t[i] - gt_t[i - 1])
        return (gt_pos[i - 1] * (1 - w) + gt_pos[i] * w,
                gt_vel[i - 1] * (1 - w) + gt_vel[i] * w)

    site = np.asarray(env.cfg.rig.site_enu, dtype=float)
    true_cls = CLASSES.index(frames[0]["gt"]["true_class"])

    rows, first_track_t = [], None
    track_num = {}
    for time, tracks in tracker:
        t_rel = (time - T0).total_seconds()
        if first_track_t is None and tracks:
            first_track_t = t_rel
        best = None
        for tr in tracks:
            sv = np.asarray(tr.state_vector).flatten()
            tid = track_num.setdefault(str(tr.id), len(track_num) + 1)
            pos, vel = sv[[0, 2, 4]], sv[[1, 3, 5]]
            gp, gv = gt_at(t_rel)
            err = float(np.linalg.norm(pos - gp))
            if best is None or err < best[3]:
                best = (tid, pos, vel, err, gp, gv)
        if best is None:
            continue
        tid, pos, vel, err, gp, gv = best
        az_e, el_e, r_e = cartesian_to_spherical(pos - site)
        az_g, el_g, r_g = cartesian_to_spherical(gp - site)
        rows.append([round(t_rel, 3), tid, *pos.tolist(), *vel.tolist(),
                     err, ang_diff_deg(az_e, az_g), el_e - el_g, r_e - r_g])

    # ---- target track + summary (same conventions as results/stonesoup) --
    n_updates = {str(tr.id): len(tr.states)
                 for tr in getattr(tracker, "tracks", set())}
    confirmed = {tid for tid, k in n_updates.items() if k >= 5}
    summary = {
        "tracker": "stonesoup EKF(CV)+GNN2D+SinglePointInitiator",
        "mode": mode, "eo_with_range": eo_with_range,
        "seed": seed, "dets_fed": per_sensor,
        "first_track_at_s": (round(first_track_t, 3)
                             if first_track_t is not None else None),
        "n_track_candidates": len(track_num),
        "n_confirmed_tracks": len(confirmed),
        "n_false_confirmed_tracks": max(0, len(confirmed) - 1),
        "max_track_updates": max(n_updates.values()) if n_updates else 0,
    }
    if rows:
        ids, counts = np.unique([r[1] for r in rows], return_counts=True)
        tgt_id = int(ids[np.argmax(counts)])
        tgt = [r for r in rows if r[1] == tgt_id]
        errs = np.array([r[8] for r in tgt])
        verrs = []
        for r in tgt:
            _, gv = gt_at(r[0])
            verrs.append(float(np.linalg.norm(np.array(r[4:7]) - gv)))
        summary.update({
            "pct_of_episode_tracked": round(
                100.0 * len({round(r[0], 1) for r in tgt}) / len(frames), 1),
            "position_rmse_m": round(float(np.sqrt(np.mean(errs ** 2))), 2),
            "pos_p95_m": round(float(np.percentile(errs, 95)), 2),
            "vel_rmse_mps": round(float(np.sqrt(np.mean(np.square(verrs))))
                                  if verrs else 0.0, 2),
            "mean_abs_az_err_deg": round(
                float(np.mean([abs(r[9]) for r in tgt])), 3),
            "mean_abs_el_err_deg": round(
                float(np.mean([abs(r[10]) for r in tgt])), 3),
            "mean_abs_range_err_m": round(
                float(np.mean([abs(r[11]) for r in tgt])), 2),
        })
        late = [r[8] for r in tgt if r[0] >= 2.0]
        if late:
            summary["steady_state_mean_err_m"] = round(float(np.mean(late)), 2)
            summary["steady_state_max_err_m"] = round(float(np.max(late)), 2)
        summary["id_accuracy_on_target_track"] = _id_readout(
            tgt, frames, site, true_cls)
    else:
        summary["pct_of_episode_tracked"] = 0.0
    return summary, rows, len(frames)


def _id_readout(tgt_rows, frames, site, true_cls: int):
    """Det-level ID readout: class posterior of the radar det closest to the
    target track at each tracked tick (Stone Soup does no classification)."""
    ft = [f["t"] for f in frames]
    correct = []
    for r in tgt_rows:
        i = bisect_right(ft, r[0]) - 1
        i = int(np.clip(i, 0, len(ft) - 1))
        dets = frames[i]["obs"]["radar"]
        best_d, best_d2 = None, np.inf
        for row in dets["dets"][:dets["n"]]:
            if float(row[3]) > 0.5:      # skip clutter-flagged lies
                continue
            t_meas = float(row[6])
            if not np.isfinite(t_meas) or abs(r[0] - t_meas) > 0.25:
                continue
            from .world import spherical_to_cartesian as _s2c
            p = _s2c(float(row[0]), float(row[1]), float(row[2])) + site
            d2 = float(np.sum((p - np.array(r[2:5])) ** 2))
            if d2 < best_d2:
                best_d, best_d2 = row, d2
        if best_d is None:
            continue
        probs = [float(best_d[7 + k]) for k in range(4)]
        correct.append(int(np.argmax(probs)) == true_cls)
    return round(float(np.mean(correct)), 3) if correct else None
