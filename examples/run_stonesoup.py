#!/usr/bin/env python3
"""Run a Stone Soup tracker on SkyGym detections.

Bridges SkyGym (Gymnasium recon gym) to Stone Soup (Dstl tracking library):

    SkyGymEnv.step()  ->  corrupted detections (radar / EO / RF)
                      ->  Stone Soup Detections (per-det measurement models)
                      ->  EKF + GNN2D MultiTargetTracker
                      ->  graded against the hidden witness GT

Two modes:
    --sensors radar   radar az/el/range only
    --sensors fusion  radar initiates; EO (az/el, 0.08 deg) and RF (az-only,
                      4 deg) measurements keep the track alive between and
                      beyond radar looks

Examples:
    python examples/run_stonesoup.py --sensors radar  --start-km 1.2
    python examples/run_stonesoup.py --sensors fusion --start-km 1.2
    python examples/run_stonesoup.py --sensors fusion --start-km 4.0 --noise 2.0 --clutter 2.5

Requires: pip install stonesoup gymnasium   (Python >= 3.9)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from skygym.config import EnvCfg, ScenarioCfg
from skygym.env import SkyGymEnv
from skygym.world import cartesian_to_spherical, ang_diff_deg
from skygym.sensors.radar import RadarSensor

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

T0 = datetime(2026, 9, 2, tzinfo=timezone.utc)


# ------------------------------------------------------------------ #
# Convention self-checks: SkyGym az = atan2(E, N), el = atan2(U, hypot)
# Stone Soup state layout is INTERLEAVED: [E, vE, N, vN, U, vU]
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
    """Return (ebr_map, eb_map, brg_map) after numeric verification."""
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
    """Initiate tracks only from range-capable (radar) detections.

    Bearing-only measurements (EO / RF) cannot be inverted to a position,
    so they update existing tracks but never spawn new ones.
    """

    def initiate(self, detections, timestamp, **kwargs):
        radar_dets = {d for d in detections
                      if isinstance(d.measurement_model,
                                    CartesianToElevationBearingRange)}
        return super().initiate(radar_dets, timestamp, **kwargs)


# ------------------------------------------------------------------ #
def run_episode(args):
    r = args.start_km * 1000.0
    sc = ScenarioCfg(
        name=args.scenario, duration_s=args.duration, seed=args.seed,
        start_min=(-1.08 * r, 0.75 * r, 60.0),
        start_max=(-0.75 * r, 1.08 * r, 140.0),
        tx_on=True, noise_scale=args.noise, clutter_scale=args.clutter)
    env = SkyGymEnv(EnvCfg())
    obs, info = env.reset(seed=args.seed, options={"scenario_cfg": sc})
    frames = [{"t": 0.0, "gt": info["gt"], "obs": obs}]
    for _ in range(int(args.duration / env.cfg.dt)):
        obs, _, term, trunc, info = env.step(None)
        frames.append({"t": info["gt"]["t"], "gt": info["gt"], "obs": obs})
        if term or trunc:
            break
    return frames, env


def build_detections(frames, env, mode, mappings):
    ebr_map, eb_map, brg_map = mappings
    radar = RadarSensor(env.cfg.rig.radar, np.random.default_rng(0))
    radar.noise_scale = frames[0]["gt"].get("noise_scale", 1.0)
    rig = env.cfg.rig
    per_sensor = {"radar": 0, "eo": 0, "rf": 0}
    ss_dets = []

    for f in frames:
        obs = f["obs"]
        for sensor, keep in (("radar", True), ("eo", mode == "fusion"),
                             ("rf", mode == "fusion")):
            if not keep:
                continue
            n = obs[sensor]["n"]
            for row in obs[sensor]["dets"][:n]:
                az, el, rng_m = row[0], row[1], row[2]
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
                    sa = math.radians(rig.eo.ang_sigma_deg
                                      * f["gt"].get("noise_scale", 1.0))
                    model = CartesianToElevationBearing(
                        ndim_state=6, mapping=eb_map,
                        noise_covar=np.diag([sa ** 2, sa ** 2]))
                    sv = np.array([math.radians(el), math.radians(az)])
                else:  # rf: azimuth only (no elevation from one station)
                    sa = math.radians(rig.rf.az_sigma_deg
                                      * f["gt"].get("noise_scale", 1.0))
                    model = Cartesian2DToBearing(
                        ndim_state=6, mapping=brg_map,
                        noise_covar=np.diag([sa ** 2]))
                    sv = np.array([math.radians(az)])
                ss_dets.append((t_meas, SSDetection(sv, timestamp=ts,
                                                    measurement_model=model)))
                per_sensor[sensor] += 1

    ss_dets.sort(key=lambda x: x[0])
    times = sorted({round(t, 6) for t, _ in ss_dets})
    detector = iter([
        (T0 + timedelta(seconds=t),
         {d for tt, d in ss_dets if abs(tt - t) < 1e-6}) for t in times])
    return detector, per_sensor, ss_dets


# ------------------------------------------------------------------ #
def run_tracker(args, mode):
    frames, env = run_episode(args)
    mappings = verified_mappings()
    detector, per_sensor, ss_dets = build_detections(frames, env, mode, mappings)

    transition = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.5), ConstantVelocity(0.5), ConstantVelocity(0.3)])
    predictor = KalmanPredictor(transition)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(),
                                        missed_distance=9.0)
    associator = GNNWith2DAssignment(hypothesiser)
    deleter = UpdateTimeDeleter(time_since_update=timedelta(seconds=1.0))
    prior = GaussianState(np.zeros((6, 1)),
                          np.diag([100.0, 25.0, 100.0, 25.0, 100.0, 25.0]),
                          timestamp=T0)
    initiator = RadarOnlyInitiator(prior, measurement_model=None)
    tracker = MultiTargetTracker(
        initiator=initiator, deleter=deleter, detector=detector,
        data_associator=associator, updater=updater)

    gt_t = np.array([f["t"] for f in frames])
    gt_pos = np.array([f["gt"]["pos"] for f in frames])

    def gt_at(t):
        i = int(np.searchsorted(gt_t, t))
        i = int(np.clip(i, 1, len(gt_t) - 1))
        w = (t - gt_t[i - 1]) / (gt_t[i] - gt_t[i - 1])
        return gt_pos[i - 1] * (1 - w) + gt_pos[i] * w

    site = np.asarray(env.cfg.rig.site_enu, dtype=float)
    track_rows, track_num, first_track_t = [], {}, None
    for time, tracks in tracker:
        t_rel = (time - T0).total_seconds()
        if first_track_t is None and tracks:
            first_track_t = t_rel
        best = None
        for tr in tracks:
            sv = np.asarray(tr.state_vector).flatten()
            tid = track_num.setdefault(str(tr.id), len(track_num) + 1)
            pos, vel = sv[[0, 2, 4]], sv[[1, 3, 5]]
            err = float(np.linalg.norm(pos - gt_at(t_rel)))
            if best is None or err < best[3]:
                best = (tid, pos, vel, err)
        if best is not None:
            tid, pos, vel, err = best
            az_e, el_e, r_e = cartesian_to_spherical(pos - site)
            az_g, el_g, r_g = cartesian_to_spherical(gt_at(t_rel) - site)
            track_rows.append([round(t_rel, 3), tid, *pos.tolist(), *vel.tolist(),
                               err, ang_diff_deg(az_e, az_g), el_e - el_g,
                               r_e - r_g])

    n_updates = {str(tr.id): len(tr.states) for tr in getattr(tracker, "tracks", set())}
    n_tracks = len(track_num)
    # confirmation filter: clutter correctly spawns 1-2 scan candidate tracks
    # under a single-point initiator; count only >=5-update tracks as confirmed
    confirmed = {tid for tid, k in n_updates.items() if k >= 5}
    summary = {
        "mode": mode, "scenario": args.scenario, "seed": args.seed,
        "duration_s": args.duration, "start_km": args.start_km,
        "noise_scale": args.noise, "clutter_scale": args.clutter,
        "dets_fed": per_sensor,
        "first_track_at_s": round(first_track_t, 3) if first_track_t else None,
        "n_track_candidates": n_tracks,
        "n_confirmed_tracks": len(confirmed),
        "n_false_confirmed_tracks": max(0, len(confirmed) - 1),
        "max_track_updates": max(n_updates.values()) if n_updates else 0,
    }
    if track_rows:
        ids, counts = np.unique([r[1] for r in track_rows], return_counts=True)
        tgt_id = int(ids[np.argmax(counts)])
        tgt = [r for r in track_rows if r[1] == tgt_id]
        errs = np.array([r[8] for r in tgt])
        summary.update({
            "pct_of_episode_tracked": round(
                100.0 * len({round(r[0], 1) for r in tgt}) / len(frames), 1),
            "position_rmse_m": round(float(np.sqrt(np.mean(errs ** 2))), 2),
            "mean_abs_az_err_deg": round(float(np.mean([abs(r[9]) for r in tgt])), 3),
            "mean_abs_el_err_deg": round(float(np.mean([abs(r[10]) for r in tgt])), 3),
            "mean_abs_range_err_m": round(float(np.mean([abs(r[11]) for r in tgt])), 2),
        })
        # steady state: skip first 2 s (velocity spin-up transient)
        late = [r[8] for r in tgt if r[0] >= 2.0]
        if late:
            summary["steady_state_mean_err_m"] = round(float(np.mean(late)), 2)
            summary["steady_state_max_err_m"] = round(float(np.max(late)), 2)
    else:
        summary["pct_of_episode_tracked"] = 0.0
    return summary, track_rows, len(frames)


# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensors", choices=["radar", "fusion"], default="fusion")
    p.add_argument("--scenario", default="approach")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--start-km", type=float, default=1.2)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--clutter", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--out", default="results/stonesoup")
    args = p.parse_args()

    summary, rows, n_frames = run_tracker(args, args.sensors)
    tag = f"{args.scenario}_d{args.duration:.0f}s_r{args.start_km:.1f}_n{args.noise:g}_c{args.clutter:g}_{args.sensors}"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, f"{tag}_tracks.csv"), "w") as f:
        f.write("t_s,track_id,est_e_m,est_n_m,est_u_m,est_vel_e,est_vel_n,"
                "est_vel_u,pos_err_m,az_err_deg,el_err_deg,range_err_m\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                             for v in r) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/{tag}_summary.json and {tag}_tracks.csv "
          f"({len(rows)} rows / {n_frames} frames)")


if __name__ == "__main__":
    main()
