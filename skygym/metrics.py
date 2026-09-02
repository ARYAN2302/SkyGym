"""Evaluate track-while-scan output against the hidden witness ground truth."""
from __future__ import annotations

import numpy as np

from .tracker import TrackWhileScan
from . import world


def _dets_from_obs(obs: dict, site_enu: np.ndarray, t_now: float,
                   max_age_s: float = 0.25) -> list[dict]:
    """Convert padded obs arrays -> tracker measurements (cartesian + R)."""
    meas = []
    for sensor, d in obs.items():
        n = int(d["n"])
        if n == 0:
            continue
        arr = d["dets"][:n]
        for row in arr:
            az, el, r = float(row[0]), float(row[1]), float(row[2])
            t_meas = float(row[6])
            if not np.isfinite(t_meas) or (t_now - t_meas) > max_age_s:
                continue
            if sensor == "rf":
                meas.append({"az_deg": az, "R_az": np.radians(4.0) ** 2,
                             "cls": _cls_from_row(row)})
                continue
            if not np.isfinite(el) or not np.isfinite(r) or r <= 0:
                continue
            pos = world.spherical_to_cartesian(az, el, r)
            if sensor == "radar":
                # sigma model mirrors RadarCfg at measured range
                r_km = r / 1000.0
                sig_r = 5.0 + 3.0 * r_km**2
                sig_a = np.radians(0.6 + 0.5 * r_km)
                R = world.spherical_cov_to_cartesian(az, el, r,
                                                     0.6 + 0.5 * r_km,
                                                     0.6 + 0.5 * r_km, sig_r)
                _ = sig_a
            else:  # eo
                sig_r = (r * 0.03 + r * 0.05 * (r / 1000.0) ** 2) if np.isfinite(r) else 1e6
                R = world.spherical_cov_to_cartesian(az, el, r, 0.08, 0.08, sig_r)
            meas.append({"pos": pos, "R": R, "cls": _cls_from_row(row)})
    return meas


def _cls_from_row(row: np.ndarray) -> dict[str, float]:
    return {c: float(row[7 + i]) for i, c in enumerate(
        ("quad", "fixed_wing", "bird", "unknown"))}


def run_tracker_on_episode(env, seed: int, options: dict | None = None,
                           q: float = 2.0) -> dict:
    """Roll one episode, feed detections to the EKF, score vs witness GT."""
    obs, info = env.reset(seed=seed, options=options)
    tws = TrackWhileScan(q=q)
    gt_series = []          # (t, pos, vel, true_class)
    est_series = []         # (t, pos_est, vel_est, class_probs)
    init_t = None
    done = False
    while not done:
        t = info["gt"]["t"]
        gt_series.append((t, info["gt"]["pos"].copy(), info["gt"]["vel"].copy(),
                          info["gt"]["true_class"]))
        meas = _dets_from_obs(obs, np.asarray(env.cfg.rig.site_enu), t)
        tws.process_tick(t, meas)
        trk = tws.confirmed_track()
        if trk is not None and trk.initiated:
            if init_t is None:
                init_t = t
            est_series.append((t, trk.x[:3].copy(), trk.x[3:].copy(),
                               trk.class_logodds.copy()))
        obs, _, term, trunc, info = env.step(None)
        done = term or trunc

    # --- metrics --------------------------------------------------------
    res = {"n_steps": len(gt_series), "n_est": len(est_series),
           "track_initiated": init_t is not None}
    if init_t is not None:
        res["init_latency_s"] = float(init_t - gt_series[0][0])
    if est_series:
        gt_t = {round(t, 3): (p, v) for t, p, v, _ in gt_series}
        p_err, v_err, cls_correct, n_matched = [], [], 0, 0
        for t, pe, ve, logodds in est_series:
            key = round(t, 3)
            if key not in gt_t:
                continue
            gp, gv = gt_t[key]
            p_err.append(np.linalg.norm(pe - gp))
            v_err.append(np.linalg.norm(ve - gv))
            n_matched += 1
            if len(logodds) and int(np.argmax(logodds)) == _cls_index(gt_series[0][3]):
                cls_correct += 1
        res["pos_rmse_m"] = float(np.sqrt(np.mean(np.square(p_err)))) if p_err else None
        res["vel_rmse_mps"] = float(np.sqrt(np.mean(np.square(v_err)))) if v_err else None
        res["pos_p95_m"] = float(np.percentile(p_err, 95)) if p_err else None
        res["id_accuracy"] = float(cls_correct / n_matched) if n_matched else None
        res["track_continuity"] = float(len(est_series) / max(1, len(gt_series) - (init_t - gt_series[0][0]) / env.cfg.dt if init_t else 1))
    return res


def _cls_index(name: str) -> int:
    return ("quad", "fixed_wing", "bird", "unknown").index(name)


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-episode metric dicts into a summary."""
    agg = {"episodes": len(results)}
    keys = ["pos_rmse_m", "vel_rmse_mps", "pos_p95_m", "init_latency_s",
            "id_accuracy", "track_continuity"]
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)),
                      "median": float(np.median(vals)),
                      "p90": float(np.percentile(vals, 90))}
    init_rate = [r.get("track_initiated", False) for r in results]
    agg["initiation_rate"] = float(np.mean(init_rate)) if init_rate else 0.0
    return agg
