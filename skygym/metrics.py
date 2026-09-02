"""Grade the standard (Stone Soup) tracking baseline against the witness.

The heavy lifting lives in skygym.stone_soup; this module keeps the
metric-dict contract used by examples/demo.py, examples/evaluate.py and
the validation gates:

    pos_rmse_m, vel_rmse_mps, pos_p95_m, init_latency_s, track_initiated,
    track_continuity, id_accuracy, plus the stonesoup summary pass-through.
"""
from __future__ import annotations

from .stone_soup import run_episode as _ss_run_episode


def run_tracker_on_episode(env, seed: int, options: dict | None = None,
                           mode: str = "fusion",
                           eo_with_range: bool = True) -> dict:
    """Roll one episode, track with the Stone Soup baseline, score vs GT.

    options: passed to env.reset ({"scenario": name, "duration_s": s} or
             {"scenario_cfg": ScenarioCfg}).
    mode:    "fusion" (radar+EO+RF) or "radar" (radar only).
    """
    summary, rows, n_frames = _ss_run_episode(
        env, seed, options=options, mode=mode, eo_with_range=eo_with_range)
    first = summary.get("first_track_at_s")
    res = {
        "n_steps": n_frames,
        "n_est": len(rows),
        "track_initiated": summary["n_confirmed_tracks"] >= 1,
        "init_latency_s": first,
        "track_continuity": round(
            summary.get("pct_of_episode_tracked", 0.0) / 100.0, 3),
        "mode": summary["mode"],
        "eo_with_range": summary["eo_with_range"],
        "dets_fed": summary["dets_fed"],
    }
    for k in ("position_rmse_m", "pos_p95_m", "vel_rmse_mps",
              "mean_abs_az_err_deg", "mean_abs_el_err_deg",
              "mean_abs_range_err_m", "steady_state_mean_err_m",
              "steady_state_max_err_m", "id_accuracy_on_target_track",
              "n_track_candidates", "n_confirmed_tracks",
              "n_false_confirmed_tracks"):
        if k in summary:
            res[k] = summary[k]
    res["id_accuracy"] = summary.get("id_accuracy_on_target_track")
    return res


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-episode metric dicts into a summary."""
    import numpy as np

    agg = {"episodes": len(results)}
    keys = ["position_rmse_m", "vel_rmse_mps", "pos_p95_m", "init_latency_s",
            "id_accuracy", "track_continuity", "mean_abs_az_err_deg",
            "steady_state_mean_err_m"]
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)),
                      "median": float(np.median(vals)),
                      "p90": float(np.percentile(vals, 90))}
    init_rate = [r.get("track_initiated", False) for r in results]
    agg["initiation_rate"] = float(np.mean(init_rate)) if init_rate else 0.0
    return agg
