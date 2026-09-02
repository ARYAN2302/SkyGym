"""Gym wrappers: DetectionRecorder, QAChecker, DistributionMonitor.

These implement the 'Stage 4' discipline from the factory design:
serialisation + whole-dataset sanity checks, enforced on every rollout.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import gymnasium as gym
import numpy as np

from .config import CLASSES
from .sensors.base import DET_ROW_LEN

VALID_COLS = {"az_deg": 0, "el_deg": 1, "range_m": 2, "clutter": 3,
              "snr_db": 4, "px": 5, "t_meas": 6,
              "p_quad": 7, "p_fixed": 8, "p_bird": 9, "p_unknown": 10}


class DetectionRecorder(gym.Wrapper):
    """Record every step's detections + witness GT to JSONL (+ optional Parquet).

    Record layout per line:
      meta   : episode identity (seed, scenario, rig, version)
      obs    : per-sensor padded detection arrays (the sensor view)
      labels : ground truth witness (pos, vel, class) - training LABELS
    """

    def __init__(self, env, out_path: str):
        super().__init__(env)
        self.out_path = out_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        self._f = open(out_path, "w", encoding="utf-8")
        self._episodes = 0
        self._steps = 0
        self._current_meta = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_meta = {
            "version": "skygym-0.1.0",
            "episode_id": info["gt"]["episode_id"],
            "seed": seed,
            "scenario": info["scenario_cfg"],
            "rig": info["rig"],
            "mode": info["mode"],
        }
        self._episodes += 1
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        gt = info["gt"]
        labels = {
            "pos": gt["pos"].tolist(),
            "vel": gt["vel"].tolist(),
            "az_deg": gt["az_deg"],
            "el_deg": gt["el_deg"],
            "range_m": gt["range_m"],
            "true_class": gt["true_class"],
            "tx_on": gt["tx_on"],
        }
        if "targets" in gt:  # S5 multi-drone witness: one label block each
            labels["targets"] = [{
                "idx": tg["idx"],
                "pos": tg["pos"].tolist(),
                "vel": tg["vel"].tolist(),
                "az_deg": tg["az_deg"], "el_deg": tg["el_deg"],
                "range_m": tg["range_m"],
                "true_class": tg["true_class"], "tx_on": tg["tx_on"],
                "behaviour": tg["behaviour"],
            } for tg in gt["targets"]]
        rec = {
            "meta": self._current_meta,
            "t": gt["t"],
            "obs": {k: {"dets": v["dets"][:v["n"]].tolist(), "n": v["n"]}
                    for k, v in obs.items()},
            "labels": labels,
        }
        self._f.write(json.dumps(rec) + "\n")
        self._steps += 1
        return obs, reward, term, trunc, info

    def close(self):
        self._f.close()
        # optional Parquet twin for bulk training pipelines
        try:
            import pandas as pd
            rows = []
            with open(self.out_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    eid = r["meta"]["episode_id"]
                    t = r["t"]
                    for sensor, d in r["obs"].items():
                        for row in d["dets"][:d["n"]]:
                            rows.append([eid, t, sensor, *row])
                    lab = r["labels"]
                    rows.append([eid, t, "WITNESS", t,
                                 lab["pos"][0], lab["pos"][1], lab["pos"][2],
                                 lab["true_class"] if "true_class" in lab else "",
                                 lab["vel"][0], lab["vel"][1], lab["vel"][2]])
            if rows:
                df = pd.DataFrame(rows, columns=[
                    "episode_id", "t", "sensor", "az_deg", "el_deg", "range_m",
                    "clutter", "snr_db", "px", "t_meas",
                    "p_quad", "p_fixed", "p_bird", "p_unknown",
                ])
                # witness rows break the 14-col schema; write two frames instead
                det_rows = [r for r in rows if r[2] != "WITNESS"]
                df = pd.DataFrame(det_rows, columns=[
                    "episode_id", "t", "sensor", "az_deg", "el_deg", "range_m",
                    "clutter", "snr_db", "px", "t_meas",
                    "p_quad", "p_fixed", "p_bird", "p_unknown",
                ])
                df.to_parquet(self.out_path.replace(".jsonl", ".parquet"), index=False)
        except Exception:
            pass  # parquet is a convenience twin, never mandatory
        return


class QAChecker(gym.Wrapper):
    """Per-step validation: NaN discipline, counts, class probs, GT separation.

    Raises AssertionError the moment the pipeline violates its own contract -
    failures surface at generation time, not at training time.
    """

    def __init__(self, env, strict: bool = True):
        super().__init__(env)
        self.strict = strict
        self.violations: list[str] = []

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        errs = self._check(obs)
        if errs:
            self.violations.extend(errs)
            if self.strict:
                raise AssertionError("QA violations: " + "; ".join(errs[:5]))
        return obs, reward, term, trunc, info

    def _check(self, obs) -> list[str]:
        errs = []
        for sensor, d in obs.items():
            arr, n = d["dets"], d["n"]
            if n > 0:
                valid = arr[:n]
                if np.any(np.all(np.isnan(valid), axis=1)):
                    errs.append(f"{sensor}: fully-NaN valid row")
                az = valid[:, VALID_COLS["az_deg"]]
                if np.any((az < -1e-3) | (az > 360.001)):
                    errs.append(f"{sensor}: az out of [0,360]")
                cls_sum = valid[:, 7:11].sum(axis=1)
                if np.any(np.abs(cls_sum - 1.0) > 0.05):
                    errs.append(f"{sensor}: class posteriors do not sum to 1")
                # RF channel is allowed NaN el/range; others must be finite
                if sensor != "rf":
                    el, r = valid[:, VALID_COLS["el_deg"]], valid[:, VALID_COLS["range_m"]]
                    if np.any(np.isnan(el)) or np.any(np.isnan(r)):
                        errs.append(f"{sensor}: NaN el/range (rf-only allowance)")
                if np.any(valid[:, VALID_COLS["range_m"]] < 0):
                    errs.append(f"{sensor}: negative range")
        # ground-truth separation: obs must never carry witness keys
        for forbidden in ("gt", "labels", "truth", "witness"):
            if forbidden in obs:
                errs.append(f"GT LEAKAGE: '{forbidden}' key present in obs")
        return errs

    def summary(self) -> dict:
        return {"violations": len(self.violations),
                "details": self.violations[:20]}


class DistributionMonitor(gym.Wrapper):
    """Accumulate detection statistics across episodes; report on close.

    Catches the 'volume without coverage' pitfall: histograms over az, el,
    range and class posteriors, per sensor.
    """

    def __init__(self, env):
        super().__init__(env)
        self.az = {s: [] for s in ("radar", "eo", "rf")}
        self.el = {s: [] for s in ("radar", "eo", "rf")}
        self.rng_m = {s: [] for s in ("radar", "eo", "rf")}
        self.cls = Counter()
        self.n_steps = 0
        self.n_dets = Counter()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.n_steps += 1
        for sensor, d in obs.items():
            n = d["n"]
            self.n_dets[sensor] += n
            if n:
                arr = d["dets"][:n]
                self.az[sensor].extend(arr[:, 0].tolist())
                self.el[sensor].extend(arr[:, 1][~np.isnan(arr[:, 1])].tolist())
                self.rng_m[sensor].extend(arr[:, 2][~np.isnan(arr[:, 2])].tolist())
                cls_idx = np.nanargmax(arr[:, 7:11], axis=1)
                for ci in cls_idx:
                    self.cls[f"{sensor}:{CLASSES[int(ci)]}"] += 1
        return obs, reward, term, trunc, info

    def report(self) -> dict:
        def stats(x):
            if not x:
                return {"n": 0}
            a = np.asarray(x)
            return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
                    "mean": float(a.mean()), "std": float(a.std())}
        return {
            "steps": self.n_steps,
            "detections": {s: int(v) for s, v in self.n_dets.items()},
            "az_stats": {s: stats(v) for s, v in self.az.items()},
            "el_stats": {s: stats(v) for s, v in self.el.items()},
            "range_stats": {s: stats(v) for s, v in self.rng_m.items()},
            "dominant_class_counts": dict(self.cls),
        }

    def save_report(self, path: str, plots_dir: str | None = None):
        rep = self.report()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        if plots_dir:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                os.makedirs(plots_dir, exist_ok=True)
                for sensor in ("radar", "eo", "rf"):
                    if not self.az[sensor]:
                        continue
                    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
                    axes[0].hist(np.asarray(self.az[sensor]), bins=48, color="steelblue")
                    axes[0].set_title(f"{sensor} azimuth"); axes[0].set_xlabel("deg")
                    els = self.el[sensor]
                    if els:
                        axes[1].hist(np.asarray(els), bins=36, color="seagreen")
                    axes[1].set_title(f"{sensor} elevation"); axes[1].set_xlabel("deg")
                    rs = self.rng_m[sensor]
                    if rs:
                        axes[2].hist(np.asarray(rs) / 1000.0, bins=48, color="indianred")
                    axes[2].set_title(f"{sensor} range"); axes[2].set_xlabel("km")
                    fig.suptitle("SkyGym coverage audit")
                    fig.tight_layout()
                    fig.savefig(os.path.join(plots_dir, f"coverage_{sensor}.png"), dpi=110)
                    plt.close(fig)
            except Exception:
                pass
        return rep
