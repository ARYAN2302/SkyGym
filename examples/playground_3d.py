#!/usr/bin/env python3
"""SkyGym 3D Playground v5 — swarm-native, RC-stick quad flight, live PPI,
EO/IR feed, terrain+sky, live Stone Soup score, timeline scrub, behaviour
commanding, single-drone Fly mode.

Core stays Python: skygym/env.py + multidrone.py -> flight.py -> sensors/*.
JS renders and composes RC sticks; the angle-mode quad controller (attitude
lag, tilt -> acceleration, yaw rate, climb-rate servo) lives in flight.py.
Same Gymnasium contract: obs = corrupted dets, info["gt"] = witness channel
(+ attitude of the manually-flown drone, control mode only).

Modes (all reachable from the v5 client, swarm is the boot default):
  Swarm 20s       - MultiDroneEnv fleet (n_drones), autopilot, data mode
  Solo auto       - autopilot single drone (data mode)
  Fly             - control mode: RC sticks (W/S pitch, A/D yaw, Q/E climb,
                    gamepad Mode-2); 1-4 possess any fleet drone, or FLY
                    SINGLE (n=1: just you and one drone, no fleet)

Exports carry ONLY what this session actually produced: per-tick witness,
the raw obs detection rows (dets CSV, dataset schema), the stick actions
you sent, the behaviour switches you commanded and the live Stone Soup
score — nothing is re-simulated or padded.

Usage: python examples/playground_3d.py  ->  http://localhost:8000/examples/playground_3d.html
"""
import argparse, http.server, io, csv, json, math, os, socketserver, sys, threading, webbrowser
from functools import partial
from urllib.parse import urlparse

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))
import numpy as np
from skygym.config import EnvCfg
from skygym.env import SkyGymEnv
from skygym.multidrone import MultiDroneEnv

try:
    from skygym.stone_soup import OnlineMultiTracker as _OMT
except Exception:                       # stonesoup/scipy missing -> no score
    _OMT = None

_lock = threading.Lock()
_env = None            # SkyGymEnv (data or control)
_menv = None           # MultiDroneEnv (swarm)
_active = None         # "single" | "multi"
_rec = []
_cur_dur = 20.0
_cur_sc = "approach"
_cur_seed = 0
_cur_n = 1
_eid = None
_otr = None            # OnlineMultiTracker (live Stone Soup score)
_otr_on = True         # client can disable via reset {"tracker": false}


def _get_env(mode: str):
    global _env, _env_mode, _active
    if _env is None or _env_mode != mode:
        _env = SkyGymEnv(EnvCfg(mode=mode))  # data=autopilot, control=manual
        _env_mode = mode
    _active = "single"
    return _env


def _get_multi_env(n: int, autopilot: bool = True):
    global _menv, _active
    max_d = max(24, 12 * n)
    key = (n, max_d, autopilot)
    if _menv is None or getattr(_menv, "_cfg_n", None) != key:
        # data = whole fleet on autopilot; control = YOU fly drone 1 (the
        # fleet keeps its autopilot - MultiDroneEnv.step handles k == 0).
        _menv = MultiDroneEnv(EnvCfg(mode="data" if autopilot else "control",
                                     max_dets_per_sensor=max_d))
        _menv._cfg_n = key
    _active = "multi"
    return _menv


def _san(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, np.floating):
        v = float(o); return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, np.ndarray): return [_san(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)): return [_san(x) for x in o]
    if isinstance(o, dict): return {k: _san(v) for k, v in o.items()}
    if isinstance(o, np.integer): return int(o)
    return o


def _obs_ser(obs):
    out = {}
    for k, v in obs.items():
        n = int(v["n"]); arr = _san(v["dets"][:n].tolist()) if n > 0 else []
        out[k] = {"dets": arr, "n": n}
    return out


def _gt_ser(gt: dict) -> dict:
    """Serialize the witness channel for JSON (single or multi fleet)."""
    out = {
        "t": float(gt["t"]),
        "pos": gt["pos"].tolist(), "vel": gt["vel"].tolist(),
        "az_deg": float(gt["az_deg"]), "el_deg": float(gt["el_deg"]),
        "range_m": float(gt["range_m"]),
        "true_class": gt["true_class"], "tx_on": bool(gt["tx_on"]),
        "scenario": gt.get("scenario", ""),
        "episode_id": gt.get("episode_id", ""),
    }
    if "targets" in gt:  # S5 multi-drone witness
        out["n_targets"] = int(gt["n_targets"])
        out["targets"] = [{
            "idx": tg["idx"],
            "pos": tg["pos"].tolist(), "vel": tg["vel"].tolist(),
            "az_deg": float(tg["az_deg"]), "el_deg": float(tg["el_deg"]),
            "range_m": float(tg["range_m"]),
            "true_class": tg["true_class"], "tx_on": bool(tg["tx_on"]),
            "behaviour": tg["behaviour"],
            **({"attitude": tg["attitude"]} if "attitude" in tg else {}),
        } for tg in gt["targets"]]
    if "attitude" in gt:
        out["attitude"] = gt["attitude"]
    return out


def _noise_scale() -> float:
    """Fleet noise multiplier of the active env (1.0 for single env)."""
    env = _menv if _active == "multi" else _env
    fleet = getattr(env, "_fleet", None)
    return float(fleet[0].noise_scale) if fleet else 1.0


def _targets_pos(gt: dict) -> list:
    return ([tg["pos"] for tg in gt["targets"]] if "targets" in gt
            else [gt["pos"]])


def _tracker_snapshot(gt: dict, obs: dict) -> dict | None:
    """One OnlineMultiTracker tick (None when disabled/unavailable)."""
    global _otr
    if not (_otr_on and _OMT is not None):
        return None
    if _otr is None:
        env = _menv if _active == "multi" else _env
        _otr = _OMT(env.cfg.rig)
    try:
        return _san(_otr.update(float(gt["t"]), obs, _targets_pos(gt),
                                _env.cfg.rig.site_enu,
                                noise_scale=_noise_scale()))
    except Exception:                   # live score must never kill a session
        return None


def _reset_tracker() -> None:
    global _otr
    _otr = None                          # rebuilt (and reset) on next tick


class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == "/api/status":
            with _lock:
                self._j({"ok": True, "dur": _cur_dur, "eid": _eid,
                         "steps": len(_rec), "active": _active,
                         "n": _cur_n, "tracker": _OMT is not None})
            return
        return super().do_GET()

    def do_POST(self):
        global _cur_dur, _cur_sc, _cur_seed, _cur_n, _rec, _eid, _active
        global _otr_on
        p = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        try: data = json.loads(body.decode() or "{}")
        except Exception: data = {}

        if p == "/api/reset":
            sc = data.get("scenario") or _cur_sc
            try: seed = int(data.get("seed", _cur_seed))
            except Exception: seed = 0
            try: dur = float(data.get("duration_s", _cur_dur))
            except Exception: dur = 20.0
            dur = float(np.clip(dur, 5, 600))
            autopilot = bool(data.get("autopilot", False))
            try: n = int(np.clip(int(data.get("n_drones", 1)), 1, 4))
            except Exception: n = 1
            _otr_on = bool(data.get("tracker", True))
            with _lock:
                _cur_dur, _cur_sc, _cur_seed, _cur_n = dur, sc, seed, n
                _rec = []
                _reset_tracker()
                if n > 1:
                    env = _get_multi_env(n, autopilot)
                    obs, info = env.reset(seed=seed, options={
                        "n_drones": n, "duration_s": dur,
                        "noise_scale": float(data.get("noise_scale", 1.0)),
                        "clutter_scale": float(data.get("clutter_scale", 1.0)),
                        "mix": ([s.strip() for s in sc.split(",")]
                                if (n > 1 and sc and "," in sc) else None)})
                    _eid = info["gt"]["episode_id"]
                    gt = _san(_gt_ser(info["gt"]))
                    score = _tracker_snapshot(info["gt"], obs)
                    _rec.append({"t": gt["t"], "obs": _obs_ser(obs),
                                 "gt": gt, "action": None, "score": score})
                    resp = {"obs": _obs_ser(obs), "gt": gt, "terminated": False,
                            "truncated": False, "duration_s": dur,
                            "mode": "multi", "n_drones": n, "score": score,
                            "tracker": _otr_on and _OMT is not None}
                    _active = "multi"
                else:
                    env = _get_env("data" if autopilot else "control")
                    obs, info = env.reset(seed=seed, options={
                        "scenario": sc, "duration_s": dur})
                    _eid = info["gt"]["episode_id"]
                    gt = _san(_gt_ser(info["gt"]))
                    score = _tracker_snapshot(info["gt"], obs)
                    _rec.append({"t": gt["t"], "obs": _obs_ser(obs),
                                 "gt": gt, "action": None, "score": score})
                    resp = {"obs": _obs_ser(obs), "gt": gt, "terminated": False,
                            "truncated": False, "duration_s": dur,
                            "mode": _env_mode, "n_drones": 1, "score": score,
                            "tracker": _otr_on and _OMT is not None}
                    _active = "single"
            self._j(resp); return

        if p == "/api/behaviour":
            with _lock:
                if _active != "multi":
                    self._j({"ok": False,
                             "error": "behaviour commanding needs a swarm "
                                      "(solo/fly-single has one drone)"})
                    return
                try: idx = int(data.get("idx", -1))
                except Exception: idx = -1
                name = str(data.get("behaviour", ""))
                try:
                    r = _menv.set_behaviour(idx, name)
                    r["fleet"] = [tg.name for tg in _menv._fleet]
                    self._j({"ok": True, **_san(r)})
                except (ValueError, IndexError) as e:
                    self._j({"ok": False, "error": str(e)})
            return

        if p == "/api/step":
            act = data.get("action")
            if act is not None:
                try:
                    a = np.asarray(act, dtype=np.float32)
                    act = a if a.shape in ((3,), (4,)) else None
                except Exception: act = None
            try: ci = int(data.get("control_idx", 0))
            except Exception: ci = 0
            try: n_steps = int(data.get("steps", 1))
            except Exception: n_steps = 1
            n_steps = int(np.clip(n_steps, 1, 8))
            with _lock:
                obs_s, gt, te, tr, frames = None, None, False, False, []
                act_out = _san(act.tolist() if hasattr(act, "tolist") else None)
                for _i in range(n_steps):
                    if _active == "multi":
                        # control-mode multi: sticks drive the possessed drone;
                        # data-mode multi ignores them (pure autopilot fleet)
                        obs, _, te, tr, info = _menv.step(act, control_idx=ci)
                    else:
                        env = _get_env(_env_mode or "control")
                        obs, _, te, tr, info = env.step(act)
                    obs_s = _obs_ser(obs)
                    gt = _san(_gt_ser(info["gt"]))
                    score = _tracker_snapshot(info["gt"], obs)
                    _rec.append({"t": gt["t"], "obs": obs_s, "gt": gt,
                                 "action": act_out, "score": score})
                    if "targets" in gt:
                        frames.append({"t": gt["t"],
                                       "pos": gt["pos"],
                                       "targets": [tg["pos"] for tg in gt["targets"]]})
                    else:
                        frames.append({"t": gt["t"], "pos": gt["pos"]})
                    if len(_rec) > 12000: _rec = _rec[-12000:]
                    if te or tr: break
                resp = {"obs": obs_s, "gt": gt, "terminated": bool(te),
                        "truncated": bool(tr), "duration_s": _cur_dur,
                        "mode": _active or "single", "n_drones": _cur_n,
                        "recording_len": len(_rec), "steps_done": len(frames),
                        "frames": frames, "score": score,
                        "tracker": _otr_on and _OMT is not None}
            self._j(resp); return

        if p == "/api/export":
            fmt = data.get("format", "jsonl")
            with _lock: rec = list(_rec); dur = _cur_dur; sc = _cur_sc; eid = _eid
            if fmt == "jsonl":
                body_str = "\n".join(json.dumps({"t": r["t"], "gt": r["gt"],
                                                 "obs": r["obs"],
                                                 "action": r.get("action"),
                                                 "score": r.get("score")})
                                     for r in rec)
                self._file(body_str, "application/jsonl",
                           f"skygym_{sc.replace(',', '_')}_{dur:.0f}s_{eid}.jsonl")
                return
            if fmt == "csv_dets":
                # one row per detection, exactly the Stage-0 dataset schema
                # (wrappers.DetectionRecorder parquet twin, minus witness)
                out = io.StringIO(); w = csv.writer(out)
                w.writerow(["episode_id", "t", "sensor", "az_deg", "el_deg",
                            "range_m", "clutter", "snr_db", "px", "t_meas",
                            "p_quad", "p_fixed", "p_bird", "p_unknown"])
                for r in rec:
                    obs = r["obs"]
                    for sensor in ("radar", "eo", "rf"):
                        ch = obs.get(sensor) or {"dets": [], "n": 0}
                        for row in ch["dets"][:ch["n"]]:
                            w.writerow([eid, r["t"], sensor] +
                                       [(None if v is None or
                                         (isinstance(v, float) and
                                          not math.isfinite(v)) else
                                        round(v, 4) if isinstance(v, float)
                                        else v) for v in row])
                self._file(out.getvalue(), "text/csv",
                           f"skygym_dets_{sc.replace(',', '_')}_{dur:.0f}s_{eid}.csv")
                return
            if fmt == "csv":
                out = io.StringIO(); w = csv.writer(out)
                multi = bool(rec) and "targets" in rec[0]["gt"]
                if multi:
                    hdr = ["t", "target_idx", "behaviour", "true_class", "tx_on",
                           "gt_pos_e", "gt_pos_n", "gt_pos_u",
                           "gt_vel_e", "gt_vel_n", "gt_vel_u",
                           "gt_range_m", "gt_az_deg", "gt_el_deg",
                           "act_pitch", "act_roll", "act_yaw", "act_climb",
                           "ss_tracked_pct", "ss_pos_rmse_m", "ss_id_switches",
                           "radar_n", "eo_n", "rf_n", "episode_id", "scenario"]
                    w.writerow(hdr)
                    for r in rec:
                        gt = r["gt"]; obs = r["obs"]; t = r.get("t", gt.get("t", 0))
                        a = r.get("action") or [None, None, None, None]
                        s = r.get("score") or {}
                        for tg in gt["targets"]:
                            w.writerow([t, tg["idx"], tg["behaviour"],
                                        tg["true_class"], tg["tx_on"],
                                        *tg["pos"], *tg["vel"], tg["range_m"],
                                        tg["az_deg"], tg["el_deg"],
                                        a[0], a[1], a[2], a[3],
                                        s.get("tracked_pct"),
                                        s.get("pos_rmse_m"),
                                        s.get("id_switches"),
                                        obs["radar"]["n"], obs["eo"]["n"],
                                        obs["rf"]["n"],
                                        gt["episode_id"], gt["scenario"]])
                else:
                    hdr = ["t", "gt_pos_e", "gt_pos_n", "gt_pos_u",
                           "gt_vel_e", "gt_vel_n", "gt_vel_u",
                           "gt_range_m", "gt_az_deg", "gt_el_deg",
                           "true_class", "tx_on",
                           "act_pitch", "act_roll", "act_yaw", "act_climb",
                           "ss_tracked_pct", "ss_pos_rmse_m", "ss_id_switches",
                           "radar_n", "radar_az_deg", "radar_el_deg",
                           "radar_range_m", "radar_snr_db", "radar_pixel_px",
                           "radar_clutter_flag", "radar_p_quad", "radar_p_fixed",
                           "radar_p_bird", "radar_p_unknown",
                           "eo_n", "eo_az_deg", "eo_el_deg", "eo_range_m",
                           "eo_snr_db", "eo_pixel_px", "eo_clutter_flag",
                           "eo_p_quad", "eo_p_fixed", "eo_p_bird", "eo_p_unknown",
                           "rf_n", "rf_az_deg", "rf_snr_db", "rf_clutter_flag",
                           "rf_p_quad", "rf_p_fixed", "rf_p_bird", "rf_p_unknown",
                           "episode_id", "scenario", "duration_s"]
                    w.writerow(hdr)
                    def _first_lie(dets):
                        if not dets: return None
                        for d in dets:
                            if d[3] == 0 or d[3] == 0.0:
                                return d
                        return dets[0]
                    for r in rec:
                        gt = r["gt"]; obs = r["obs"]; t = r.get("t", gt.get("t", 0))
                        rd = _first_lie(obs["radar"]["dets"]) if obs["radar"]["n"] > 0 else None
                        eo = _first_lie(obs["eo"]["dets"]) if obs["eo"]["n"] > 0 else None
                        rf = _first_lie(obs["rf"]["dets"]) if obs["rf"]["n"] > 0 else None
                        def g(d, i): return d[i] if d is not None and len(d) > i else None
                        a = r.get("action") or [None, None, None, None]
                        s = r.get("score") or {}
                        w.writerow([t, *gt["pos"], *gt["vel"], gt["range_m"],
                                    gt["az_deg"], gt["el_deg"], gt["true_class"],
                                    gt["tx_on"],
                                    a[0], a[1], a[2], a[3],
                                    s.get("tracked_pct"), s.get("pos_rmse_m"),
                                    s.get("id_switches"),
                                    obs["radar"]["n"], g(rd, 0), g(rd, 1), g(rd, 2),
                                    g(rd, 4), g(rd, 5), g(rd, 3),
                                    g(rd, 7), g(rd, 8), g(rd, 9), g(rd, 10),
                                    obs["eo"]["n"], g(eo, 0), g(eo, 1), g(eo, 2),
                                    g(eo, 4), g(eo, 5), g(eo, 3),
                                    g(eo, 7), g(eo, 8), g(eo, 9), g(eo, 10),
                                    obs["rf"]["n"], g(rf, 0), g(rf, 4), g(rf, 3),
                                    g(rf, 7), g(rf, 8), g(rf, 9), g(rf, 10),
                                    gt["episode_id"], gt["scenario"], dur])
                self._file(out.getvalue(), "text/csv",
                           f"skygym_{sc.replace(',', '_')}_{dur:.0f}s_{eid}.csv")
                return
            self._j({"recording": rec, "duration_s": dur}); return
        self.send_error(404)

    def _file(self, body_str, ctype, fname):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body_str.encode())))
        self.end_headers(); self.wfile.write(body_str.encode())

    def _j(self, obj):
        d = json.dumps(obj, allow_nan=False).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(d)))
        self.end_headers(); self.wfile.write(d)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self): self.send_response(204); self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="SkyGym 3D Playground v5")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    os.chdir(os.path.abspath(ROOT))
    _get_env("control").reset(seed=0, options={"scenario": "approach", "duration_s": 20})
    h = partial(H, directory=os.path.abspath(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), h) as httpd:
        url = f"http://{args.host}:{args.port}/examples/playground_3d.html"
        print("== SkyGym 3D Playground v5 == Swarm (2-4 drones) | Solo auto | Fly (single or 1-4 possess, FPV)")
        print("   Live Stone Soup score + timeline scrub + EO/IR feed + behaviour commanding")
        print(f"Serving {os.path.abspath(ROOT)} at {url}")
        print("Core: skygym/env.py + multidrone.py -> flight.py -> sensors/*")
        if not args.no_browser:
            try: webbrowser.open(url)
            except Exception: pass
        print("Ctrl+C to stop.")
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nStopped.")


if __name__ == "__main__":
    main()
