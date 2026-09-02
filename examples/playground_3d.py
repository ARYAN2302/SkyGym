#!/usr/bin/env python3
"""SkyGym 3D Interactive Playground — DJI Edition (Gymnasium).

Proper Gymnasium playground with a real 3D DJI drone (Three.js rendering)
and the SAME algo/logic layer as before:

    Python (truth): skygym/env.py -> flight.py (3-DOF + drag) -> sensors/* (radar/eo/rf)
                    Env is the Gymnasium contract: obs = corrupted dets, info["gt"] = witness
    JS (render):    only renders & sends accel — never physics/sensors

Features:
  - Duration bar: choose 15s / 20s / 60s / 120s / custom slider; env truncation honors it
  - Live recording: every step's dets + gt buffered -> Export JSONL/CSV when episode ends or on demand
  - Control vs Autopilot: control mode sends your accel, data mode flies scripted behaviour
  - 3D DJI model + trail + detection markers all driven by Python truth

Usage:
    python examples/playground_3d.py                 # http://localhost:8000/examples/playground_3d.html
    python examples/playground_3d.py --port 8001 --scenario serpentine
"""
import argparse
import http.server
import json
import math
import os
import socketserver
import sys
import threading
import webbrowser
from functools import partial
from urllib.parse import urlparse, parse_qs

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

import numpy as np
from skygym.config import EnvCfg
from skygym.env import SkyGymEnv

# Global env + recording
_lock = threading.Lock()
_env: SkyGymEnv | None = None
_env_mode: str | None = None
_recording: list[dict] = []  # list of {t, obs, gt}
_current_duration: float = 60.0
_current_scenario: str = "approach"
_current_seed: int = 0
_episode_id: str | None = None


def _get_env(mode: str):
    global _env, _env_mode
    if _env is None or _env_mode != mode:
        _env = SkyGymEnv(EnvCfg(mode=mode))  # mode = "data" (autopilot) or "control"
        _env_mode = mode
    return _env


def _sanitize(o):
    """Replace NaN/inf with None for JSON (JS JSON.parse can't handle NaN)."""
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, np.floating):
        v = float(o)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(o, np.ndarray):
        return [_sanitize(x) for x in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, np.integer):
        return int(o)
    return o


def _obs_to_ser(obs):
    out = {}
    for k, v in obs.items():
        n = int(v["n"])
        dets = v["dets"]
        # dets is (max_dets, 11) with NaN padding -> slice to n, sanitize
        arr = _sanitize(dets[:n].tolist()) if n > 0 else []
        out[k] = {"dets": arr, "n": n}
    return out


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            with _lock:
                self._send_json({"ok": True, "duration": _current_duration, "scenario": _current_scenario, "episode_id": _episode_id, "recording_len": len(_recording)})
            return
        return super().do_GET()

    def do_POST(self):
        global _current_duration, _current_scenario, _current_seed, _recording, _episode_id
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body.decode() or "{}")
        except Exception:
            data = {}

        if parsed.path == "/api/reset":
            scenario = data.get("scenario", _current_scenario) or "approach"
            try:
                seed = int(data.get("seed", _current_seed))
            except Exception:
                seed = 0
            try:
                duration = float(data.get("duration_s", _current_duration))
            except Exception:
                duration = 60.0
            duration = float(np.clip(duration, 5, 600))
            autopilot = bool(data.get("autopilot", False))
            mode = "data" if autopilot else "control"

            with _lock:
                _current_duration = duration
                _current_scenario = scenario
                _current_seed = seed
                env = _get_env(mode)
                obs, info = env.reset(seed=seed, options={"scenario": scenario, "duration_s": duration})
                _episode_id = info["gt"]["episode_id"]
                _recording = []
                # record t=0
                _recording.append({"t": float(info["gt"]["t"]), "obs": _obs_to_ser(obs), "gt": _sanitize({"pos": info["gt"]["pos"].tolist(), "vel": info["gt"]["vel"].tolist(), "az_deg": float(info["gt"]["az_deg"]), "el_deg": float(info["gt"]["el_deg"]), "range_m": float(info["gt"]["range_m"]), "true_class": info["gt"]["true_class"], "tx_on": bool(info["gt"]["tx_on"]), "scenario": info["gt"]["scenario"], "episode_id": _episode_id})})
                resp = {
                    "obs": _obs_to_ser(obs),
                    "gt": _sanitize({"t": float(info["gt"]["t"]), "pos": info["gt"]["pos"].tolist(), "vel": info["gt"]["vel"].tolist(), "az_deg": float(info["gt"]["az_deg"]), "el_deg": float(info["gt"]["el_deg"]), "range_m": float(info["gt"]["range_m"]), "true_class": info["gt"]["true_class"], "tx_on": bool(info["gt"]["tx_on"]), "scenario": info["gt"]["scenario"], "episode_id": _episode_id, "duration_s": duration, "mode": mode}),
                    "terminated": False, "truncated": False, "duration_s": duration, "mode": mode,
                }
            self._send_json(resp)
            return

        if parsed.path == "/api/step":
            action = data.get("action", None)
            if action is not None:
                try:
                    arr = np.asarray(action, dtype=np.float32)
                    if arr.shape != (3,):
                        action = None
                    else:
                        action = arr
                except Exception:
                    action = None
            with _lock:
                env = _get_env(_env_mode or "control")
                obs, _, terminated, truncated, info = env.step(action)
                obs_ser = _obs_to_ser(obs)
                gt_ser = _sanitize({"t": float(info["gt"]["t"]), "pos": info["gt"]["pos"].tolist(), "vel": info["gt"]["vel"].tolist(), "az_deg": float(info["gt"]["az_deg"]), "el_deg": float(info["gt"]["el_deg"]), "range_m": float(info["gt"]["range_m"]), "true_class": info["gt"]["true_class"], "tx_on": bool(info["gt"]["tx_on"]), "scenario": info["gt"]["scenario"], "episode_id": info["gt"]["episode_id"]})
                # append to recording (cap at ~10000 to avoid memory blowup)
                _recording.append({"t": gt_ser["t"], "obs": obs_ser, "gt": gt_ser})
                if len(_recording) > 10000:
                    _recording = _recording[-10000:]
                resp = {"obs": obs_ser, "gt": gt_ser, "terminated": bool(terminated), "truncated": bool(truncated), "duration_s": _current_duration, "mode": _env_mode, "recording_len": len(_recording)}
            self._send_json(resp)
            return

        if parsed.path == "/api/export":
            fmt = data.get("format", "jsonl")
            with _lock:
                rec = list(_recording)
                dur = _current_duration
                sc = _current_scenario
            if fmt == "jsonl":
                lines = [json.dumps({"t": r["t"], "gt": r["gt"], "obs": r["obs"]}) for r in rec]
                body_str = "\n".join(lines)
                self.send_response(200)
                self.send_header("Content-Type", "application/jsonl")
                self.send_header("Content-Disposition", f'attachment; filename="skygym_{sc}_{dur:.0f}s_{_episode_id}.jsonl"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body_str.encode())))
                self.end_headers()
                self.wfile.write(body_str.encode())
                return
            elif fmt == "csv":
                # flattened CSV: t, pos_x,y,z, vel_x,y,z, range, az, el, radar_n, eo_n, rf_n
                import io, csv
                out = io.StringIO()
                w = csv.writer(out)
                w.writerow(["t","pos_e","pos_n","pos_u","vel_e","vel_n","vel_u","range_m","az_deg","el_deg","true_class","tx_on","radar_n","eo_n","rf_n","episode_id","scenario","duration_s"])
                for r in rec:
                    gt = r["gt"]
                    obs = r["obs"]
                    w.writerow([gt["t"], gt["pos"][0], gt["pos"][1], gt["pos"][2], gt["vel"][0], gt["vel"][1], gt["vel"][2], gt["range_m"], gt["az_deg"], gt["el_deg"], gt["true_class"], gt["tx_on"], obs["radar"]["n"], obs["eo"]["n"], obs["rf"]["n"], gt["episode_id"], gt["scenario"], dur])
                body_str = out.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="skygym_{sc}_{dur:.0f}s_{_episode_id}.csv"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body_str.encode())))
                self.end_headers()
                self.wfile.write(body_str.encode())
                return
            else:
                self._send_json({"recording": rec, "duration_s": dur, "scenario": sc, "episode_id": _episode_id})
                return

        self.send_error(404, "unknown api")

    def _send_json(self, obj):
        data = json.dumps(obj, allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="SkyGym 3D Playground server (DJI, full Gymnasium)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scenario", default="approach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    global _current_scenario, _current_seed
    _current_scenario = args.scenario
    _current_seed = args.seed

    os.chdir(os.path.abspath(ROOT))
    _get_env("control").reset(seed=_current_seed, options={"scenario": _current_scenario, "duration_s": _current_duration})

    handler = partial(Handler, directory=os.path.abspath(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/examples/playground_3d.html"
        print(f"== SkyGym 3D Playground (DJI) — FULL GYM ==")
        print(f"Serving {os.path.abspath(ROOT)} at {url}")
        print(f"  Logic: skygym/env.py -> flight.py -> sensors/* (env is Gymnasium, obs=corrupted dets, info[gt]=witness)")
        print(f"  Render: Three.js + DJI Mavic (scale 1:1m) + orbit controls")
        print(f"  API: POST /api/reset {{scenario, duration_s, autopilot, seed}}  POST /api/step {{action:[ax,ay,az]|null}}  POST /api/export {{format}}")
        if not args.no_browser:
            try:
                webbrowser.open(url)
                print(f"Opened -> {url}")
            except Exception as e:
                print(f"(no browser: {e})")
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
