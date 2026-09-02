#!/usr/bin/env python3
"""SkyGym 3D Interactive Playground — DJI Edition.

Proper Gymnasium playground with a real 3D DJI drone (Three.js rendering)
and the SAME algo/logic layer as before:

    Python: skygym/env.py + flight.py + sensors/*  (truth + corruption)
    JS:     only rendering + input — never the physics/sensors

Usage:
    python examples/playground_3d.py                 # open http://localhost:8000/examples/playground_3d.html
    python examples/playground_3d.py --port 8001 --scenario serpentine --seed 7
    python examples/playground_3d.py --no-browser    # just serve

Then fly with WASD/QE/Space or the on-screen joystick. Orbit with mouse.
"""
import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser
from functools import partial
from urllib.parse import urlparse

# ensure skygym importable when launched from repo root or examples/
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

import numpy as np
from skygym.config import EnvCfg
from skygym.env import SkyGymEnv

# Global env + lock (single player, single drone — Gymnasium semantics)
_env: SkyGymEnv | None = None
_lock = threading.Lock()
_default_scenario = "approach"
_default_seed = 0


def _get_env():
    global _env
    if _env is None:
        _env = SkyGymEnv(EnvCfg())
    return _env


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        # serve API helpers as JSON? only GET we handle is static files
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body.decode() or "{}")
        except Exception:
            data = {}

        if parsed.path == "/api/reset":
            scenario = data.get("scenario", _default_scenario)
            seed = int(data.get("seed", _default_seed))
            duration = float(data.get("duration_s", 300.0))
            with _lock:
                env = _get_env()
                obs, info = env.reset(seed=seed, options={"scenario": scenario, "duration_s": duration})
                # convert obs to serializable: dets are np arrays -> list, n stays int
                obs_ser = {}
                for k, v in obs.items():
                    dets = v["dets"]
                    # dets is (max_dets, 11) float32 with NaNs padded; slice to n then to list
                    n = int(v["n"])
                    arr = dets[:n].tolist() if n > 0 else []
                    # also include up to 24 rows for generic client? we send n + full padded as fallback
                    obs_ser[k] = {"dets": arr, "n": n}
                resp = {
                    "obs": obs_ser,
                    "gt": {
                        "t": float(info["gt"]["t"]),
                        "pos": info["gt"]["pos"].tolist(),
                        "vel": info["gt"]["vel"].tolist(),
                        "az_deg": float(info["gt"]["az_deg"]),
                        "el_deg": float(info["gt"]["el_deg"]),
                        "range_m": float(info["gt"]["range_m"]),
                        "true_class": info["gt"]["true_class"],
                        "tx_on": bool(info["gt"]["tx_on"]),
                        "scenario": info["gt"]["scenario"],
                        "episode_id": info["gt"]["episode_id"],
                    },
                    "terminated": False,
                    "truncated": False,
                }
            self._send_json(resp)
            return

        if parsed.path == "/api/step":
            # action: None (autopilot/hover) or [ax,ay,az] in m/s^2
            action = data.get("action", None)
            if action is not None:
                try:
                    action = np.asarray(action, dtype=np.float32)
                except Exception:
                    action = None
            with _lock:
                env = _get_env()
                obs, _, terminated, truncated, info = env.step(action)
                obs_ser = {}
                for k, v in obs.items():
                    n = int(v["n"])
                    arr = v["dets"][:n].tolist() if n > 0 else []
                    obs_ser[k] = {"dets": arr, "n": n}
                resp = {
                    "obs": obs_ser,
                    "gt": {
                        "t": float(info["gt"]["t"]),
                        "pos": info["gt"]["pos"].tolist(),
                        "vel": info["gt"]["vel"].tolist(),
                        "az_deg": float(info["gt"]["az_deg"]),
                        "el_deg": float(info["gt"]["el_deg"]),
                        "range_m": float(info["gt"]["range_m"]),
                        "true_class": info["gt"]["true_class"],
                        "tx_on": bool(info["gt"]["tx_on"]),
                        "scenario": info["gt"]["scenario"],
                        "episode_id": info["gt"]["episode_id"],
                    },
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            self._send_json(resp)
            return

        self.send_error(404, "unknown api")

    def _send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # CORS for local dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="SkyGym 3D Playground server (DJI drone, same logic layer)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scenario", default="approach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open browser")
    args = ap.parse_args()

    global _default_scenario, _default_seed
    _default_scenario = args.scenario
    _default_seed = args.seed

    # serve from repo root so /examples/playground_3d.html resolves
    os.chdir(os.path.abspath(ROOT))

    # pre-warm env
    _get_env().reset(seed=_default_seed, options={"scenario": _default_scenario, "duration_s": 300})

    handler = partial(Handler, directory=os.path.abspath(ROOT))
    # allow reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/examples/playground_3d.html"
        print(f"== SkyGym 3D Playground (DJI) ==")
        print(f"Serving {os.path.abspath(ROOT)} at {url}")
        print(f"  Gymnasium logic: skygym/env.py + flight.py + sensors/*  (unchanged)")
        print(f"  Rendering: Three.js r160 + procedural DJI Mavic model")
        print(f"  Controls: WASD/QE/Space + joystick + mouse orbit")
        print(f"  API: POST /api/reset  {{scenario}}  |  POST /api/step {{action:[ax,ay,az] or null}}")
        if not args.no_browser:
            try:
                webbrowser.open(url)
                print(f"Opened browser -> {url}")
            except Exception as e:
                print(f"(could not auto-open browser: {e})")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
