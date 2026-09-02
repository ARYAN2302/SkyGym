#!/usr/bin/env python3
"""SkyGym Simple 3D Gym — 1 drone, 3 data points, 3 modes.

Core stays Python: skygym/env.py -> flight.py (3-DOF+drag) -> sensors/* (radar/eo/rf)
JS only renders & sends accel. Same Gymnasium contract: obs = corrupted dets, info["gt"] = witness.

Modes:
  Auto 20s — autopilot (approach) for 20s, truncated, data generated for exactly 20s
  Auto 40s — same for 40s
  Manual ∞ — control mode, you fly as long as you want (WASD/QE/joystick), up to 600s

Usage: python examples/playground_3d.py  -> http://localhost:8000/examples/playground_3d.html
"""
import argparse, http.server, json, math, os, socketserver, sys, threading, webbrowser
from functools import partial
from urllib.parse import urlparse

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))
import numpy as np
from skygym.config import EnvCfg
from skygym.env import SkyGymEnv

_lock = threading.Lock()
_env = None
_env_mode = None
_rec = []
_cur_dur = 20.0
_cur_sc = "approach"
_cur_seed = 0
_eid = None

def _get_env(mode: str):
    global _env, _env_mode
    if _env is None or _env_mode != mode:
        _env = SkyGymEnv(EnvCfg(mode=mode))  # data=autopilot, control=manual
        _env_mode = mode
    return _env

def _san(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, np.floating):
        v=float(o); return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, np.ndarray): return [_san(x) for x in o.tolist()]
    if isinstance(o, (list,tuple)): return [_san(x) for x in o]
    if isinstance(o, dict): return {k:_san(v) for k,v in o.items()}
    if isinstance(o, np.integer): return int(o)
    return o

def _obs_ser(obs):
    out={}
    for k,v in obs.items():
        n=int(v["n"]); arr=_san(v["dets"][:n].tolist()) if n>0 else []
        out[k]={"dets":arr,"n":n}
    return out

class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path=="/api/status":
            with _lock: self._j({"ok":True,"dur":_cur_dur,"eid":_eid,"steps":len(_rec)})
            return
        return super().do_GET()
    def do_POST(self):
        global _cur_dur,_cur_sc,_cur_seed,_rec,_eid
        p=urlparse(self.path).path
        body=self.rfile.read(int(self.headers.get("Content-Length",0)) or 0)
        try: data=json.loads(body.decode() or "{}")
        except: data={}
        if p=="/api/reset":
            sc=data.get("scenario") or _cur_sc
            try: seed=int(data.get("seed", _cur_seed))
            except: seed=0
            try: dur=float(data.get("duration_s", _cur_dur))
            except: dur=20.0
            dur=float(np.clip(dur,5,600))
            autopilot=bool(data.get("autopilot", False))
            mode="data" if autopilot else "control"
            with _lock:
                _cur_dur,_cur_sc,_cur_seed=dur,sc,seed
                env=_get_env(mode)
                obs,info=env.reset(seed=seed, options={"scenario":sc,"duration_s":dur})
                _eid=info["gt"]["episode_id"]; _rec=[]
                gt=_san({"t":float(info["gt"]["t"]),"pos":info["gt"]["pos"].tolist(),"vel":info["gt"]["vel"].tolist(),"az_deg":float(info["gt"]["az_deg"]),"el_deg":float(info["gt"]["el_deg"]),"range_m":float(info["gt"]["range_m"]),"true_class":info["gt"]["true_class"],"tx_on":bool(info["gt"]["tx_on"]),"scenario":info["gt"]["scenario"],"episode_id":_eid})
                _rec.append({"t":gt["t"],"obs":_obs_ser(obs),"gt":gt})
                resp={"obs":_obs_ser(obs),"gt":gt,"terminated":False,"truncated":False,"duration_s":dur,"mode":mode}
            self._j(resp); return
        if p=="/api/step":
            act=data.get("action")
            if act is not None:
                try:
                    a=np.asarray(act,dtype=np.float32)
                    act=a if a.shape==(3,) else None
                except: act=None
            with _lock:
                env=_get_env(_env_mode or "control")
                obs,_,te,tr,info=env.step(act)
                obs_s=_obs_ser(obs)
                gt=_san({"t":float(info["gt"]["t"]),"pos":info["gt"]["pos"].tolist(),"vel":info["gt"]["vel"].tolist(),"az_deg":float(info["gt"]["az_deg"]),"el_deg":float(info["gt"]["el_deg"]),"range_m":float(info["gt"]["range_m"]),"true_class":info["gt"]["true_class"],"tx_on":bool(info["gt"]["tx_on"]),"scenario":info["gt"]["scenario"],"episode_id":info["gt"]["episode_id"]})
                _rec.append({"t":gt["t"],"obs":obs_s,"gt":gt})
                if len(_rec)>12000: _rec=_rec[-12000:]
                resp={"obs":obs_s,"gt":gt,"terminated":bool(te),"truncated":bool(tr),"duration_s":_cur_dur,"mode":_env_mode,"recording_len":len(_rec)}
            self._j(resp); return
        if p=="/api/export":
            fmt=data.get("format","jsonl")
            with _lock: rec=list(_rec); dur=_cur_dur; sc=_cur_sc; eid=_eid
            if fmt=="jsonl":
                body_str="\n".join(json.dumps({"t":r["t"],"gt":r["gt"],"obs":r["obs"]}) for r in rec)
                self.send_response(200); self.send_header("Content-Type","application/jsonl")
                self.send_header("Content-Disposition",f'attachment; filename="skygym_{sc}_{dur:.0f}s_{eid}.jsonl"')
                self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Content-Length",str(len(body_str.encode()))); self.end_headers(); self.wfile.write(body_str.encode()); return
            if fmt=="csv":
                import io,csv
                out=io.StringIO(); w=csv.writer(out)
                w.writerow(["t","pos_e","pos_n","pos_u","vel_e","vel_n","vel_u","range_m","az_deg","el_deg","true_class","tx_on","radar_n","eo_n","rf_n","episode_id","scenario","duration_s"])
                for r in rec:
                    gt=r["gt"]; obs=r["obs"]; t=r.get("t",gt.get("t",0))
                    w.writerow([t,gt["pos"][0],gt["pos"][1],gt["pos"][2],gt["vel"][0],gt["vel"][1],gt["vel"][2],gt["range_m"],gt["az_deg"],gt["el_deg"],gt["true_class"],gt["tx_on"],obs["radar"]["n"],obs["eo"]["n"],obs["rf"]["n"],gt["episode_id"],gt["scenario"],dur])
                body_str=out.getvalue()
                self.send_response(200); self.send_header("Content-Type","text/csv")
                self.send_header("Content-Disposition",f'attachment; filename="skygym_{sc}_{dur:.0f}s_{eid}.csv"')
                self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Content-Length",str(len(body_str.encode()))); self.end_headers(); self.wfile.write(body_str.encode()); return
            self._j({"recording":rec,"duration_s":dur}); return
        self.send_error(404)
    def _j(self,obj):
        d=json.dumps(obj,allow_nan=False).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Content-Length",str(len(d))); self.end_headers(); self.wfile.write(d)
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS"); self.send_header("Access-Control-Allow-Headers","Content-Type"); super().end_headers()
    def do_OPTIONS(self): self.send_response(204); self.end_headers()

def main():
    import argparse
    ap=argparse.ArgumentParser(description="SkyGym Simple Gym: 1 drone, Auto 20/40, Manual ∞, 3 data points")
    ap.add_argument("--port",type=int,default=8000); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--no-browser",action="store_true")
    args=ap.parse_args()
    os.chdir(os.path.abspath(ROOT))
    _get_env("control").reset(seed=0, options={"scenario":"approach","duration_s":20})
    h=partial(H, directory=os.path.abspath(ROOT))
    socketserver.TCPServer.allow_reuse_address=True
    with socketserver.TCPServer((args.host,args.port), h) as httpd:
        url=f"http://{args.host}:{args.port}/examples/playground_3d.html"
        print(f"== SkyGym Simple Gym == 1 drone · Auto 20s/40s · Manual ∞ · 3 data points (radar/eo/rf)")
        print(f"Serving {os.path.abspath(ROOT)} at {url}")
        print(f"Core: skygym/env.py -> flight.py -> sensors/*  (Gymnasium obs=corrupted dets)")
        if not args.no_browser:
            try: webbrowser.open(url)
            except: pass
        print("Ctrl+C to stop.")
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nStopped.")
if __name__=="__main__": main()
