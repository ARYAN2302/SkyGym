"""S4 interactive playground: YOU fly the drone; the sensors report what they see.

Run from a terminal (needs a TTY):
    python examples/interactive.py                     # manual control
    python examples/interactive.py --autopilot         # watch scripted flight
    python examples/interactive.py --scenario orbit    # scripted scenario

Manual keys (press Enter after each command, or chain like "wwd"):
    w/s : forward/back acceleration    a/d : left/right
    q/e : down/up                      space : hover in place
    p   : print latest detections      g : print witness ground truth (debug)
    r   : render snapshot PNG          x : quit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from skygym.config import EnvCfg
from skygym.env import SkyGymEnv

KEY_TO_ACC = {
    "w": np.array([0.0, 2.0, 0.0]),    # north
    "s": np.array([0.0, -2.0, 0.0]),   # south
    "d": np.array([2.0, 0.0, 0.0]),    # east
    "a": np.array([-2.0, 0.0, 0.0]),   # west
    "e": np.array([0.0, 0.0, 2.0]),    # up
    "q": np.array([0.0, 0.0, -2.0]),   # down
}


def fmt_dets(obs: dict) -> str:
    lines = []
    for sensor, d in obs.items():
        n = d["n"]
        if n == 0:
            lines.append(f"  {sensor:6s}: ---")
            continue
        arr = d["dets"][:n]
        for row in arr:
            az = row[0]
            el = row[1]
            r = row[2]
            cls_idx = int(np.nanargmax(row[7:11]))
            cls = ("quad", "fixed_wing", "bird", "unknown")[cls_idx]
            conf = row[7 + cls_idx]
            rng_s = f"{r/1000:6.2f}km" if np.isfinite(r) else "    --  "
            el_s = f"{el:5.1f}" if np.isfinite(el) else "  ---"
            clutter = " [clutter]" if row[3] > 0.5 else ""
            lines.append(f"  {sensor:6s}: az={az:6.1f} el={el_s} r={rng_s} "
                         f"cls={cls:10s} conf={conf:.2f}{clutter}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="approach",
                    help="initial scenario; manual control overrides flight")
    ap.add_argument("--autopilot", action="store_true",
                    help="scripted autopilot flies; just watch the report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--snap-dir", default="output/snaps")
    args = ap.parse_args()

    cfg = EnvCfg(mode="data" if args.autopilot else "control")
    env = SkyGymEnv(cfg)
    obs, info = env.reset(seed=args.seed,
                          options={"scenario": args.scenario,
                                   "duration_s": args.duration})
    print("== SkyGym interactive playground ==")
    print(f"scenario={args.scenario} mode={'autopilot' if args.autopilot else 'manual'}")
    if not args.autopilot:
        print("keys: w/s/a/d move | q/e down/up | space hover | p detections | "
              "g truth | r snapshot | x quit")
    os.makedirs(args.snap_dir, exist_ok=True)

    step = 0
    done = False
    while not done:
        gt = info["gt"]
        if args.autopilot:
            obs, _, te, tr, info = env.step(None)
            cmd_hint = "[autopilot]"
        else:
            keys = input(f"\nt={gt['t']:6.1f}s pos={np.round(gt['pos'],0)} > ").strip().lower()
            if keys == "x":
                break
            if keys == "p":
                print(fmt_dets(obs))
                continue
            if keys == "g":
                print(f"  WITNESS: pos={np.round(gt['pos'],1)} vel={np.round(gt['vel'],1)} "
                      f"az={gt['az_deg']:.1f} el={gt['el_deg']:.1f} "
                      f"r={gt['range_m']/1000:.2f}km cls={gt['true_class']}")
                continue
            if keys == "r":
                img = env.render()
                if img is not None:
                    import matplotlib
                    matplotlib.use("Agg")
                    from PIL import Image
                    p = os.path.join(args.snap_dir, f"snap_{step:05d}.png")
                    Image.fromarray(img).save(p)
                    print(f"  saved {p}")
                continue
            acc = np.zeros(3)
            if keys == "" or keys == "space":
                acc = -1.0 * env._state.vel  # damp to hover
            else:
                for ch in keys:
                    acc = acc + KEY_TO_ACC.get(ch, 0.0)
            obs, _, te, tr, info = env.step(acc.astype(np.float32))
            cmd_hint = f"[a={np.round(acc,1)}]"
        step += 1
        done = te or tr
        # live one-line sensor feed every 10 steps
        if step % 10 == 0:
            n_r = obs["radar"]["n"]; n_e = obs["eo"]["n"]; n_f = obs["rf"]["n"]
            print(f"{cmd_hint} t={gt['t']:6.1f}s dets: radar={n_r} eo={n_e} rf={n_f} "
                  f"r={gt['range_m']/1000:.2f}km")
    print("\nepisode end.")


if __name__ == "__main__":
    main()
