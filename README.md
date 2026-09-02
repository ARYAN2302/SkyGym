# SkyGym

**A Gymnasium playground for counter-UAS recon data generation.** A drone flies.
Sensors report what they *think* they see. Ground truth stays hidden in the
witness channel. Everything is reproducible from a seed.

```
                ┌──────────────────────────────────────────────────┐
   action       │                    SkyGymEnv                     │      obs
  (None =  ───► │  flight model ─► TRUE STATE ─► sensor corruption │ ───► corrupted
   autopilot)   │  (3-DOF + drag)  (witness)    (noise/Pd/clutter/ │      detections
                │                                ID confusion/lat.)│      {az, el, r, ID}
                └──────────────────────────────────────────────────┘
                                                    │
                                              info["gt"]  ← witness channel
                                              (labels & eval ONLY — never input)
```

## The one twist

Every existing drone gym returns the drone's **own state** as the observation.
SkyGym returns the **recon sensor report** about the target — azimuth,
elevation, range and a corrupted threat-class posterior — exactly the
detection contract a counter-UAS tracking/fusion stack consumes:

```python
obs = {
  "radar": {"dets": (24, 11), "n": int},   # az, el, range + noise, Pd misses, clutter
  "eo":    {"dets": (24, 11), "n": int},   # pixel-bearing (tight), stereo range, blob ID
  "rf":    {"dets": (24, 11), "n": int},   # az-only (wide), protocol-ID, silent if TX off
}
info["gt"]  # hidden truth: pos, vel, true class — for labels/eval, never for input
```

Row layout (11 floats): `az_deg, el_deg, range_m, clutter_flag, snr_db,
pixel_px, t_meas, p_quad, p_fixed_wing, p_bird, p_unknown`.
NaN = "sensor does not measure this" (RF has no el/range).

## Why this is honest (validation gates)

| Gate | Test | Status |
|---|---|---|
| Geometry roundtrip | spherical ⇄ cartesian < 1e-6 deg/m | ✅ |
| Seed reproducibility | same seed → bit-identical detection stream | ✅ |
| Noise matches spec | Monte-Carlo σ within 25% of configured σ, unbiased | ✅ |
| Pd physics | SNR ∝ 1/r⁴, Pd collapses with range & small RCS (quad 0.93 @ 2 km → 0.04 @ 4 km; birds fade fast) | ✅ |
| GT separation | no witness keys in obs, enforced per step by QA wrapper | ✅ |
| Consumer honesty | EKF track-while-scan + GNN: 100% initiation, 4.5 m median RMSE, ~0.1 s latency, ~100% ID on test-split episodes | ✅ |

`pytest` runs all gates in < 2 s.

## Install & quickstart

```bash
pip install -e ".[dev]"

# 1) one scenario end-to-end: tracker eval + recorded rollout + coverage plots
python examples/demo.py --scenario serpentine --seed 42

# 2) generate a labelled dataset (JSONL per episode + manifest + QA report)
python examples/generate_dataset.py --episodes 60 --split train --out output/ds_train
python examples/generate_dataset.py --episodes 20 --split test  --out output/ds_test

# 3) batch-evaluate the baseline EKF consumer (defaults to TEST seed range)
python examples/evaluate_tracker.py --episodes 8

# 4) fly it yourself (terminal with TTY)
python examples/interactive.py --scenario approach
python examples/interactive.py --autopilot --scenario orbit
```

## What's inside

```
skygym/
  world.py      ENU frames, spherical transforms, measurement Jacobian/covariance
  flight.py     3-DOF point mass (drag, speed/accel limits, ground floor)
                + autopilot behaviours: hover · approach · orbit · waypoint ·
                serpentine (2 g weaver) · egress
  sensors/
    radar.py    az/el/range · SNR(r, RCS) → Pd · σ_r ∝ r² · σ_ang ∝ r ·
                Poisson clutter · micro-Doppler class confusion
    eo.py       pixel footprint → Pd · pixel-quantised bearing · stereo range ·
                blob-degraded ID · slaved/fixed gimbal modes
    rf.py       az-only DF (σ≈4°) · protocol fingerprint (best ID) ·
                deaf when drone TX off
  env.py        Gymnasium Env (reset/step/render, Dict obs space, witness info)
  wrappers.py   DetectionRecorder (JSONL+Parquet) · QAChecker · DistributionMonitor
  tracker.py    EKF track-while-scan, GNN gating, az-only updates, M-of-N initiation
  metrics.py    RMSE / initiation latency / continuity / ID accuracy vs witness
  scenarios.py  scenario sampling, grid×random dataset builder, seed-range splits
```

## Data hygiene rules (enforced in code, not docs)

- **Seed discipline** — every episode records its seed; train/val/test use
  disjoint seed ranges (`SEED_RANGES`), so evaluation can never peek at
  training scenarios.
- **Range-dependent noise** — σ grows with range exactly where SNR dies.
  No flat-noise lies.
- **Threat ID is always a corrupted posterior** — the true class never
  crosses the sensor boundary.
- **Coverage audits** — `DistributionMonitor` histograms az/el/range per
  sensor so "10 000 copies of one scenario" can't masquerade as volume.

## Scaling path

1. **Now** — NumPy 3-DOF, 1 drone, ~10⁵ steps/s, Gymnasium single-agent.
2. **S5 multi-drone** — PettingZoo Parallel API; N targets in the same env
   class (n=1 is the degenerate case); data association becomes the game.
3. **S6 fidelity** — PyBullet backend (attitude dynamics) and/or Isaac-style
   vectorised backend behind the same Gymnasium API; BlenderProc render
   branch for image-based ID when needed.

## Status

S1–S4 complete (env + sensors + wrappers + tracker + dataset builder +
interactive control). 28/28 tests passing. MIT licensed. Defensive research
tooling — synthetic data for detection/tracking algorithm development.
