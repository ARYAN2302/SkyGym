# SkyGym

**A Gymnasium environment for counter-UAS recon data generation.** A drone
flies. Sensors report what they *think* they see. Ground truth stays hidden
in the witness channel. Everything is reproducible from a seed — and the
standard tracking baseline is [Stone Soup](https://github.com/dstl/Stone-Soup),
Dstl's open-source tracker.

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

## The one Difference

Every existing drone gym returns the drone's **own state** as the observation.
SkyGym returns the **recon sensor report about the target** — azimuth,
elevation, range and a corrupted threat-class posterior — exactly the
detection contract a counter-UAS tracking/fusion stack consumes:

```python
obs = {
  "radar": {"dets": (24, 11), "n": int},   # az/el/range + noise, Pd misses, clutter
  "eo":    {"dets": (24, 11), "n": int},   # pixel-bearing (tight), stereo range, blob ID
  "rf":    {"dets": (24, 11), "n": int},   # az-only (wide), protocol-ID, silent if TX off
}
info["gt"]  # hidden truth: pos, vel, true class — for labels/eval, never for input
```

Row layout (11 floats): `az_deg, el_deg, range_m, clutter_flag, snr_db,
pixel_px, t_meas, p_quad, p_fixed_wing, p_bird, p_unknown`.
NaN = "sensor does not measure this" (RF has no el/range; EO range is NaN in
mono mode).

## Install

```bash
pip install -e ".[dev]"        # env + sensors + wrappers + tests (incl. Stone Soup)
pip install -e ".[tracking]"   # just the tracking baseline (stonesoup)
```

Python ≥ 3.9 (developed on 3.13, stonesoup ≥ 1.0).

## Quickstart

```bash
# 1) one scenario end-to-end: tracking eval + recorded rollout + coverage plots
python examples/demo.py --scenario serpentine --seed 42

# 2) generate a labelled dataset (JSONL per episode + manifest + QA report)
python examples/generate_dataset.py --episodes 60 --split train --out output/ds_train
python examples/generate_dataset.py --episodes 20 --split test  --out output/ds_test

# 3) batch-evaluate the Stone Soup tracking baseline (defaults to TEST seeds)
python examples/evaluate.py --episodes 12 --scenario approach
python examples/evaluate.py --episodes 8 --mode radar

# 4) single-episode tracking benchmark, radar-only vs 3-sensor fusion
python examples/run_stonesoup.py --sensors radar  --start-km 1.2
python examples/run_stonesoup.py --sensors fusion --start-km 1.2
python examples/run_stonesoup.py --sensors fusion --start-km 4.0 --noise 2.0 --clutter 2.5

# 5) fly it yourself (terminal with TTY)
python examples/interactive.py --scenario approach
python examples/interactive.py --autopilot --scenario orbit

# 6) 3D playground (Three.js + local HTTP bridge into SkyGymEnv)
python examples/playground_3d.py                # → http://localhost:8000/examples/playground_3d.html
python examples/playground_3d.py --scenario serpentine --port 8001
```

## How it works (module map)

```
skygym/
  world.py       ENU frames, spherical transforms, measurement Jacobian/covariance
  flight.py      3-DOF point mass (drag, speed/accel limits, ground floor)
                 + autopilot behaviours: hover · approach · orbit · waypoint ·
                 serpentine (2 g weaver) · egress
  scenarios.py   scenario sampling, grid×random dataset builder, seed-range splits
  sensors/
    radar.py     az/el/range · SNR(r, RCS) → Pd · σ_r ∝ r² · σ_ang ∝ r ·
                 Poisson clutter every scan · micro-Doppler class confusion
    eo.py        pixel footprint → Pd · pixel-quantised bearing · stereo range ·
                 blob-degraded ID · slaved/fixed gimbal modes
    rf.py        az-only DF (σ≈4°) · protocol fingerprint (best ID) ·
                 deaf when drone TX off
  env.py         Gymnasium Env (reset/step, Dict obs space, witness info)
  wrappers.py    DetectionRecorder (JSONL+Parquet) · QAChecker · DistributionMonitor
  stone_soup.py  ★ standard tracking baseline: detections → Stone Soup EKF/GNN
                 → graded tracks (the only consumer the test suite gates on)
  metrics.py     RMSE / latency / continuity / ID accuracy vs witness (adapter)
examples/
  demo.py            end-to-end single-episode demo (eval + record + plots)
  generate_dataset.py train/val/test dataset builder
  evaluate.py        batch evaluation of the tracking baseline
  run_stonesoup.py   single-episode tracking benchmark CLI
  interactive.py     terminal flight control
  playground_3d.py   3D playground server (bridges Three.js UI ↔ SkyGymEnv)
  playground_3d.html Three.js scene + DJI model + HUD + exports
```

## The three sensors (and how each one lies/courupts)

| | Radar | EO/IR | RF DF |
|---|---|---|---|
| Measures | az, el, range | az, el (+ stereo range) | az only |
| Accuracy | σ_ang = 0.6°+0.5°/km · σ_r = 5 m + 3 m/km² | σ_ang = 0.08° (pixel) · σ_r ≈ 10 %·r | σ_az ≈ 4° |
| Signature weakness | SNR ∝ 1/r⁴ → Pd collapses with range (quad: 0.93 @ 2 km → 0.04 @ 4 km) | small blob → ID degrades; Pd falls with angular size | deaf when `tx_on=False`; bearing-only |
| False contacts | Poisson clutter every scan (birds/ground) | none modelled | none modelled |
| ID quality | confused with birds at low SNR | collapses to blob | best (protocol fingerprint, ~95 %+ when on) |

No single sensor sees everything — that asymmetry is the point. Fusion is
where the accuracy lives (see benchmark below).

## Tracking baseline: Stone Soup (standard since v0.2.0)

`skygym/stone_soup.py` is the single bridge from SkyGym detections to Stone
Soup and back to graded numbers. Per-detection measurement models:

- radar → `CartesianToElevationBearingRange` (az, el, range, σ from the radar model)
- EO → same EBR model **with stereo range** when finite, else `CartesianToElevationBearing`
- RF → `Cartesian2DToBearing` (az-only)

Tracker: EKF (constant-velocity 3D) + `GNNWith2DAssignment` (Mahalanobis
gate) + single-point initiator (**bearing-only sensors update but never
initiate**) + 1 s update-time deleter. Compass-vs-bearing conventions are
verified numerically at every run (`verified_mappings()`), because a wrong
axis mapping silently produces nonsense tracks.

```bash
python examples/run_stonesoup.py --sensors fusion --start-km 1.2   # → results/stonesoup/ (gitignored)
```

20 s approach episodes, seed 20260902:

| Scenario | Mode | Tracked | Pos RMSE | Steady-state | Az err | ID acc |
|---|---|---|---|---|---|---|
| 1.2 km, n1, c1 | radar only | 99.0 % | 32.8 m | 20.2 m | 0.248° | 100 %* |
| 1.2 km, n1, c1 | **fusion** | **99.5 %** | **22.1 m** | **3.9 m** | **0.081°** | **100 %*** |
| 4 km, n2, c2.5 | radar only | 11.9 % | 2197 m | track fragments < 2 s | — | — |
| 4 km, n2, c2.5 | **fusion** | **34.8 %** | **1157 m** | RF keeps a bearing | 2.74° | — |

\* det-level ID readout at the tracked position (Stone Soup does not classify;
the readout scores the class posteriors SkyGym emits alongside detections).

Reading: at 1.2 km fusion cuts steady-state error ~5× (EO's 0.08° bearing
pulls the track; radar anchors range; RF adds an independent bearing). At
4 km radar Pd collapses and radar-only tracking dies; RF bearings roughly
triple the tracked fraction — but range stays unobservable without radar
fixes. That gap is exactly what a learned fusion policy or a multi-hypothesis
tracker should close next.

Batch statistics over the test seed range (`examples/evaluate.py`,
3 approach episodes, 20 s): 100 % initiation, 0.985 median continuity,
2.2 m mean steady-state error.

**Why Stone Soup is the standard.** SkyGym ≤ v0.1.x shipped a hand-rolled
EKF track-while-scan baseline (`skygym/tracker.py`, removed in v0.2.0). A
head-to-head on identical episodes showed the two agree at close range
(built-in 17 m vs Stone Soup 33 m RMSE, radar-only) but diverge under sensor
heterogeneity: the greedy associator let RF's wide 4° gate capture the track
on 86/201 ticks (starving the radar fix, fragmenting track identity, 64 %
continuity), while Stone Soup's global 2D assignment held 99.5 %. Independent,
community-tested components make the benchmark numbers citable — and swap-in
upgrades (UKF, JPDA, MHT, IMM) are configuration changes, not rewrites.

## Dataset generation & data hygiene

```bash
python examples/generate_dataset.py --episodes 60 --split train --out output/ds_train
```

Rules enforced in code, not docs:

- **Seed discipline** — every episode records its seed; train/val/test use
  disjoint ranges (`SEED_RANGES` in `scenarios.py`): train 0–8 M,
  val 8–9 M, test 9–10 M. Evaluation can never peek at training scenarios.
- **Range-dependent noise** — σ grows with range exactly where SNR dies.
  No flat-noise lies.
- **Threat ID is always a corrupted posterior** — the true class never
  crosses the sensor boundary.
- **GT separation** — no witness keys in obs, enforced per step by the QA
  wrapper.
- **Coverage audits** — `DistributionMonitor` histograms az/el/range per
  sensor so "10 000 copies of one scenario" can't masquerade as volume.

## Validation gates

`pytest` runs all gates in seconds (27 tests):

| Gate | Test |
|---|---|
| Geometry roundtrip | spherical ⇄ cartesian < 1e-6 deg/m |
| Seed reproducibility | same seed → bit-identical detection stream |
| Noise matches spec | Monte-Carlo σ within 25 % of configured σ, unbiased |
| Pd physics | SNR ∝ 1/r⁴, Pd collapses with range & small RCS |
| GT separation | no witness keys in obs, enforced per step |
| Tracker honesty | Stone Soup bridge: track initiated, bounded RMSE, sane ID readout |

## 3D playground

`examples/playground_3d.py` serves a Three.js scene that talks to a real
`SkyGymEnv` over a local HTTP bridge — the JS side only renders and sends
acceleration commands; flight physics, sensors and recording stay in Python
(`Gymnasium` contract untouched). Modes: Auto 20 s / Auto 40 s (autopilot),
Manual ∞ (WASD/QE + joystick). One-click JSONL/CSV export of the recorded
episode (truth + per-sensor lie content, one CSV = one trainable example).

Known limitations (and the upgrade path):

1. **HTTP-per-step loop** — mitigated (reentrancy guard + drift-corrected
   catch-up batching), but a WebSocket transport would remove the remaining
   jitter on slow machines.
2. **Scale hacks** — the drone is drawn ~96× oversize so it stays visible at
   km distances. The clean fix is real-scale rendering + a chase camera +
   detections as distance-scaled sprites, plus a 2D radar PPI scope overlay
   for the "what does each sensor see" story.
3. **Trail rebuilds** — the thick trail re-uploads its geometry every step;
   preallocating a `BufferAttribute` + `drawRange` removes the GC churn.
4. **Detections as meshes** — pooled `InstancedMesh` instead of per-step
   sphere allocation.

## Roadmap

1. **Multi-drone (S5)** — PettingZoo Parallel API; N targets in the same env
   class (n=1 is the degenerate case); data association becomes the game.
2. **Fidelity (S6)** — PyBullet backend (attitude dynamics) and/or
   vectorised backend behind the same Gymnasium API; image-based ID branch.
3. **Learned fusion** — the reason this benchmark exists: train a policy on
   the (lie, truth) pairs to beat the Stone Soup baseline where classical
   filters die (far range, bearing-only regimes, dense clutter).
4. **Stronger classical baselines** — UKF / IMM / JPDA / MHT via Stone Soup
   config swaps; multi-target episodes with per-target ID.

## Changelog

- **v0.2.0** — Stone Soup is the standard tracking baseline
  (`skygym/stone_soup.py` + `examples/evaluate.py`); EO stereo range is now
  fused; det-level ID readout added; removed the hand-rolled
  `skygym/tracker.py` baseline and its eval script; generated results no
  longer tracked in git; playground loop made jitter-free (reentrancy guard
  + catch-up batching).
- **v0.1.1** — radar Poisson clutter drawn every scan (was only drawn when
  the target was beyond max range, contradicting the docstring).
- **v0.1.0** — S1–S4: env + sensors + wrappers + dataset builder +
  interactive control + 3D playground.

## Status

MIT licensed. Defensive research tooling — synthetic data for
detection/tracking algorithm development. Tests: 27 passing
(`python -m pytest`).
