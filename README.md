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

# 5) MULTI-DRONE (S5): fleet tracking with global assignment + identity switches
python examples/run_multidrone.py --n 3 --episodes 4 --duration 20          # spread fleet
python examples/run_multidrone.py --n 2 --episodes 3 --duration 60 \
       --mix approach,approach --radius 300,320                              # collision course

# 6) fly it yourself (terminal with TTY)
python examples/interactive.py --scenario approach
python examples/interactive.py --autopilot --scenario orbit

# 7) 3D playground v2 (Three.js + PPI scope + swarm + chase cam)
python examples/playground_3d.py                # → http://localhost:8000/examples/playground_3d.html
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
  multidrone.py  ★ S5: MultiDroneEnv — N drones vs one site; sector fleet spawn,
                 anonymous detections, per-target witness blocks
  wrappers.py    DetectionRecorder (JSONL+Parquet) · QAChecker · DistributionMonitor
  stone_soup.py  ★ standard tracking baseline: detections → Stone Soup EKF/GNN
                 → graded tracks (single + multi-target with Hungarian grading)
  metrics.py     RMSE / latency / continuity / ID accuracy vs witness (adapter)
examples/
  demo.py            end-to-end single-episode demo (eval + record + plots)
  generate_dataset.py train/val/test dataset builder
  evaluate.py        batch evaluation of the tracking baseline
  run_stonesoup.py   single-episode tracking benchmark CLI
  run_multidrone.py  multi-drone fleet benchmark CLI (CSV + identity switches)
  interactive.py     terminal flight control
  playground_3d.py   3D playground v2 server (single + swarm bridge)
  playground_3d.html Three.js scene · PPI scope · chase cam · fleet HUD
docs/
  PLAYGROUND_RESEARCH.md  design study: how to build an outstanding playground
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
EKF track-while-scan baseline (removed in v0.2.0). A head-to-head on
identical episodes showed the two agree at close range but diverge under
sensor heterogeneity: a greedy associator let RF's wide 4° gate capture the
track on 86/201 ticks (starving the radar fix, fragmenting identity), while
Stone Soup's global 2D assignment held 99.5 %. Independent, community-tested
components make benchmark numbers citable — and swap-in upgrades (UKF, JPDA,
MHT, IMM) are configuration changes, not rewrites.

## Multi-drone (S5): when position is easy and identity is not

`skygym/multidrone.py` adds `MultiDroneEnv`: N drones (2–8) fly scripted
behaviours around one sensor site. Sensors observe the **whole fleet at
every scan** (`Sensor.poll_multi`) and emit **anonymous detections** — no
detection is tagged with the drone that produced it. The witness channel
carries one truth block per target (`gt.targets[]`), so recovering the
detection↔target correspondence *is* the benchmark task.

Fleet spawn is sector-based (drone k at ~360°·k/N) with behaviour-dependent
radii, so trajectories cross and merge by construction. `run_episode_multi`
in `stone_soup.py` grades every tick with a **Hungarian (global) assignment**
between tracks and truth targets, gated at 250 m, and reports per-target
coverage, RMSE and **identity switches**.

```bash
python examples/run_multidrone.py --n 3 --episodes 4 --duration 20
python examples/run_multidrone.py --n 2 --duration 60 \
       --mix approach,approach --radius 300,320   # collision course
```

Three regimes, same tracker (EKF + GNN2D, fusion, seeds 20260902/31337/555+):

| Regime | Geometry | Tracked | RMSE | ID switches |
|---|---|---|---|---|
| **Spread fleet** (3 drones, 20 s ×4) | min separation ≈ 600–1300 m | **99.9 %** | 3.9 m median | **0** |
| **Merge** (3 converging, 30 s ×3) | min separation ≈ 190 m | **99.9 %** | 2.1 m median | **0** |
| **Collision course** (2 converging, 60 s ×3) | drones overlap at the asset (0.1 m apart!) | **100 %** | 2.0 m median | **14–36 / episode** |

Reading: with well-separated targets, global assignment is essentially a
solved problem — one stable track per target across the whole episode, even
radar-only. But when two drones physically overlap, position tracking stays
perfect (~2 m) while **identities churn** — the two tracks swap labels every
time the crossing geometry defeats the per-scan assignment. That is the
single clearest motivation for identity-aware association (learned embeddings,
MHT, JPDA with track management) — and it is only visible because the env
keeps both the truth and the detections honest.

Second honest number: the naive single-point initiator confirms
~6–12 false tracks per episode from clutter (they survive on RF bearings).
The S5 CSV therefore also serves as a false-track benchmark
(`nearest_truth_m` column makes them easy to score).

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

`pytest` runs all gates in seconds (34 tests):

| Gate | Test |
|---|---|
| Geometry roundtrip | spherical ⇄ cartesian < 1e-6 deg/m |
| Seed reproducibility | same seed → bit-identical detection stream |
| Noise matches spec | Monte-Carlo σ within 25 % of configured σ, unbiased |
| Pd physics | SNR ∝ 1/r⁴, Pd collapses with range & small RCS |
| GT separation | no witness keys in obs, enforced per step |
| Tracker honesty | Stone Soup bridge: track initiated, bounded RMSE, sane ID readout |
| S5 fleet | poll ≡ poll_multi · anonymous dets scale with fleet · deterministic spawn · multi grading sane · recorder per-target labels |

## 3D playground v2

`python examples/playground_3d.py` → http://localhost:8000/examples/playground_3d.html

The playground is a real instrument panel, not a tech demo. The JS side only
renders and sends acceleration commands — physics, sensors and recording stay
in Python under the untouched Gymnasium contract. Modes: **Auto 20 s / 40 s**
(autopilot), **Swarm 20 s** (2–4 drones via `MultiDroneEnv`), **Manual ∞**
(WASD/QE + joystick, flies drone 1).

- **Two layers, never mixed** — the 3D scene renders truth (fleet, trails,
  sensor-volume wireframes at the real 5/6/8 km sensor ranges); the
  **PPI radar scope** (bottom-left) renders measurements: north-up polar,
  2 km range rings, rotating sweep, afterglow blips, clutter in amber,
  RF bearings as rim arcs, per-drone truth ghosts. Watching the PPI is the
  fastest way to understand why RF cannot give range.
- **Camera grammar** (`C`) — Orbit · Chase (damped velocity-vector follow with
  look-ahead) · Tower (the sensor operator's frame) · Top (tactical north-up).
- **Swarm-native HUD** — per-drone colour, trail, PPI ghost and a witness
  fleet table (behaviour / range / az / alt) that never shifts layout.
- **Smooth motion** — fixed-step accumulator with catch-up batching and
  render-time interpolation; instanced detection pools (no per-frame
  allocation); scale toggle 40× / real 0.5 m for honest screenshots.
- **One-click export** — JSONL or CSV; swarm CSV writes one labelled row per
  target per tick.

The engineering rationale — what Foxglove/Cesium/PPI scopes do, why v1 felt
bad, the seven design principles, the latency budget and the verdict
criteria for the next iteration — is written up in
[`docs/PLAYGROUND_RESEARCH.md`](docs/PLAYGROUND_RESEARCH.md).

## Roadmap

1. **Identity-aware association (S5 follow-up)** — the collision-course
   regime (100 % tracked, dozens of ID switches) is the target: track-state
   embeddings, MHT or JPDA via Stone Soup config swaps; a learned associator
   trained on S5 (lie, truth) pairs.
2. **Fidelity (S6)** — PyBullet backend (attitude dynamics) and/or
   vectorised backend behind the same Gymnasium API; image-based ID branch.
3. **Learned fusion** — the reason this benchmark exists: train a policy on
   the (lie, truth) pairs to beat the Stone Soup baseline where classical
   filters die (far range, bearing-only regimes, dense clutter, false-track
   suppression).
4. **Live tracking overlay in the playground** — run `stone_soup.py` inside
   the playground server and draw tracks vs truth ghosts in real time.
5. **PettingZoo-style control** — manual control of the whole fleet
   (adversarial evasion vs the tracker).

## Changelog

- **v0.3.0** — S5 multi-drone: `MultiDroneEnv` + `Sensor.poll_multi()`
  (anonymous fleet detections), `run_episode_multi` with Hungarian grading,
  identity-switch metric; `examples/run_multidrone.py` benchmark CLI;
  playground v2 (PPI scope, camera grammar, instanced detections, swarm
  mode, per-target witness HUD, swarm CSV export); design study in
  `docs/PLAYGROUND_RESEARCH.md`; 34 tests.

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
detection/tracking algorithm development. Tests: 34 passing
(`python -m pytest`).
