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

# 6) 3D playground (Three.js · RC-stick quad flight · FPV · swarm · PPI scope · live Stone Soup score)
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
  playground_3d.py   3D playground v5 server (single + swarm bridge + live tracker)
  playground_3d.html Three.js scene · RC-stick flight · FPV · PPI scope · EO feed · timeline
  (all tracker examples import the single standard path: skygym.stone_soup)
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

## 3D playground v5

`python examples/playground_3d.py` → http://localhost:8000/examples/playground_3d.html

The playground is a live instrument, not a menu-driven demo: it **boots
straight into a running 3-drone swarm** (P8 — alive at t = 0, no dead scene,
no mode hunt) and makes the drones impossible to lose (P9 — every drone
carries a screen-space label with range readout that clamps to the screen
edge as a pointing arrow when off-camera). Physics, sensors and recording
stay in Python under the Gymnasium contract; the client composes RC sticks.

- **Real quad flight (v4)** — manual flight is an **angle-mode quadcopter**,
  not a sliding point: sticks command pitch/roll tilt, yaw rate and climb
  rate; attitude follows through a first-order lag; tilt becomes
  acceleration via `g·tan(tilt)` rotated by yaw (`skygym/flight.py`,
  control mode only — data-mode trajectories and benchmarks are untouched).
  Keys **W/S** pitch, **A/D** yaw, **Q/E** climb; **gamepad Mode-2** analog
  sticks (12% deadzone); the drone mesh banks, pitches and spins its props
  from server-truth attitude, and the fleet table shows live HDG.
- **Possess any drone + FPV** — click the canvas or press **1–4** to take
  the stick of drone k *while the rest of the fleet keeps its autopilot*
  (`MultiDroneEnv.step(action, control_idx=k)`). Camera **C** cycles
  Chase · **FPV** (nose cam with Pointer-Lock mouse-look, ESC releases,
  own mesh hidden, crosshair overlay) · Orbit · Tower · Top; Chase and the
  HUD follow whoever you possess. The HUD marks your row **YOU**.
- **Modes** — Swarm (autopilot fleet, auto-looping episodes with a fresh
  seed each time), Solo auto (single drone), **Fly single** (manual, just
  you and one drone — no fleet), Fly-in-swarm (possess any of 1–4 via
  click or keys 1–4, **R** hands the drone back to its autopilot).
- **Live Stone Soup score (v5, Phase C)** — an `OnlineMultiTracker`
  (`skygym/stone_soup.py`) runs the standard EKF(CV)+GNN2D baseline
  tick-by-tick inside the playground server and grades every tick against
  the witness with the same Hungarian assignment as the batch benchmark:
  top-pill tracked %, running RMSE, ID switches and false-track counters
  (live + cumulative — RF bearings keep clutter-born tracks alive, and the
  cumulative count shows it honestly).
- **Timeline scrub (v5, Phase C)** — every tick of the session is buffered
  in the client; drag the bottom timeline to pause and replay any past
  moment (drones + detections rendered from the recording), **LIVE** to
  jump back to real time. What you scrub is what you flew.
- **EO/IR gimbal feed (v5, Phase B)** — a picture-in-picture white-hot IR
  view from the sensor site (60° FOV = `EOCfg`), cued to your drone,
  graded with scanlines + vignette. At 1 km the drone is exactly the
  faint blob the real EO channel reports — the feed *shows* the sensing
  problem instead of hiding it.
- **Terrain + night sky (v5, Phase B)** — procedural heightfield (flat pad,
  hills mid-range, ridges beyond the 2.5 km ops area) and a gradient
  night-sky dome with sun glow and stars; **Y** toggles back to the flat
  engineering grid.
- **Behaviour commanding (v5)** — the fleet table's **CMD** column switches
  any autopilot drone's behaviour MID-EPISODE (hover / approach / orbit /
  waypoint_cruise / serpentine / egress) via
  `MultiDroneEnv.set_behaviour(k, name)` — parameters are rebuilt from the
  drone's current state, so orbits start at the current range and
  serpentines weave along the current heading. Trigger an evasion and
  watch the live Stone Soup score react.
- **Mission timer & pace (v5)** — set the episode duration (5–600 s), the
  timer pill carries a progress bar that turns red near the end, and the
  pace selector runs the same physics at 0.5× / 1× / 2× / 4× wall-clock.
- **Session-faithful exports (v5)** — exports carry ONLY what this session
  produced, never a re-simulation: **CSV** = per-target witness (gt-prefixed
  columns) + your stick actions (`act_*`) + the live tracker score
  (`ss_*`) + detection counts; **Dets CSV** = every raw detection row in
  the exact Stage-0 dataset schema (`episode_id, t, sensor, az_deg, …,
  p_unknown`); **JSONL** = full frames `{t, gt, obs, action, score}`.
- **Two layers, never mixed (P1)** — the 3D scene renders truth (fleet,
  trails, sensor-volume wireframes at the real 3/5/8 km sensor ranges); the
  **PPI radar scope** renders measurements: north-up polar, 2 km range
  rings, rotating sweep, afterglow blips, clutter in amber, RF bearings as
  rim arcs (RF never gives range), per-drone dashed truth ghosts. The gap
  between the layers is the research problem, made visible.
- **Camera grammar (P3)** — **Chase (default**; damped velocity-vector
  follow with look-ahead) · Orbit · Tower (the sensor operator's frame) ·
  Top (tactical north-up). One keypress (`C`) apart, frame-rate-independent
  smoothing `α = 1 − exp(−λ·dt)`.
- **Honest failure** — open the HTML from disk (no server) and you get an
  explicit banner with the exact command to run, instead of a silently dead
  page.
- **Swarm-native HUD (P7)** — per-drone colour, trail, PPI ghost and a
  witness fleet table (behaviour / range / az / alt / TX) that never shifts
  layout.
- **Smooth motion (P4/P5)** — fixed-step accumulator with catch-up batching
  and render-time interpolation; instanced detection pools and bearing-ray
  segments (zero per-frame allocation); scale toggle 40× / real 0.5 m for
  honest screenshots.
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
4. ~~Live tracking overlay in the playground~~ — **shipped in v0.5.0**
   (`OnlineMultiTracker` live scoreboard + timeline scrub).
5. **PettingZoo-style control** — manual control of the whole fleet
   (adversarial evasion vs the tracker).

## Changelog

- **v0.5.0** — playground v5 (Phases B + C) and live-scoring core:
  `skygym/stone_soup.py` gains `OnlineMultiTracker` — the batch
  EKF(CV)+GNN2D pipeline made tick-by-tick (same conventions, same
  Hungarian grading) powering the playground scoreboard; `n_false` counts
  live never-assigned tracks (batch-consistent), `n_false_cum` counts
  every confirmation this run. `skygym/multidrone.py` gains
  `set_behaviour(k, name)` — mid-episode behaviour switching with
  continuous parameter hand-off (hover holds here, orbit starts at current
  range, serpentine keeps the heading; scripted data generation
  untouched). Playground: EO/IR white-hot gimbal feed, procedural terrain
  + night sky, live Stone Soup scoreboard, timeline scrub of the recorded
  session, per-drone behaviour CMD selects, R to release a possessed
  drone to autopilot, Fly-single mode (manual, no fleet), mission
  duration input + 0.5–4× pace control, session-faithful exports
  (witness CSV with `gt_`/`act_`/`ss_` columns, raw-detections CSV in the
  dataset schema, JSONL with action + score per frame). CSV truth columns
  renamed with an explicit `gt_` prefix — they were always witness, never
  tracker output. Live tracker timestamps are env-tick times: the
  scoreboard is not bit-identical to the batch replay and does not claim
  to be.
- **v0.4.0** — playground v4: real quadcopter flight. New angle-mode flight
  controller (`QuadCfg`/`QuadAttitude`/`sticks_to_accel`, control mode only):
  RC sticks → tilt/yaw-rate/climb-rate commands → first-order attitude lag →
  `g·tan(tilt)` acceleration in ENU → the same drag/speed-limited plant.
  Client: possess any drone (keys 1–4, `MultiDroneEnv.step(control_idx=)`),
  body-frame stick mapping, gamepad Mode-2 with deadzones, FPV nose camera
  with Pointer-Lock mouse-look + crosshair, realistic X-frame quad mesh with
  spinning props and server-truth attitude, HDG column/HUD, data-mode
  benchmark contract byte-identical (no attitude in data-mode gt). 44 tests.
- **v0.3.1** — playground v3 client rebuilt from scratch: boots into a live
  3-drone swarm (no dead scene), every drone carries an always-on screen-space
  label with edge-clamped off-screen arrows and range readout, click-to-fly
  (click canvas = manual control of drone 1), Chase is the default camera, and
  an explicit banner appears if the API is unreachable (no more silently dead
  page). Removed the terminal `interactive.py` example (superseded by the
  playground).
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
  terminal interactive example + 3D playground.

## Status

MIT licensed. Defensive research tooling — synthetic data for
detection/tracking algorithm development. Tests: 34 passing
(`python -m pytest`).
