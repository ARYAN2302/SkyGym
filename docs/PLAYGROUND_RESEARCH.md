# Research: How to Build an Outstanding 3D Drone-Recon Playground

*Design study behind the SkyGym playground (September 2026). Part 1: the v2
principles behind the shipped v2/v3 client. Part 2: the second research pass —
how truly interactive 3D web sims work, the v3 gap analysis, and the v4 plan.*

This document distils what the best-in-class tools do, why the v1 playground
felt bad, and the concrete engineering rules the v2 client follows. It is
written so the next iteration can be judged against explicit principles
instead of taste.

---

## 1. What the reference class does

| System | What it gets right | What we borrowed |
|---|---|---|
| **Foxglove Studio** (robotics) | Panel grammar: one 3D scene + many small specialised panels (plot, table, log); strict *message timestamp* discipline; renders at 60 fps while data arrives at arbitrary rates | Decouple *sim time* (10 Hz server) from *render time* (60 fps); never block the render loop on the network |
| **CesiumJS / Resium** digital twins | Real-world scale honesty: things are drawn at true metres; the camera machinery (fly-to, follow, look-at) is a first-class citizen | Real-scale site + sensor volumes; camera modes as an explicit, switchable grammar |
| **ATC / radar PPI scopes** | 70 years of HCI refinement: north-up polar display, range rings, sweep line, afterglow blips, truth-vs-track symbology | A real PPI panel: rings at sensor-specific ranges, sweep at the radar's PRF, clutter in amber, targets in green, truth ghost in cyan |
| **DJI FPV / flight HUDs** | Velocity-vector (flight-path) marker, altitude/range ladder, status pills that never move | Fixed HUD anchor points; range/velocity readouts that update in place (no layout jitter) |
| **Gazebo / Isaac Sim** | Physics/render decoupling and deterministic replay; camera interpolation between physics steps | Fixed-step accumulator on the client; interpolation between the last two server states |
| **Track-and-trace consoles** (MOT challenge viewers) | Always visualise the *association*: which detection went to which track; ID switches shown as colour changes | Detections carry sensor colour; assigned vs unassigned distinguishable; the PPI shows what the tracker sees, the 3D scene shows truth |

## 2. Why v1 felt bad — a diagnosis

1. **One fixed orbital camera** looking at a 2 km world from a human-scale
   viewpoint: the drone was a speck, so it was inflated 96× — which then made
   sensor geometry (FOV, beamwidth, ranges) visually meaningless.
2. **No sensor point of view**: the three sensors produce fundamentally
   different geometries (polar radar, pixel EO, RF bearings) but the UI showed
   all detections as identical 3D spheres floating in space.
3. **Motion was true (10 Hz steps) but looked wrong**: straight
   request→render made the drone jump every 100 ms under network jitter.
4. **Detections were re-meshed every frame** (new geometry objects per
   detection per frame) — GC pressure and frame drops as soon as clutter rose.
5. **No multi-target story**: the whole UI assumed exactly one drone.

## 3. The seven principles of playground v2

**P1 — One world, two layers.** The 3D scene renders *truth* (drone, trail,
site, sensor volumes). The PPI panel renders *measurements* (blips, clutter).
Never mix them: a radar blip is not a drone. This separation is the whole
point of SkyGym (obs vs witness) made visible.

**P2 — Real scale, honest geometry, visual affordances.** Physics stays in
real metres. Radar max range (8 km), EO FOV wedge and RF range ring are drawn
as wireframes so the user *sees* why detections appear/disappear. The drone
gets a *scale boost* (default 40×) as an explicit toggle — visibility, not
deception — and a "real scale" mode for honest screenshots.

**P3 — Camera grammar.** Four named cameras, one keypress (`C`) apart:
*Orbit* (free), *Chase* (damped follow: position `lerp(λ≈3/s)` behind the
velocity vector with look-ahead `pos + v·0.8 s`, so turns read as banking),
*Tower* (site-mounted, the sensor operator's view — this is the sensor's
actual frame of reference), *Top* (tactical north-up). Chase smoothing is
frame-rate independent: `α = 1 − exp(−λ·dt)`.

**P4 — Fixed-step sim, free-running render.** The client keeps a wall-clock
accumulator and requests `n = floor(acc/100 ms)` steps per request (catch-up
capped), then *interpolates* the drone transform between the previous and
current server state. Render never waits on fetch; fetch never double-fires
(reentrancy guard, kept from v1.1).

**P5 — Zero per-frame allocation.** Detections render through two
`InstancedMesh` pools (target-dets, clutter) with per-instance colour;
RF/mono-EO bearing-only reports render as pooled line segments from the site
(a bearing is a *ray*, not a point — drawing it as a point was the core lie
of v1). Trails are pre-allocated `BufferGeometry` rings written in place.

**P6 — The PPI is the tracker's eye.** Canvas 2D, north-up, ground-range
projection (`r·cos(el)`), range rings every 2 km to the radar's 8 km max,
sweep line rotating at a stylised 1 rev/s, blips fading like scope afterglow
(~2 s), clutter amber vs target green by the *sensor's own* clutter flag
(which real trackers must not trust), and the truth ghost in dashed cyan.
RF bearings are rim arcs. This single panel teaches more about the data than
any paragraph of the README.

**P7 — Swarm-native.** Every drone is a first-class object (own colour,
trail, halo, PPI ghost, HUD row) driven by `gt.targets[]`. Manual control
still flies drone 1 only; the fleet keeps its autopilot — exactly like the
S5 benchmark data path.

## 4. Latency & performance budget

- Server step: 10 Hz (one env `dt` per step). One `/api/step` round trip on
  localhost ≈ 2–5 ms; the batch catch-up (1–8 steps) absorbs any jitter.
- Render: 60 fps target; scene graph static except instance matrices and
  trail buffers → measured < 3 ms/frame on integrated GPUs for ≤ 4 drones.
- GC: no `new Vector3` inside the render loop (module-scope temporaries).
- Payload: per step ≈ 40 dets × 11 floats + 4 targets × 6 floats ≈ 4 KB JSON
  → ~40 KB/s. Fine for localhost; a binary path (MessagePack/WS) is the
  documented next step if this ever leaves the loopback.

## 5. Deliberately out of scope (for now)

- WebSocket push (HTTP+accumulator is sufficient at 10 Hz and keeps the
  server stdlib-only — `http.server`, zero dependencies).
- Live Stone Soup overlay in the browser (tracking runs offline in the
  benchmark; wiring the tracker into the playground server is future work,
  the S5 grading CSV is the offline equivalent).
- Textures/skybox: photorealism is not the goal; legibility is.

## 6. Verdict criteria for the next iteration

1. Can a first-time user explain *why* the RF channel cannot give range
   after 30 s of watching the PPI?
2. During a 3-drone merge, can they see the moment identity becomes
   ambiguous (PPI: blips cross; 3D: truth ghosts cross)?
3. Is motion smooth under CPU load (interpolation, not response)?
4. Does any readout ever shift layout while updating? (It must not.)

---

# Part 2 — Toward a *truly* interactive 3D playground (September 2026 research pass)

Part 1 shipped: the v2/v3 client follows P1–P7 and passes all four verdict
criteria on a good day. But v3 is still "watch a simulation with some input" —
not a *playground*. This part records a second research pass (how the best
browser-native interactive 3D systems actually work), a gap analysis against
v3, and a phased build plan.

## 7. Reference patterns — how truly interactive 3D web sims are done

| Pattern / system | What it does | Takeaway for SkyGym |
|---|---|---|
| **Foxglove WebSocket protocol** | Server *pushes* structured telemetry over one two-way socket; client renders at 60 fps while data arrives at any rate. Became the de-facto "RViz in the browser" standard. (Foxglove Studio itself went closed-source in 2024 — an argument for owning our viewer.) | Own the protocol, keep it tiny: one socket, two message types (frame, command). |
| **Gambetta netcode model** (client-server architecture series) | Server-authoritative sim at fixed step; client interpolates entities ~100 ms in the past; *only the possessed entity* gets client-side prediction. | v3 already interpolates. Prediction only needed for the drone you fly — and only if input latency ever feels bad. |
| **Pointer Lock API** (MDN; W3C Pointer Lock 2.0, active spec 2026) | Raw relative mouse movement while the cursor is hidden — the only correct basis for mouse-look flight/FPV cameras. ESC release is observable. | The missing primitive for a *real* fly-the-drone camera. |
| **Gamepad API** (MDN controls guide; Windows Flight Arcade; jsRC browser RC sim) | `navigator.getGamepads()` polled per frame; analog sticks + deadzones; a small helper class maps axis indices. Mode-2 RC layout: left stick = throttle/yaw, right = pitch/roll. | RC-style analog flight is the domain-authentic control scheme; keyboard stays as fallback. |
| **Rerun viewer** (Rust/wgpu/egui; web via WASM) | Log-first: record once, then *scrub the timeline* — pause, step, replay any window; URL-based sharing of a view. | Our `_rec` buffer + deterministic seeds already hold the data; expose it as a timeline. |
| **CesiumJS** | Time-dynamic geospatial: global 3D terrain/buildings via 3D Tiles, first-class time-dynamic telemetry. Heavyweight (its own terrain streaming stack). | Right tool if we ever want *real Earth* context. For a 2–8 km site, a DEM heightmap patch gives 90% of the feel at ~0% of the cost. |
| **Three.js render-to-texture** (`WebGLRenderTarget`) | Render a second camera's view into a texture and display it on a panel/mesh — "a TV showing another camera". | The EO gimbal feed: render the scene *from the drone's camera* — the single most honest way to show what the EO witness actually sees. |
| **Three.js ecosystem position** | Three.js dominates web 3D (1.8–5M weekly npm downloads vs ~11k for Babylon). Babylon bundles more framework; Three stays a rendering library — right size for our single-file, no-build client. | Stay on Three.js. No rebuild. |
| **WebGPU (status 2026)** | Universal browser support since late 2025; `WebGPURenderer` production-ready in Three r171+; but real-world reports of regressions vs WebGL in some scenes. | Not a goal. Our scene is tiny (< 3 ms/frame). Optional flag someday; WebGL stays default. |

## 8. Gap analysis — v3 vs the reference class

| # | Gap | Why it matters | Severity |
|---|---|---|---|
| G1 | **World-frame flight**: WASD applies ENU acceleration — pressing W pushes *north*, never *forward*. No yaw, no attitude. | Feels like moving a chess piece, not flying. Kills the "interactive" promise outright. | Critical |
| G2 | **Possess drone 1 only**; the other fleet drones are scenery. | The swarm is the product (S5); you must be able to *be* any drone. | Critical |
| G3 | **No pointer-lock mouse-look, no FPV camera.** Cameras are external only. | No first-person "you are the drone" moment; Tower cam is the closest thing and it is passive. | High |
| G4 | **No analog device support** (Gamepad API). Virtual joystick is binary-ish and on-canvas. | RC pilots and gamepad users are the natural audience. | Medium |
| G5 | **No sensor-perspective rendering** — the EO camera is the star witness (0.08° bearing, blob-confused ID) but the user never *sees* its viewpoint. | The core pedagogy of SkyGym (obs ≠ truth) is shown in 2D (PPI) but not in 3D. | High |
| G6 | **Flat grid Earth.** No terrain, no sky, no horizon — at real scale the scene reads as abstract. | Perceived "3D-ness" is mostly terrain/sky/horizon, not mesh count. | Medium |
| G7 | **Client-pull transport.** 1–8-step HTTP batching works at 10 Hz localhost, but each batch is a request; no server push, no time-scale > 1× without request storms. | Fine today; blocks smooth fast-forward and non-localhost use. | Medium (deferred OK) |
| G8 | **No timeline.** Pause exists (client-side accumulator stop) but no step-one-frame, no scrub-back over `_rec`, no replay sharing. | The sim is deterministic and recorded — not exposing that is leaving Rerun's best feature on the table. | Medium |
| G9 | **No reason to fly well.** Nothing scores your flying against the tracker. | The adversarial loop (evade the EKF) is the game — and hard human-flown episodes are exactly the data a future learned tracker needs. | High (product) |

## 9. Design decisions for v4

**D1 — Flight model: body-frame, attitude-bearing.**
`W/S` pitch (forward/back accel along the drone's heading), `A/D` roll-yaw
turn, `Q/E` climb/descend; the drone mesh banks with turn rate (visual only —
flight.py stays the physics truth). Action stays a 3-vector; only the *frame*
and mapping change client-side (server contract untouched).

**D2 — Cameras gain two modes; possession for all.**
`1..4` keys possess drone *k* (label + HUD row flip to YOU). New cameras:
**FPV** (drone nose cam, pointer-locked mouse-look via Pointer Lock API, ESC
releases) and **Gimbal** (site-tracking camera rendered from the drone).
Existing Orbit/Chase/Tower/Top unchanged; Chase follows the possessed drone.

**D3 — Gamepad: Mode-2 RC sticks.**
Left stick = throttle/yaw, right = pitch/roll; 12% deadzones; keyboard remains
the fallback. Poll `navigator.getGamepads()` in the existing input poll — no
event plumbing needed.

**D4 — EO gimbal feed via `WebGLRenderTarget`.**
A second camera at the drone (gimbal aims at site or velocity vector), rendered
to a 512×288 target each frame (or every 3rd frame — 10 Hz is *honest*; the
real sensor is 30 Hz with 30 ms latency), shown as a panel + optionally on the
drone mesh as a "recorded" material. Cheap post shader: mono palette + noise +
vignette scaled by target range → the blob-growth ID story becomes visible.

**D5 — Terrain and sky, the 80/20.**
One DEM heightmap tile (open elevation data) around the site → 256×256
displaced plane + triplanar-ish vertex colour; Three.js `Sky` + fog; shadows
off by default (cost/benefit poor at this scale). Cesium stays documented as
the "real Earth" upgrade path, explicitly not now.

**D6 — Transport: defer, with a trigger.**
Stay on client-pull + accumulator until one of: non-localhost use, time-scale
> 2×, or > 4 drones. Then: one hand-rolled RFC 6455 WebSocket in stdlib
(~150 lines, single client, server→push frames + client→commands), JSON
frames first, binary (Float32 layout) only if profiling demands. Keep the
HTTP API as fallback — the recording/export path depends on it.

**D7 — Timeline: scrub the recording, Rerun-style.**
`Space` pause (exists) → add `,`/`.` single-frame step, drag-to-scrub over
`_rec` (client renders frame k on demand), and "restart from seed" everywhere.
Replay share = URL fragment `#seed/scenario/dur/t` (Rerun's URL-sharing idea,
trivial because everything is deterministic).

**D8 — The game: evade the tracker.**
Wire the Stone Soup fusion tracker into the playground server (it is already
importable): while you fly, the tracker tracks you live. Score HUD: seconds
tracked / ID switches / RMSE; " Mission failed: lost over 500 m for 3 s".
This is the adversarial loop the README roadmap promised — and every human-
flown evasion is a hard episode the benchmark suite can't generate.

## 10. Build order

| Phase | Contents | Acceptance test |
|---|---|---|
| **A — Fly** (G1,G2,G3,G4) | Body-frame controls, possession 1–4, FPV + pointer lock, gamepad Mode-2 | A first-time user banks the drone around the mast within 60 s without reading docs |
| **B — See** (G5,G6) | EO gimbal feed panel + noise shader, DEM terrain + sky | User can *say* why ID confidence collapses with range after 30 s of watching the feed |
| **C — Play** (G8,G9) | Tracker-in-the-loop score, timeline scrub, seed-share URL | User loses lock honestly at 4 km clutter 2.5×, sees it on the PPI *and* the score |
| **D — Scale** (G7, only if triggered) | stdlib WebSocket push, time-scale 1–4× | 4-drone swarm at 2× time-scale, < 10 ms main-thread frame time |

## 11. Verdict criteria (updated)

1. Part 1's four criteria still hold (PPI legibility, merge ambiguity visible,
   smooth under load, zero layout shift).
2. Does flight *feel* like flying (banking, momentum, heading-relative) within
   one minute, no docs?
3. Can the user explain the EO ID story from the gimbal feed alone?
4. Is losing the tracker *your fault* — visible, scoreable, and fun enough to
   retry three times?
