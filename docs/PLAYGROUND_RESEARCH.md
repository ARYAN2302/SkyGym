# Research: How to Build an Outstanding 3D Drone-Recon Playground

*Design study behind the SkyGym playground v2 (September 2026).*

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
