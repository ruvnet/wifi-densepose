---
name: consumer-nlos
description: Plan, review, and verify RuView consumer time-of-flight non-line-of-sight research, including raw-transient reproduction, motion-induced aperture, CSI fusion, iOS boundaries, and real-hardware acceptance evidence.
---

# Consumer NLOS research and verification

Use this skill to plan or review RuView's consumer time-of-flight non-line-of-sight
research. It is an advisory contributor workflow. It does not capture hardware,
authorize sensing, replace the Rust runtime, or turn a simulator result into a
hardware claim.

## Non-negotiable measurement boundary

The MIT method consumes a transient photon-count histogram for each SPAD zone.
Do not substitute an ARKit scene-depth map, mesh, conventional point cloud,
WiFi CSI frame, or synthetic replay. Those inputs may provide pose, context,
RF likelihoods, or deterministic software tests, but they do not contain the
delayed optical multipath samples needed for NLOS inversion.

The supported reproduction target is a pinned, upstream-compatible ST assembly
with verified raw or compact-normalized histogram access. Record the actual
board/silicon/firmware/API identity rather than inferring it from the kit name.
Pin the full upstream `sidsoma/consumer-nlos` commit, scoped enrollment
reference, firmware/configuration, calibration, capture manifests, analysis and
RuView commit before comparing results.

## Start here

```bash
npx @ruvnet/ruview nlos plan
npx @ruvnet/ruview nlos verify --repo .
```

The first command returns the four gated phases. The second performs static
inspection only. `STATIC_DISCOVERY_ONLY`, an available-build pass, or a skip is
not full Gate A, a research result, or proof of live NLOS behavior.

Run available build gates explicitly:

```bash
npx @ruvnet/ruview nlos verify --repo . --run-builds
```

Repository selection and build execution are CLI-only. The MCP tools remain
read-only advisory surfaces and reject `repo` and `run_builds` so a remote tool
call cannot choose an arbitrary checkout or launch its toolchains.
Run builds only in a trusted checkout: Cargo, Swift, Jest and Expo may execute
repository code. The verifier scrubs the child environment and redacts output,
but it is not a sandbox.

Expected optional surfaces:

1. `v2/crates/ruview-nlos` for typed transient frames, motion-induced aperture
   state, calibrated likelihoods, fusion, and `ruview.nlos.track.v1` output.
2. `ui/ios-nlos` for the pure Swift contract plus the public-API ARKit adapter.
3. `ui/mobile` for authenticated live tracks and deterministic `SYNTHETIC`
   replay. The browser does not claim direct access to iPhone photon histograms.

An absent optional surface is an explicit skip. Once its marker exists, missing
required files or a mismatched schema is a failure.

## Upstream reproduction

1. Freeze the experiment protocol before capture. Record hypotheses, endpoints,
   exclusions, randomization, scene strata, sensor configuration, target class,
   calibration procedure, and statistical analysis.
2. Reproduce the upstream tracker before changing its state model. Use verified
   live transient histograms, external ground truth, an opaque line-of-sight blocker,
   and a controlled relay surface. Start with the upstream retroreflective
   target before diffuse targets.
3. Preserve zone, timing bin width, counts, ambient signal, sensor timestamp,
   monotonic sequence, wall geometry, pose, calibration identity, and capture
   provenance. Never flatten a transient frame to a depth point before NLOS
   preprocessing.
4. Measure end-to-end track update rate. The preregistered reproduction gate is
   at least 27 Hz, meaning within ten percent of the reported 30 Hz target.
5. Treat published accuracy as `CLAIMED` until reproduced. Generated fixtures
   are `SYNTHETIC`; captured replay remains distinct `REPLAY`/L1 evidence. Use
   `MEASURED` only with a named live capture manifest, L2 level and external
   ground-truth reproducer.

## CSI fusion experiment

Use paired sequences: the same capture split, initial state, target trajectory,
and scoring code for LiDAR-only and LiDAR-plus-CSI arms. Keep calibration and
threshold selection inside the training/calibration partition. The test
partition remains sealed until LiDAR-only, CSI-only, and fused arms are frozen.

The retained fusion model must satisfy both conditions:

1. The LiDAR-only reproduction arm produces at least 27 end-to-end updates per
   second on live hardware.
2. Fusion reduces mean target position error by at least 25 percent **or**
   reduces lost-track rate by at least 25 percent relative to LiDAR-only over at
   least 100 paired sequences.

Do not count a replay, simulator, duplicated frame, or LiDAR-derived pseudo-CSI
feature as independent RF evidence. Report both endpoints even when only one is
the success endpoint. Stratify by reflectivity, range, motion, relay surface,
and RF geometry so a pooled gain cannot hide a failing domain.

## Evidence gate

Create a repository-contained JSON record conforming to
`ruview.nlos.acceptance.v1`. It separates `LIVE_HARDWARE` provenance,
`MEASURED` claim tag and L2 evidence, and requires external ground truth, frozen
protocol, actual sensor/transient identity, nonzero artifact/review digests,
zero synthetic/replay frames, independent CSI with CSI-only ablation, at least
100 paired sequences, full paired position-endpoint coverage, shared lost-track
exposure, recomputable aggregate metrics (including offered rate and fused frame
loss), adjusted confidence intervals and frozen numeric guardrails. The v1 live
gate accepts an enrolled external ST VL53L8-series sensor label; it cannot be
used to attest an iPhone sensor.

Then require the real-hardware gate:

```bash
npx @ruvnet/ruview nlos verify --repo . \
  --evidence-file evidence/nlos/acceptance.json \
  --require-research-pass
```

The verifier fails closed on malformed, out-of-repository, oversized, replay,
or synthetic evidence. A passing JSON record is an integrity and arithmetic
check, not an independent audit of the physical capture; reviewers must inspect
the immutable capture manifest and ground-truth synchronization evidence.

## Native and web iOS boundary

Apple's documented ARKit surfaces expose processed scene depth, smoothed depth,
pose, and reconstructed meshes. Until Apple documents access to the required
per-zone transient histograms, use those APIs only for line-of-sight context,
pose, display, or transport. A native app may consume authenticated, versioned RuView tracks from
the external sensor pipeline. A web client may consume an authenticated live
stream or a visibly watermarked replay. Neither is evidence that the built-in
iPhone LiDAR ran the MIT inversion.

## Privacy and security

1. Record approved purpose, space, time, controller and retention. Obtain all
   required operator/participant notice and consent, provide pause/withdrawal,
   show a persistent indicator, and prohibit hidden sensing.
2. Before any live promotion, bind every frame to an authenticated sensor,
   session, calibration, tenant/workspace, coordinate frame and monotonic
   sequence. The G0 scaffold does not yet provide every binding. Reject stale,
   duplicate, future, oversized, or coordinate-frame-incompatible inputs before
   fusion.
3. Keep raw transients and CSI local by default. Export bounded track hypotheses,
   covariance, provenance, and expiration unless the frozen protocol requires
   raw capture retention under an approved data policy.
4. Never infer identity. Use random, session-scoped track identifiers and short
   expiry. Unknown quality stays unknown; it never falls back to a live claim.
5. RuVector, RuField, and WorldGraph carry state and evidence. They do not create
   sensing authority or compensate for missing photons.

## Authoritative references

1. [Nature paper](https://doi.org/10.1038/s41586-026-10502-x)
2. [Author manuscript](https://arxiv.org/html/2605.17865v1)
3. [MIT project and reported 30 Hz demonstration](https://cornar.media.mit.edu/)
4. [Upstream implementation](https://github.com/sidsoma/consumer-nlos)
5. [ST VL53L8CH histogram interface](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html)
6. [Apple scene-depth sample](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth)
