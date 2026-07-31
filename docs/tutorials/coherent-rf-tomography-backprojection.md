# Coherent Wideband RF Tomography: Simulating and Reconstructing with `wifi-densepose-sar`

A walkthrough of the `wifi-densepose-sar` crate (ADR-287): simulating
synthetic-aperture radar (SAR) style measurements and reconstructing a 3D
reflectivity image from them via delay-and-sum backprojection.

**Estimated time:** 30 minutes.

**What you will build:** A small Rust program that simulates a handheld
stepped-frequency radar sweep past a couple of point targets, reconstructs
a 3D image from the resulting complex measurements, and extracts a sparse
point cloud from it — then verifies the reconstruction's resolution
against closed-form theory.

**Who this is for:** Rust developers comfortable with basic signal
processing terminology (frequency, bandwidth, phase) who want to
understand what a coherent RF imaging pipeline actually computes, or who
are evaluating whether this crate is a useful building block for their own
radar-imaging research.

---

## Table of Contents

1. [What This Is (and Isn't)](#1-what-this-is-and-isnt)
2. [Prerequisites](#2-prerequisites)
3. [The Physics in Five Minutes](#3-the-physics-in-five-minutes)
4. [Your First Reconstruction](#4-your-first-reconstruction)
5. [Range Resolution: Why Bandwidth Matters](#5-range-resolution-why-bandwidth-matters)
6. [Cross-Range Resolution: Why You Need to Move the Antenna](#6-cross-range-resolution-why-you-need-to-move-the-antenna)
7. [The Antenna-Pose Coherence Budget](#7-the-antenna-pose-coherence-budget)
8. [Extracting a Point Cloud](#8-extracting-a-point-cloud)
9. [Benchmarking Your Own Scenario](#9-benchmarking-your-own-scenario)
10. [Where This Could Go Next](#10-where-this-could-go-next)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What This Is (and Isn't)

This crate exists because of a real question: could this repo build
something like [Applied Electrodynamics' WaveSight](https://www.ae-dyn.com/)
— a handheld device that images through walls using radio waves? The
honest answer, worked out in ADR-287, is **no, not as a hardware product**
— that needs a custom coherent RF front end, a calibrated antenna array,
and real-time reconstruction hardware, which is an 18–36 month, high
six-to-seven-figure hardware engineering program, not a software change.

What *is* useful to build, and what this crate is, is the **reconstruction
algorithm** such a device needs: given coherent, phase-preserving,
stepped-frequency measurements recorded from several known antenna
positions, recover the 3D locations of the things that reflected the
signal. That's a well-understood problem (synthetic-aperture radar,
ground-penetrating radar imaging, and microwave tomography all solve
versions of it) with textbook closed-form math behind it.

Every number in this crate comes from its own **synthetic forward
simulator** — there is no real radio hardware anywhere in this crate, and
none of its tests, benchmarks, or accuracy numbers say anything about how
well a real device would perform through a real wall. That's evidence
level **L0 (Synthetic)** in this repo's [ADR-282](../adr/ADR-282-ruview-ecosystem-positioning.md)
evidence ladder, and it stays L0 until (if ever) real wideband RF hardware
feeds this pipeline real measurements.

## 2. Prerequisites

- Rust 1.75+ (workspace MSRV), already set up if you can build the rest of
  this repo's `v2/` workspace.
- No special hardware. Everything in this tutorial runs from synthetic
  data.

```bash
cd v2
cargo test -p wifi-densepose-sar --no-default-features
```

If that passes (24 tests, 0 failed), you're ready.

## 3. The Physics in Five Minutes

A stepped-frequency radar sweeps `K` frequencies `f_0..f_{K-1}` across a
band of total width `B` (the bandwidth). At each of `M` antenna positions
`p_0..p_{M-1}` along a handheld sweep, it records one complex number per
frequency — amplitude and phase, not just amplitude, which is what makes
this "coherent."

For a point scatterer at position `x` with reflectivity `σ`, range
`R = |p_m - x|` from antenna position `m`, the forward model this crate
simulates is:

```text
y_{m,k} = sigma / R^2 * exp(-i * 4*pi * f_k * R / c)
```

`4*pi*f*R/c` is the two-way (round-trip) propagation phase; `1/R^2` is the
two-way free-space spreading loss. With several targets, the measurement
is just the sum of each target's contribution (superposition — this crate
never models multipath/interaction between targets, only free-space direct
paths).

**Reconstruction (backprojection)** inverts this: for every candidate
voxel `x` in a 3D grid, it multiplies each measurement by the *complex
conjugate* of the phase the forward model would have applied for a target
at `x`, then sums:

```text
I(x) = | (1/MK) * sum_m sum_k  y_{m,k} * R_{m,x}^2 * exp(+i * 4*pi * f_k * R_{m,x} / c) |
```

If `x` coincides with a real target, every term's phase correction exactly
cancels the phase the forward model applied — the sum adds up
constructively ("coherent gain"). At any other voxel, the phases are
essentially uncorrelated across the `(m, k)` grid and the sum averages
toward zero. That's the entire algorithm: matched filtering, done in 3D,
one voxel at a time.

## 4. Your First Reconstruction

Add `wifi-densepose-sar` to a scratch binary or run this in a workspace
example. It simulates two targets, reconstructs, and finds the brightest
voxel:

```rust
use wifi_densepose_sar::{
    backproject, linear_aperture, simulate_measurement, FrequencySweep,
    Point3, ScatteringTarget, VoxelGrid,
};

fn main() {
    // A 1-meter handheld sweep, 21 antenna positions along it.
    let poses = linear_aperture(
        Point3::new(-0.5, 0.0, 0.0),
        Point3::new(0.5, 0.0, 0.0),
        21,
    );

    // Sweep 2-6 GHz (4 GHz of bandwidth) in 32 steps.
    let sweep = FrequencySweep::new(2.0e9, 6.0e9, 32);

    // One target, 2 meters downrange, reflectivity 1.0 (arbitrary units).
    let target = ScatteringTarget::new(Point3::new(0.0, 2.0, 0.0), 1.0);

    // Simulate the measurement with a touch of noise (seeded -- rerunning
    // with the same seed gives byte-identical output).
    let measurement = simulate_measurement(&poses, &sweep, &[target], 0.01, 42);

    // Reconstruct a 21x21x21 voxel grid around where we expect the target.
    let grid = VoxelGrid::new(Point3::new(-0.3, 1.7, -0.3), 0.03, 21, 21, 21);
    let image = backproject(&measurement, &poses, &sweep, &grid);

    let (peak_location, peak_magnitude) = image.peak();
    println!("true target:          {:?}", target.position);
    println!("reconstructed peak:   {peak_location:?} (magnitude {peak_magnitude:.4})");
}
```

Run it and you should see the reconstructed peak within a couple of
centimeters of the true target position — well inside the voxel spacing
used here (3 cm). That's `tests/reconstruct.rs::single_point_target_reconstructs_at_its_true_location`
running live.

## 5. Range Resolution: Why Bandwidth Matters

How close together can two targets be *along the same bearing* (same
antenna, different distance) before they blur into one blob? The classic
radar answer: `ΔR = c / (2B)` — resolution improves with more swept
bandwidth, full stop. Carrier frequency, antenna count, and aperture
length don't enter into it at all.

```rust
use wifi_densepose_sar::resolution::range_resolution_m;

let dr = range_resolution_m(4.0e9); // 4 GHz swept bandwidth
println!("range resolution: {:.1} cm", dr * 100.0);
// -> range resolution: 3.7 cm
```

`tests/physics_validation.rs::range_separated_targets_resolve_only_beyond_range_resolution`
proves this isn't just a formula sitting in a doc comment: it forward-simulates
two targets 4x `ΔR` apart (they resolve into two distinct peaks) and 0.25x
`ΔR` apart (they merge into one), using the *same* `range_resolution_m`
call to pick the separations.

## 6. Cross-Range Resolution: Why You Need to Move the Antenna

A single antenna position, no matter how much bandwidth it sweeps, cannot
tell two targets apart if they're at the same range but different bearing
— all it measures is round-trip distance, which is the same for both. This
is exactly why "handheld... sweep the antenna around" matters: moving the
antenna across a synthetic aperture of length `L` gives you angular
information, with cross-range resolution:

```text
delta_CR ~= lambda * R / (2 * L)
```

— finer with a longer aperture, a shorter wavelength (higher carrier
frequency), or a closer target.

```rust
use wifi_densepose_sar::resolution::cross_range_resolution_m;

let short = cross_range_resolution_m(4.0e9, 0.05, 2.0); // 5cm sweep
let long = cross_range_resolution_m(4.0e9, 1.0, 2.0);   // 1m sweep
println!("5cm aperture:  {:.2} m cross-range resolution", short);
println!("1m aperture:   {:.2} m cross-range resolution", long);
// -> a 20x longer aperture gives 20x finer cross-range resolution
```

`tests/physics_validation.rs::cross_range_separated_targets_resolve_only_with_long_enough_aperture`
demonstrates this end-to-end: the same pair of cross-range-separated
targets resolves into two peaks with a 1m synthetic aperture and collapses
into one with a 5cm aperture, no other change.

## 7. The Antenna-Pose Coherence Budget

Backprojection assumes you know exactly where the antenna was at each
measurement. If your position tracking (in a real device: visual-inertial
odometry, encoders, whatever) is off by `Δp`, the phase correction applied
during reconstruction is wrong by an amount that grows with `Δp` and with
frequency. The classical rule of thumb for "still well focused": keep the
round-trip path error under a quarter wavelength, which works out to an
antenna-position tolerance of `λ/8`:

```rust
use wifi_densepose_sar::resolution::max_coherent_pose_error_m;

let budget = max_coherent_pose_error_m(8.0e9); // 8 GHz carrier
println!("position tolerance at 8 GHz: {:.1} mm", budget * 1000.0);
// -> position tolerance at 8 GHz: 4.7 mm
```

`tests/physics_validation.rs::phase_error_from_pose_jitter_degrades_focus_beyond_pose_budget`
verifies this isn't just asserted: it perturbs the *true* antenna positions
away from the *assumed* ones used in reconstruction, and shows focus at the
true target location degrades as that perturbation grows — the concrete
mechanism behind why real SAR/GPR imaging systems need accurate pose
tracking, not just a good radio.

## 8. Extracting a Point Cloud

A dense voxel grid isn't a useful end product — you want a short list of
detected points:

Continuing the program from §4 (which already has `image` in scope):

```rust
use wifi_densepose_sar::extract_point_cloud;

let points = extract_point_cloud(&image, 0.5); // 50%-of-peak threshold
for p in &points {
    println!("{:?} magnitude={:.3}", p.position, p.magnitude);
}
```

`extract_point_cloud` does threshold + 6-connected local-maximum
extraction — a real blob will still yield one point, not one per voxel
inside it. There is deliberately no clustering, material classification,
or confidence calibration here (ADR-287 §5): that needs real data to
calibrate against, which this crate does not have.

## 9. Benchmarking Your Own Scenario

```bash
cargo bench -p wifi-densepose-sar
```

The shipped benchmark (`benches/backprojection_bench.rs`) sweeps 512 /
4,096 / 32,768-voxel grids with 21 poses x 32 frequencies. Reconstruction
is embarrassingly parallel over voxels (each voxel's cost is independent),
so it's rayon-parallelized already — see the crate README for the last
recorded MEASURED numbers on the reference machine.

## 10. Where This Could Go Next

This crate deliberately stops short of several things (ADR-287 §5):

- It's monostatic (one antenna, both TX and RX) — real handheld SAR/MIMO
  devices often use multiple simultaneous antenna elements.
- The forward model is free-space only — no multipath, no per-material
  attenuation (contrast `ruview-unified`'s narrowband Fresnel material
  model, which isn't yet extended to wideband).
- It isn't wired into `ruview-unified`'s `FmcwRadarCube` adapter or
  `GaussianMap` — ADR-278 names that as the eventual integration point,
  once (and if) a reconstruction system is ready for it.

If you're picking this up to extend it, start with ADR-287's "Follow-up"
section rather than guessing at scope.

## 11. Troubleshooting

**"My reconstructed peak isn't near my target."** Check your voxel grid
actually covers the target's true location — `backproject` happily
reconstructs whatever region you ask for; if the target is outside the
grid, you'll get whatever's brightest inside it instead (usually noise).

**"Two targets I expected to resolve didn't."** Compute
`range_resolution_m`/`cross_range_resolution_m` for your actual bandwidth
and aperture length and check your separation against them — resolution
is a hard physical limit here, not a tuning parameter.

**"Backprojection is slow for my grid size."** Cost is
`O(voxels x poses x freqs)` and already parallelized over voxels via
rayon; the only way to go faster is fewer voxels, fewer poses, or fewer
frequency steps (each is a hard tradeoff against resolution or aperture
coverage — see §5/§6).
