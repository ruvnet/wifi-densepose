# wifi-densepose-sar

Coherent wideband RF tomography research crate (ADR-287): synthetic
stepped-frequency multi-position measurement simulation + delay-and-sum
backprojection reconstruction of a 3D reflectivity field.

**This is not a hardware capability.** It is the reconstruction primitive a
handheld through-wall RF imaging device would need, validated against its
own synthetic ground truth. Every number this crate produces is
SYNTHETIC / evidence level L0 (ADR-282) until real wideband RF hardware
(a VNA, SDR, or purpose-built radar front end) exists to feed it real
measurements. See the crate-level doc comment in `src/lib.rs` for the full
honesty boundary, and the tutorial at
`docs/tutorials/coherent-rf-tomography-backprojection.md` for a walkthrough.

## Quick example

```rust
use wifi_densepose_sar::{
    backproject, linear_aperture, simulate_measurement, FrequencySweep,
    Point3, ScatteringTarget, VoxelGrid,
};

let poses = linear_aperture(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0), 21);
let sweep = FrequencySweep::new(2.0e9, 6.0e9, 32);
let target = ScatteringTarget::new(Point3::new(0.0, 2.0, 0.0), 1.0);
let measurement = simulate_measurement(&poses, &sweep, &[target], 0.01, 42);

let grid = VoxelGrid::new(Point3::new(-0.3, 1.7, -0.3), 0.03, 21, 21, 21);
let image = backproject(&measurement, &poses, &sweep, &grid);
let (peak_location, peak_magnitude) = image.peak();
println!("reconstructed target near {peak_location:?}, magnitude {peak_magnitude:.4}");
```

## Testing

```bash
cargo test -p wifi-densepose-sar --no-default-features
cargo bench -p wifi-densepose-sar
```

`tests/physics_validation.rs` checks the reconstruction's actual behavior
against the closed-form formulas in `resolution.rs` (range resolution,
cross-range/synthetic-aperture resolution, and the antenna-pose coherence
budget) rather than merely asserting them: 25 tests (22 unit + 3
integration), 0 failed, clippy-clean.

## Performance (MEASURED)

`cargo bench -p wifi-densepose-sar`, 21 antenna poses × 32 frequency steps
(672 measurement terms/voxel), rayon-parallelized over voxels, this
machine, release profile:

| Voxels | Median time | Throughput |
|-------:|------------:|-----------:|
| 512    | 300 µs      | ~1.71M voxels/s |
| 4,096  | 1.97 ms     | ~2.08M voxels/s |
| 32,768 | 14.5 ms     | ~2.26M voxels/s |

Scales as expected: each voxel's cost is independent (`O(poses × freqs)`
per voxel, embarrassingly parallel), so throughput is roughly constant
across grid sizes and total time scales linearly with voxel count.

**Optimization (MEASURED, criterion regression detection, p < 0.001): ~4.4-4.5x
faster** than the first-shipped implementation, across all three grid
sizes. Frequencies in a [`FrequencySweep`](src/measurement.rs) are evenly
spaced by construction, so the per-(pose, frequency) phase term is an
arithmetic progression; `focus_at_point` now evaluates the phasor once per
pose and advances it by a fixed complex-multiply step per frequency,
instead of one `sin`/`cos` pair (`Complex64::from_polar`) per frequency —
K trig evaluations become 2. Proven equivalent (not just faster) to an
independently-reimplemented direct per-frequency reference in
`reconstruct::tests::backprojection_incremental_rotation_matches_direct_per_frequency_computation`,
across several sweep sizes and both on-target and off-target points.
