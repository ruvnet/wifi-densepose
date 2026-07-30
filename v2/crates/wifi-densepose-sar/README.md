# wifi-densepose-sar

Coherent wideband RF tomography research crate (ADR-283): synthetic
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
budget) rather than merely asserting them: 24 tests (21 unit + 3
integration), 0 failed, clippy-clean.

## Performance (MEASURED)

`cargo bench -p wifi-densepose-sar`, 21 antenna poses × 32 frequency steps
(672 measurement terms/voxel), rayon-parallelized over voxels, this
machine, release profile:

| Voxels | Median time | Throughput |
|-------:|------------:|-----------:|
| 512    | 1.47 ms     | ~348,000 voxels/s |
| 4,096  | 10.4 ms     | ~394,000 voxels/s |
| 32,768 | 73.5 ms     | ~446,000 voxels/s |

Scales as expected: each voxel's cost is independent (`O(poses × freqs)`
per voxel, embarrassingly parallel), so throughput is roughly constant
across grid sizes and total time scales linearly with voxel count.
