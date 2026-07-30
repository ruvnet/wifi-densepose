//! Criterion benchmark for the backprojection reconstruction kernel.
//! `cargo bench -p wifi-densepose-sar` reports MEASURED throughput --
//! see the crate README for the last recorded numbers.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use wifi_densepose_sar::geometry::linear_aperture;
use wifi_densepose_sar::measurement::{simulate_measurement, FrequencySweep, ScatteringTarget};
use wifi_densepose_sar::reconstruct::{backproject, VoxelGrid};
use wifi_densepose_sar::Point3;

fn backprojection_benchmark(c: &mut Criterion) {
    let poses = linear_aperture(Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0), 21);
    let sweep = FrequencySweep::new(2.0e9, 6.0e9, 32);
    let targets = vec![
        ScatteringTarget::new(Point3::new(0.0, 2.0, 0.0), 1.0),
        ScatteringTarget::new(Point3::new(0.3, 1.8, 0.1), 0.6),
    ];
    let measurement = simulate_measurement(&poses, &sweep, &targets, 0.01, 7);

    let mut group = c.benchmark_group("backproject");
    for &voxels_per_axis in &[8usize, 16, 32] {
        let grid = VoxelGrid::new(
            Point3::new(-0.5, 1.5, -0.5),
            1.0 / voxels_per_axis as f64,
            voxels_per_axis,
            voxels_per_axis,
            voxels_per_axis,
        );
        group.bench_with_input(BenchmarkId::from_parameter(grid.len()), &grid, |b, grid| {
            b.iter(|| backproject(&measurement, &poses, &sweep, grid));
        });
    }
    group.finish();
}

criterion_group!(benches, backprojection_benchmark);
criterion_main!(benches);
