mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::affine::benches,
    benchmarks::matmul::benches,
    benchmarks::random::benches,
    benchmarks::where_cond::benches
);
