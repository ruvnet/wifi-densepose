//! Criterion benchmarks for the per-frame hot path (ADR-324).
//!
//! Reproducer: `cd v2 && cargo bench -p ruview-offaxis`
//! Any numbers quoted from this bench are MEASURED on the machine that ran
//! that command; re-run locally before relying on them.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruview_offaxis::{
    field_peak, off_axis, CoarseParallax, CoarseParallaxConfig, OneEuro3, OneEuroConfig, Screen,
    Vec3,
};

fn bench_projection(c: &mut Criterion) {
    let screen = Screen::centered(0.6, 0.34).unwrap();
    c.bench_function("off_axis_projection", |b| {
        let mut t = 0.0_f64;
        b.iter(|| {
            t += 0.016;
            let eye = Vec3::new(0.1 * t.sin(), 0.05 * t.cos(), 0.65);
            black_box(off_axis(black_box(&screen), black_box(eye), 0.05, 100.0).unwrap())
        })
    });

    c.bench_function("off_axis_view_projection_combined", |b| {
        let eye = Vec3::new(0.1, -0.03, 0.65);
        let oa = off_axis(&screen, eye, 0.05, 100.0).unwrap();
        b.iter(|| black_box(black_box(&oa).view_projection()))
    });
}

fn bench_filter(c: &mut Criterion) {
    c.bench_function("one_euro3_step", |b| {
        let mut f = OneEuro3::new(OneEuroConfig::default());
        let mut t = 0.0_f64;
        b.iter(|| {
            t += 0.016;
            black_box(f.filter(Vec3::new(t.sin(), t.cos(), 0.65), t))
        })
    });
}

fn make_grid(nx: usize, nz: usize) -> Vec<f32> {
    // Deterministic pseudo-field with one hot cell.
    let mut v: Vec<f32> = (0..nx * nz).map(|i| 0.05 + (i % 7) as f32 * 0.01).collect();
    v[(nz / 3) * nx + nx / 4] = 0.9;
    v
}

fn bench_field_peak(c: &mut Criterion) {
    let g20 = make_grid(20, 20);
    c.bench_function("field_peak_20x20", |b| {
        b.iter(|| black_box(field_peak(black_box(&g20), 20, 20)))
    });

    let g100 = make_grid(100, 100);
    c.bench_function("field_peak_100x100", |b| {
        b.iter(|| black_box(field_peak(black_box(&g100), 100, 100)))
    });
}

fn bench_full_tier_b_frame(c: &mut Criterion) {
    // The whole Tier B per-frame path: grid scan → parallax → projection.
    let screen = Screen::centered(0.6, 0.34).unwrap();
    let g20 = make_grid(20, 20);
    c.bench_function("tier_b_full_frame_20x20", |b| {
        let mut cp = CoarseParallax::new(CoarseParallaxConfig::default());
        let mut t = 0.0_f64;
        b.iter(|| {
            t += 0.016;
            let peak = field_peak(black_box(&g20), 20, 20);
            let eye = cp.update(peak, t);
            black_box(off_axis(&screen, eye, 0.05, 100.0).unwrap())
        })
    });
}

criterion_group!(
    benches,
    bench_projection,
    bench_filter,
    bench_field_peak,
    bench_full_tier_b_frame
);
criterion_main!(benches);
