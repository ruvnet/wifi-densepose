use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruview_forecast_core::SeriesKey;
use ruview_forecast_train::corpus::JsonlWindow;

fn fixture() -> Vec<u8> {
    let context = 64usize;
    let horizon = 12usize;
    let variates = 4usize;
    let window = JsonlWindow {
        version: 1,
        series_key: SeriesKey::new("synthetic-room", "synthetic-device", "synthetic-session")
            .unwrap(),
        context_start_ms: 1_000,
        variates: variates as u16,
        values: (0..context * variates)
            .map(|index| ((index as f32) * 0.01).sin())
            .collect(),
        observed_mask: vec![1; context * variates],
        targets: (0..horizon * variates)
            .map(|index| ((index as f32) * 0.03).cos())
            .collect(),
        target_mask: vec![1; horizon * variates],
    };
    serde_json::to_vec(&window).unwrap()
}

fn decode_jsonl_window(c: &mut Criterion) {
    let bytes = fixture();
    c.bench_function("decode_bounded_jsonl_window", |b| {
        b.iter(|| {
            let decoded: JsonlWindow = serde_json::from_slice(black_box(&bytes)).unwrap();
            black_box(decoded)
        })
    });
}

criterion_group!(benches, decode_jsonl_window);
criterion_main!(benches);
