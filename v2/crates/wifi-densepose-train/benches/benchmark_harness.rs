//! ADR-288 benchmarks: bfee parser throughput and split assignment over
//! synthetic, code-generated corpora (no dataset files are read or written).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use wifi_densepose_train::dataset::widar::{encode_bfee_frame, parse_bfee_bytes, WIDAR_SUBCARRIERS};
use wifi_densepose_train::protocols::leakage::LeakageAudit;
use wifi_densepose_train::protocols::{SampleMeta, SplitPlan, SplitProtocol, SplitSide};

/// Deterministic synthetic bfee log: `num_records` framed 3×3 records.
fn synthetic_log(num_records: usize) -> Vec<u8> {
    let (n_rx, n_tx) = (3u8, 3u8);
    let pairs = WIDAR_SUBCARRIERS * n_rx as usize * n_tx as usize;
    let mut bytes = Vec::new();
    for t in 0..num_records {
        let csi: Vec<(i16, i16)> = (0..pairs)
            .map(|i| {
                let re = ((t * 37 + i * 13) % 1024) as i16 - 512;
                let im = ((t * 17 + i * 7) % 1024) as i16 - 512;
                (re, im)
            })
            .collect();
        bytes.extend_from_slice(&encode_bfee_frame(t as u32, t as u16, n_rx, n_tx, &csi));
    }
    bytes
}

/// Deterministic synthetic metadata corpus.
fn synthetic_metas(n: usize) -> Vec<SampleMeta> {
    (0..n)
        .map(|i| SampleMeta {
            subject_id: 1 + (i % 17) as u32,
            environment_id: 1 + (i % 3) as u32,
            orientation_id: 1 + (i % 5) as u32,
            gesture_id: 1 + (i % 6) as u32,
            recording_id: (i / 50) as u64,
            window_index: (i % 50) as u64,
        })
        .collect()
}

/// Deterministic synthetic corpus whose domain attributes are constant per
/// recording (as real datasets are), so protocol splits keep recordings whole
/// and the leakage audit exercises its full passing path.
fn synthetic_recording_metas(n: usize, windows_per_recording: usize) -> Vec<SampleMeta> {
    (0..n)
        .map(|i| {
            let recording = (i / windows_per_recording) as u64;
            SampleMeta {
                subject_id: 1 + (recording % 17) as u32,
                environment_id: 1 + (recording % 3) as u32,
                orientation_id: 1 + (recording % 5) as u32,
                gesture_id: 1 + (recording % 6) as u32,
                recording_id: recording,
                window_index: (i % windows_per_recording) as u64,
            }
        })
        .collect()
}

fn bench_bfee_parser(c: &mut Criterion) {
    let bytes = synthetic_log(500);
    let mut group = c.benchmark_group("widar_bfee_parse");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("500_records_3x3", |b| {
        b.iter(|| {
            let parse = parse_bfee_bytes(black_box(&bytes));
            assert_eq!(parse.records.len(), 500);
            parse
        })
    });
    group.finish();
}

fn bench_split_assignment(c: &mut Criterion) {
    let metas = synthetic_metas(10_000);
    let mut group = c.benchmark_group("split_assignment");
    group.throughput(Throughput::Elements(metas.len() as u64));
    for protocol in [
        SplitProtocol::CrossSubject,
        SplitProtocol::CrossEnvironment,
        SplitProtocol::CrossOrientation,
        SplitProtocol::RandomBaseline,
    ] {
        let plan = SplitPlan::new(protocol, 42, 0.3).expect("valid fraction");
        group.bench_function(protocol.tag(), |b| {
            b.iter(|| plan.partition(black_box(&metas)))
        });
    }
    group.finish();
}

fn bench_leakage_audit(c: &mut Criterion) {
    // ~10k windows in 200 recordings; a clean cross-subject split so the
    // audit runs every check (recording crossing + claimed disjointness) to
    // completion instead of failing fast.
    let metas = synthetic_recording_metas(10_000, 50);
    let plan = SplitPlan::new(SplitProtocol::CrossSubject, 42, 0.3).expect("valid fraction");
    let mut train = Vec::new();
    let mut test = Vec::new();
    for meta in &metas {
        match plan.assign(meta) {
            SplitSide::Train => train.push(*meta),
            SplitSide::Test => test.push(*meta),
        }
    }
    assert!(!train.is_empty() && !test.is_empty(), "degenerate corpus");

    let audit = LeakageAudit::for_protocol(SplitProtocol::CrossSubject);
    let mut group = c.benchmark_group("leakage_audit");
    group.throughput(Throughput::Elements(metas.len() as u64));
    group.bench_function("cross_subject_10k_windows", |b| {
        b.iter(|| {
            audit
                .audit(black_box(&train), black_box(&test))
                .expect("clean split must pass")
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_bfee_parser,
    bench_split_assignment,
    bench_leakage_audit
);
criterion_main!(benches);
