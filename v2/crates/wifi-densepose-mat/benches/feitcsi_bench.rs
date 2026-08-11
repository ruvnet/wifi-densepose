//! Criterion benchmark for FeitCSI record parse throughput (ADR-289).
//!
//! Measures `parse_record` over synthetic in-code fixtures at the wideband
//! 802.11ax shapes: 20 MHz (242 tones), 80 MHz (996) and the headline
//! 160 MHz / 1992-subcarrier frames an AX210 delivers. Fixtures are
//! deterministic; no wall-clock or randomness feeds the parsed bytes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use wifi_densepose_mat::integration::feitcsi::{parse_record, synth, FeitCsiStreamReader};

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("feitcsi_parse");

    // (label, chan_width_val, HE tone count)
    let shapes = [
        ("he20_242sc", 0u32, 242u32),
        ("he80_996sc", 2u32, 996u32),
        ("he160_1992sc", 3u32, 1992u32),
    ];

    for (label, cw, sc) in shapes {
        // 2x1 MIMO, HE modulation, fixed timestamp: deterministic bytes.
        let bytes = synth::record_bytes(2, 1, sc, cw, 4, 1_000_000);
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                let (record, consumed) =
                    parse_record(black_box(&bytes)).expect("valid synthetic record");
                black_box((record.csi.len(), consumed))
            })
        });
    }

    group.finish();
}

/// Stream-reader throughput over an in-memory multi-record capture at the
/// headline 160 MHz / 1992-subcarrier shape. Exercises the reusable payload
/// scratch buffer in `FeitCsiStreamReader` (one raw-byte allocation per
/// stream, not per record).
fn bench_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("feitcsi_stream");

    const RECORDS: u64 = 16;
    let mut capture = Vec::new();
    for ts in 0..RECORDS {
        // 2x1 MIMO, 160 MHz HE, deterministic device timestamps.
        capture.extend_from_slice(&synth::record_bytes(2, 1, 1992, 3, 4, ts * 1_000));
    }

    group.throughput(Throughput::Bytes(capture.len() as u64));
    group.bench_function("he160_1992sc_x16_records", |b| {
        b.iter(|| {
            let mut reader = FeitCsiStreamReader::new(std::io::Cursor::new(black_box(&capture[..])));
            let mut records = 0u64;
            while let Some(rec) = reader.read_next().expect("valid synthetic capture") {
                black_box(rec.csi.len());
                records += 1;
            }
            assert_eq!(records, RECORDS);
            black_box(records)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_parse, bench_stream);
criterion_main!(benches);
