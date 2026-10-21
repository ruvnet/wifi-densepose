use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(a: &Tensor, b: &Tensor) {
    a.matmul(&b.t().unwrap()).unwrap();
}

fn run_bench(c: &mut Criterion, device: &Device) {
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let dtype = DType::F32;
    let lhs = Tensor::zeros((b, m, k), dtype, device).unwrap();
    let rhs = Tensor::zeros((b, n, k), dtype, device).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(device.bench_name("matmul"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&lhs), black_box(&rhs));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_bench(c, &device);
    }
}

criterion_group!(benches, criterion_benchmark);
