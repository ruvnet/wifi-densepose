//! Deterministic CPU inference benchmarks over generated tensor fixtures.

#![allow(missing_docs)]

use burn_core::tensor::{backend::Backend, Tensor};
use burn_ndarray::{NdArray, NdArrayDevice};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruview_forecast_model::{ForecastModelConfig, ModelInput, RuForecastMixer};

type Cpu = NdArray<f32>;

fn generated_input(
    config: &ForecastModelConfig,
    variates: usize,
    device: NdArrayDevice,
) -> ModelInput<Cpu> {
    ModelInput::new(
        config,
        Tensor::zeros([1, config.context_len, variates], &device),
        Tensor::ones([1, config.context_len, variates], &device),
        Tensor::zeros([1, config.context_len, variates], &device),
        Tensor::zeros([1, config.context_len, config.time_width], &device),
        Tensor::zeros([1, config.horizon, config.time_width], &device),
        Tensor::zeros([1, variates, config.descriptor_width], &device),
        Tensor::ones([1, variates], &device),
    )
    .expect("valid generated benchmark input")
}

fn forecast_inference(c: &mut Criterion) {
    let device = NdArrayDevice::default();
    Cpu::seed(&device, 0x5255_5646);
    let mut group = c.benchmark_group("forecast_inference");

    let tiny = ForecastModelConfig::tiny_ci();
    let tiny_model = RuForecastMixer::<Cpu>::init(&tiny, &device).expect("valid tiny config");
    let tiny_input = generated_input(&tiny, 4, device);
    group.bench_with_input(
        BenchmarkId::new("tiny_ci", "B1_T64_V4_H12_Q7"),
        &tiny_input,
        |bencher, input| {
            bencher.iter(|| {
                let output = tiny_model
                    .forward(black_box(input.clone()))
                    .expect("valid generated input");
                black_box(output.normalized_quantiles.into_data())
            });
        },
    );

    if std::env::var("RUFORECAST_BENCH_LARGE").as_deref() == Ok("1") {
        let deployment = ForecastModelConfig::large_linux();
        let deployment_model =
            RuForecastMixer::<Cpu>::init(&deployment, &device).expect("valid deployment config");
        let deployment_input = generated_input(&deployment, 32, device);
        group.bench_with_input(
            BenchmarkId::new("deployment", "B1_T1024_V32_H300_Q7"),
            &deployment_input,
            |bencher, input| {
                bencher.iter(|| {
                    let output = deployment_model
                        .forward(black_box(input.clone()))
                        .expect("valid generated input");
                    black_box(output.normalized_quantiles.into_data())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, forecast_inference);
criterion_main!(benches);
