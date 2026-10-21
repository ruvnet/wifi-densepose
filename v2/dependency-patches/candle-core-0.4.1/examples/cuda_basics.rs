#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Module, Tensor};

use candle_core::quantized::{QMatMul, QTensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::randn(0f32, 1.0, (72, 256), &device)?;
    let q_cpu = q.to_device(&Device::Cpu)?;
    let q = QTensor::quantize(&q, candle_core::quantized::GgmlDType::Q8K)?;
    let q = QMatMul::from_qtensor(q)?;
    let x = Tensor::randn(0f32, 1.0, (5, 256), &device)?;
    let res_q_cuda = q.forward(&x)?;
    println!("{res_q_cuda}");

    let q_cpu = QTensor::quantize(&q_cpu, candle_core::quantized::GgmlDType::Q8K)?;
    let q_cpu_tensor = q_cpu.dequantize(&Device::Cpu)?;
    let q_cpu = QMatMul::from_qtensor(q_cpu)?;
    let x_cpu = x.to_device(&Device::Cpu)?;
    let res_q_cpu = q_cpu.forward(&x_cpu)?;
    println!("{res_q_cpu}");

    let res_mm = x_cpu.matmul(&q_cpu_tensor.t()?)?;
    let diff = (res_mm - res_q_cuda.to_device(&Device::Cpu))?
        .abs()?
        .flatten_all()?
        .max(0)?;
    println!("{diff}");
    Ok(())
}
