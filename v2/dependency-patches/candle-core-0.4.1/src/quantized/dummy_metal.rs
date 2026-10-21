#![allow(unused)]
use super::GgmlDType;
use crate::{Error, MetalDevice, MetalStorage, Result};

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
}

impl QMetalStorage {
    pub fn zeros(_: &MetalDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<MetalStorage> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn quantize(&mut self, _src: &MetalStorage) -> Result<()> {
        Err(Error::NotCompiledWithMetalSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &MetalStorage,
        _layout: &crate::Layout,
    ) -> Result<(MetalStorage, crate::Shape)> {
        Err(Error::NotCompiledWithMetalSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &MetalDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithMetalSupport)
}
