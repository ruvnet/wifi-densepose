//! Sparse Layers
use crate::Tensor;
use std::borrow::Borrow;

/// Configuration option for an embedding layer.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingConfig {
    pub sparse: bool,
    pub scale_grad_by_freq: bool,
    pub ws_init: super::Init,
    pub padding_idx: i64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            sparse: false,
            scale_grad_by_freq: false,
            ws_init: super::Init::Randn { mean: 0., stdev: 1. },
            padding_idx: -1,
        }
    }
}

/// An embedding layer.
///
/// An embedding layer acts as a simple lookup table that stores embeddings.
/// This is commonly used to store word embeddings.
#[derive(Debug)]
pub struct Embedding {
    pub ws: Tensor,
    config: EmbeddingConfig,
}

pub fn embedding<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_embeddings: i64,
    embedding_dim: i64,
    config: EmbeddingConfig,
) -> Embedding {
    let vs = vs.borrow();
    Embedding { ws: vs.var("weight", &[num_embeddings, embedding_dim], config.ws_init), config }
}

impl super::module::Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::embedding(
            &self.ws,
            xs,
            self.config.padding_idx,
            self.config.scale_grad_by_freq,
            self.config.sparse,
        )
    }
}
