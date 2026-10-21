//! Recurrent Neural Networks
use crate::{Device, Tensor};
use std::borrow::Borrow;

/// Trait for Recurrent Neural Networks.
#[allow(clippy::upper_case_acronyms)]
pub trait RNN {
    type State;

    /// A zero state from which the recurrent network is usually initialized.
    fn zero_state(&self, batch_dim: i64) -> Self::State;

    /// Applies a single step of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, features].
    fn step(&self, input: &Tensor, state: &Self::State) -> Self::State;

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    /// The initial state is the result of applying zero_state.
    fn seq(&self, input: &Tensor) -> (Tensor, Self::State) {
        let batch_dim = input.size()[0];
        let state = self.zero_state(batch_dim);
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, state: &Self::State) -> (Tensor, Self::State);
}

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct LSTMState(pub (Tensor, Tensor));

impl LSTMState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> Tensor {
        (self.0).0.shallow_clone()
    }

    /// The cell state vector.
    pub fn c(&self) -> Tensor {
        (self.0).1.shallow_clone()
    }
}

// The GRU and LSTM layers share the same config.
/// Configuration for the GRU and LSTM layers.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct RNNConfig {
    pub has_biases: bool,
    pub num_layers: i64,
    pub dropout: f64,
    pub train: bool,
    pub bidirectional: bool,
    pub batch_first: bool,
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
}

impl Default for RNNConfig {
    fn default() -> Self {
        RNNConfig {
            has_biases: true,
            num_layers: 1,
            dropout: 0.,
            train: true,
            bidirectional: false,
            batch_first: true,
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
        }
    }
}

fn rnn_weights<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    hidden_dim: i64,
    gate_dim: i64,
    num_directions: i64,
    c: RNNConfig,
) -> Vec<Tensor> {
    let vs = vs.borrow();
    let mut flat_weights = vec![];
    for layer_idx in 0..c.num_layers {
        for direction_idx in 0..num_directions {
            let in_dim = if layer_idx == 0 { in_dim } else { hidden_dim * num_directions };
            let suffix = if direction_idx == 1 { "_reverse" } else { "" };
            let w_ih = vs.var(
                &format!("weight_ih_l{layer_idx}{suffix}"),
                &[gate_dim, in_dim],
                c.w_ih_init,
            );
            let w_hh = vs.var(
                &format!("weight_hh_l{layer_idx}{suffix}"),
                &[gate_dim, hidden_dim],
                c.w_hh_init,
            );
            flat_weights.push(w_ih);
            flat_weights.push(w_hh);
            if c.has_biases {
                let b_ih = vs.var(
                    &format!("bias_ih_l{layer_idx}{suffix}"),
                    &[gate_dim],
                    c.b_ih_init.unwrap(),
                );
                let b_hh = vs.var(
                    &format!("bias_hh_l{layer_idx}{suffix}"),
                    &[gate_dim],
                    c.b_hh_init.unwrap(),
                );
                flat_weights.push(b_ih);
                flat_weights.push(b_hh);
            }
        }
    }
    flat_weights
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct LSTM {
    flat_weights: Vec<Tensor>,
    hidden_dim: i64,
    config: RNNConfig,
    device: Device,
}

/// Creates a LSTM layer.
pub fn lstm<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    hidden_dim: i64,
    c: RNNConfig,
) -> LSTM {
    let vs = vs.borrow();
    let num_directions = if c.bidirectional { 2 } else { 1 };
    let gate_dim = 4 * hidden_dim;
    let flat_weights = rnn_weights(vs, in_dim, hidden_dim, gate_dim, num_directions, c);

    if vs.device().is_cuda() && crate::Cuda::cudnn_is_available() {
        let _ = Tensor::internal_cudnn_rnn_flatten_weight(
            &flat_weights,
            4,
            in_dim,
            2, /* 2 for LSTM see rnn.cpp in pytorch */
            hidden_dim,
            0, /* disables projections */
            c.num_layers,
            c.batch_first,
            c.bidirectional,
        );
    }
    LSTM { flat_weights, hidden_dim, config: c, device: vs.device() }
}

impl RNN for LSTM {
    type State = LSTMState;

    fn zero_state(&self, batch_dim: i64) -> LSTMState {
        let num_directions = if self.config.bidirectional { 2 } else { 1 };
        let layer_dim = self.config.num_layers * num_directions;
        let shape = [layer_dim, batch_dim, self.hidden_dim];
        let zeros = Tensor::zeros(shape, (self.flat_weights[0].kind(), self.device));
        LSTMState((zeros.shallow_clone(), zeros.shallow_clone()))
    }

    fn step(&self, input: &Tensor, in_state: &LSTMState) -> LSTMState {
        let input = input.unsqueeze(1);
        let (_output, state) = self.seq_init(&input, in_state);
        state
    }

    fn seq_init(&self, input: &Tensor, in_state: &LSTMState) -> (Tensor, LSTMState) {
        let LSTMState((h, c)) = in_state;
        let flat_weights = self.flat_weights.iter().collect::<Vec<_>>();
        let (output, h, c) = input.lstm(
            &[h, c],
            &flat_weights,
            self.config.has_biases,
            self.config.num_layers,
            self.config.dropout,
            self.config.train,
            self.config.bidirectional,
            self.config.batch_first,
        );
        (output, LSTMState((h, c)))
    }
}

/// A GRU state, this contains a single tensor.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GRUState(pub Tensor);

impl GRUState {
    pub fn value(&self) -> Tensor {
        self.0.shallow_clone()
    }
}

/// A Gated Recurrent Unit (GRU) layer.
///
/// <https://en.wikipedia.org/wiki/Gated_recurrent_unit>
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GRU {
    flat_weights: Vec<Tensor>,
    hidden_dim: i64,
    config: RNNConfig,
    device: Device,
}

/// Creates a new GRU layer.
pub fn gru<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    hidden_dim: i64,
    c: RNNConfig,
) -> GRU {
    let vs = vs.borrow();
    let num_directions = if c.bidirectional { 2 } else { 1 };
    let gate_dim = 3 * hidden_dim;
    let flat_weights = rnn_weights(vs, in_dim, hidden_dim, gate_dim, num_directions, c);

    if vs.device().is_cuda() && crate::Cuda::cudnn_is_available() {
        let _ = Tensor::internal_cudnn_rnn_flatten_weight(
            &flat_weights,
            4,
            in_dim,
            3, /* 3 for GRU see rnn.cpp in pytorch */
            hidden_dim,
            0, /* disables projections */
            c.num_layers,
            c.batch_first,
            c.bidirectional,
        );
    }
    GRU { flat_weights, hidden_dim, config: c, device: vs.device() }
}

impl RNN for GRU {
    type State = GRUState;

    fn zero_state(&self, batch_dim: i64) -> GRUState {
        let num_directions = if self.config.bidirectional { 2 } else { 1 };
        let layer_dim = self.config.num_layers * num_directions;
        let shape = [layer_dim, batch_dim, self.hidden_dim];
        GRUState(Tensor::zeros(shape, (self.flat_weights[0].kind(), self.device)))
    }

    fn step(&self, input: &Tensor, in_state: &GRUState) -> GRUState {
        let input = input.unsqueeze(1);
        let (_output, state) = self.seq_init(&input, in_state);
        state
    }

    fn seq_init(&self, input: &Tensor, in_state: &GRUState) -> (Tensor, GRUState) {
        let GRUState(h) = in_state;
        let (output, h) = input.gru(
            h,
            &self.flat_weights,
            self.config.has_biases,
            self.config.num_layers,
            self.config.dropout,
            self.config.train,
            self.config.bidirectional,
            self.config.batch_first,
        );
        (output, GRUState(h))
    }
}
