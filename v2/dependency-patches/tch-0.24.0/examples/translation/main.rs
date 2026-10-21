/* Translation with a Sequence to Sequence Model and Attention.

   This follows the line of the PyTorch tutorial:
   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
   And trains a Sequence to Sequence (seq2seq) model using attention to
   perform translation between French and English.

   The dataset can be downloaded from the following link:
   https://download.pytorch.org/tutorial/data.zip
   The eng-fra.txt file should be moved in the data directory.
*/
use anyhow::Result;
use rand::prelude::*;
use tch::nn::{GRUState, Module, OptimizerConfig, RNN};
use tch::{nn, Device, Kind, Tensor};

mod dataset;
use dataset::Dataset;
mod lang;
use lang::Lang;

const MAX_LENGTH: usize = 10;
const LEARNING_RATE: f64 = 0.001;
const SAMPLES: usize = 100_000;
const HIDDEN_SIZE: usize = 256;

struct Encoder {
    embedding: nn::Embedding,
    gru: nn::GRU,
}

impl Encoder {
    fn new(vs: nn::Path, in_dim: usize, hidden_dim: usize) -> Self {
        let in_dim = in_dim as i64;
        let hidden_dim = hidden_dim as i64;
        let gru = nn::gru(&vs, hidden_dim, hidden_dim, Default::default());
        let embedding = nn::embedding(&vs, in_dim, hidden_dim, Default::default());
        Encoder { embedding, gru }
    }

    fn forward(&self, xs: &Tensor, state: &GRUState) -> (Tensor, GRUState) {
        let xs = self.embedding.forward(xs).view([1, -1]);
        let state = self.gru.step(&xs, state);
        (state.value().squeeze_dim(1), state)
    }
}

struct Decoder {
    device: Device,
    embedding: nn::Embedding,
    gru: nn::GRU,
    attn: nn::Linear,
    attn_combine: nn::Linear,
    linear: nn::Linear,
}

impl Decoder {
    fn new(vs: nn::Path, hidden_dim: usize, out_dim: usize) -> Self {
        let hidden_dim = hidden_dim as i64;
        let out_dim = out_dim as i64;
        Decoder {
            device: vs.device(),
            embedding: nn::embedding(&vs, out_dim, hidden_dim, Default::default()),
            gru: nn::gru(&vs, hidden_dim, hidden_dim, Default::default()),
            attn: nn::linear(&vs, hidden_dim * 2, MAX_LENGTH as i64, Default::default()),
            attn_combine: nn::linear(&vs, hidden_dim * 2, hidden_dim, Default::default()),
            linear: nn::linear(&vs, hidden_dim, out_dim, Default::default()),
        }
    }

    fn forward(
        &self,
        xs: &Tensor,
        state: &GRUState,
        enc_outputs: &Tensor,
        is_training: bool,
    ) -> (Tensor, GRUState) {
        let xs = self.embedding.forward(xs).dropout(0.1, is_training).view([1, -1]);
        let attn_weights =
            Tensor::cat(&[&xs, &state.value().squeeze_dim(1)], 1).apply(&self.attn).unsqueeze(0);
        let (sz1, sz2, sz3) = enc_outputs.size3().unwrap();
        let enc_outputs = if sz2 == MAX_LENGTH as i64 {
            enc_outputs.shallow_clone()
        } else {
            let shape = [sz1, MAX_LENGTH as i64 - sz2, sz3];
            let zeros = Tensor::zeros(shape, (Kind::Float, self.device));
            Tensor::cat(&[enc_outputs, &zeros], 1)
        };
        let attn_applied = attn_weights.bmm(&enc_outputs).squeeze_dim(1);
        let xs = Tensor::cat(&[&xs, &attn_applied], 1).apply(&self.attn_combine).relu();
        let state = self.gru.step(&xs, state);
        (self.linear.forward(&state.value()).log_softmax(-1, Kind::Float).squeeze_dim(1), state)
    }
}

struct Model {
    encoder: Encoder,
    decoder: Decoder,
    decoder_start: Tensor,
    decoder_eos: usize,
    device: Device,
}

impl Model {
    fn new(vs: nn::Path, ilang: &Lang, olang: &Lang, hidden_dim: usize) -> Self {
        Model {
            encoder: Encoder::new(&vs / "enc", ilang.len(), hidden_dim),
            decoder: Decoder::new(&vs / "dec", hidden_dim, olang.len()),
            decoder_start: Tensor::from_slice(&[olang.sos_token() as i64]).to_device(vs.device()),
            decoder_eos: olang.eos_token(),
            device: vs.device(),
        }
    }

    #[allow(clippy::assign_op_pattern)]
    fn train_loss(&self, input_: &[usize], target: &[usize], rng: &mut ThreadRng) -> Tensor {
        let mut state = self.encoder.gru.zero_state(1);
        let mut enc_outputs = vec![];
        for &s in input_.iter() {
            let s = Tensor::from_slice(&[s as i64]).to_device(self.device);
            let (out, state_) = self.encoder.forward(&s, &state);
            enc_outputs.push(out);
            state = state_;
        }
        let enc_outputs = Tensor::stack(&enc_outputs, 1);
        let use_teacher_forcing: bool = rng.gen();
        let mut loss = Tensor::from(0f32).to_device(self.device);
        let mut prev = self.decoder_start.shallow_clone();
        for &s in target.iter() {
            let (output, state_) = self.decoder.forward(&prev, &state, &enc_outputs, true);
            state = state_;
            let target_tensor = Tensor::from_slice(&[s as i64]).to_device(self.device);
            loss = loss + output.nll_loss(&target_tensor);
            let (_, output) = output.topk(1, -1, true, true);
            if self.decoder_eos == i64::try_from(&output).unwrap() as usize {
                break;
            }
            prev = if use_teacher_forcing { target_tensor } else { output };
        }
        loss
    }

    fn predict(&self, input_: &[usize]) -> Vec<usize> {
        let mut state = self.encoder.gru.zero_state(1);
        let mut enc_outputs = vec![];
        for &s in input_.iter() {
            let s = Tensor::from_slice(&[s as i64]).to_device(self.device);
            let (out, state_) = self.encoder.forward(&s, &state);
            enc_outputs.push(out);
            state = state_;
        }
        let enc_outputs = Tensor::stack(&enc_outputs, 1);
        let mut prev = self.decoder_start.shallow_clone();
        let mut output_seq: Vec<usize> = vec![];
        for _i in 0..MAX_LENGTH {
            let (output, state_) = self.decoder.forward(&prev, &state, &enc_outputs, true);
            let (_, output) = output.topk(1, -1, true, true);
            let output_ = i64::try_from(&output).unwrap() as usize;
            output_seq.push(output_);
            if self.decoder_eos == output_ {
                break;
            }
            state = state_;
            prev = output;
        }
        output_seq
    }
}

struct LossStats {
    total_loss: f64,
    samples: usize,
}

impl LossStats {
    fn new() -> LossStats {
        LossStats { total_loss: 0., samples: 0 }
    }

    fn update(&mut self, loss: f64) {
        self.total_loss += loss;
        self.samples += 1;
    }

    fn avg_and_reset(&mut self) -> f64 {
        let avg = self.total_loss / self.samples as f64;
        self.total_loss = 0.;
        self.samples = 0;
        avg
    }
}

pub fn main() -> Result<()> {
    let dataset = Dataset::new("eng", "fra", MAX_LENGTH)?.reverse();
    let ilang = dataset.input_lang();
    let olang = dataset.output_lang();
    let pairs = dataset.pairs();
    println!("Input:  {} {} words.", ilang.name(), ilang.len());
    println!("Output: {} {} words.", olang.name(), olang.len());
    println!("Pairs:  {}.", pairs.len());
    let mut rng = thread_rng();
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = Model::new(vs.root(), ilang, olang, HIDDEN_SIZE);
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    let mut loss_stats = LossStats::new();
    for idx in 1..=SAMPLES {
        let (input_, target) = pairs.choose(&mut rng).unwrap();
        let loss = model.train_loss(input_, target, &mut rng);
        opt.backward_step(&loss);
        loss_stats.update(f64::try_from(loss)? / target.len() as f64);
        if idx % 1000 == 0 {
            println!("{} {}", idx, loss_stats.avg_and_reset());
            for _pred_index in 1..5 {
                let (input_, target) = pairs.choose(&mut rng).unwrap();
                let predict = model.predict(input_);
                println!("in:  {}", ilang.seq_to_string(input_));
                println!("tgt: {}", olang.seq_to_string(target));
                println!("out: {}", olang.seq_to_string(&predict));
            }
        }
    }
    Ok(())
}
