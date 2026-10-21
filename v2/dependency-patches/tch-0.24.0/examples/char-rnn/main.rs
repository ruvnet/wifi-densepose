/* This example uses the tinyshakespeare dataset which can be downloaded at:
   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

   It has been heavily inspired by https://github.com/karpathy/char-rnn
*/

use anyhow::Result;
use tch::data::TextData;
use tch::nn::{Linear, Module, OptimizerConfig, LSTM, RNN};
use tch::{nn, Device, Kind, Tensor};

const LEARNING_RATE: f64 = 0.01;
const HIDDEN_SIZE: i64 = 256;
const SEQ_LEN: i64 = 180;
const BATCH_SIZE: i64 = 256;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 1024;

/// Generates some sample string using the LSTM + Linear model.
fn sample(data: &TextData, lstm: &LSTM, linear: &Linear, device: Device) -> String {
    let labels = data.labels();
    let mut state = lstm.zero_state(1);
    let mut last_label = 0i64;
    let mut result = String::new();
    for _index in 0..SAMPLING_LEN {
        let input = Tensor::zeros([1, labels], (Kind::Float, device));
        let _ = input.narrow(1, last_label, 1).fill_(1.0);
        state = lstm.step(&input, &state);
        let sampled_y = linear
            .forward(&state.h())
            .squeeze_dim(0)
            .softmax(-1, Kind::Float)
            .multinomial(1, false);
        last_label = i64::try_from(sampled_y).unwrap();
        result.push(data.label_to_char(last_label))
    }
    result
}

pub fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let data = TextData::new("data/input.txt")?;
    let labels = data.labels();
    println!("Dataset loaded, {labels} labels.");
    let lstm = nn::lstm(vs.root(), labels, HIDDEN_SIZE, Default::default());
    let linear = nn::linear(vs.root(), HIDDEN_SIZE, labels, Default::default());
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    for epoch in 1..(1 + EPOCHS) {
        let mut sum_loss = 0.;
        let mut cnt_loss = 0.;
        for batch in data.iter_shuffle(SEQ_LEN + 1, BATCH_SIZE) {
            let xs_onehot = batch.narrow(1, 0, SEQ_LEN).onehot(labels);
            let ys = batch.narrow(1, 1, SEQ_LEN).to_kind(Kind::Int64);
            let (lstm_out, _) = lstm.seq(&xs_onehot.to_device(device));
            let logits = linear.forward(&lstm_out);
            let loss = logits
                .view([BATCH_SIZE * SEQ_LEN, labels])
                .cross_entropy_for_logits(&ys.to_device(device).view([BATCH_SIZE * SEQ_LEN]));
            opt.backward_step_clip(&loss, 0.5);
            sum_loss += f64::try_from(loss)?;
            cnt_loss += 1.0;
        }
        println!("Epoch: {}   loss: {:5.3}", epoch, sum_loss / cnt_loss);
        println!("Sample: {}", sample(&data, &lstm, &linear, device));
    }
    Ok(())
}
