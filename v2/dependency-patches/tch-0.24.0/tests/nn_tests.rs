use tch::nn::{group_norm, layer_norm};
use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Kind, Reduction, Tensor};

mod test_utils;
use test_utils::*;

#[test]
fn optimizer_test() {
    tch::manual_seed(42);
    // Create some linear data.
    let xs = Tensor::from_slice(&(1..15).collect::<Vec<_>>()).to_kind(Kind::Float).view([-1, 1]);
    let ys = &xs * 0.42 + 1.337;

    // Fit a linear model (with deterministic initialization) on the data.
    let vs = nn::VarStore::new(Device::Cpu);
    let mut opt = nn::Sgd::default().build(&vs, 0.).unwrap();
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
        bias: true,
    };
    let mut linear = nn::linear(vs.root(), 1, 1, cfg);

    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let initial_loss: f64 = from(&loss);
    assert!(initial_loss > 1.0, "{}", "initial loss {initial_loss}");

    opt.set_lr(1e-2);
    // Optimization loop.
    for _idx in 1..50 {
        let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
        opt.backward_step(&loss);
    }
    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let final_loss: f64 = from(&loss);
    assert!(final_loss < 0.25, "{}", "final loss {final_loss}");

    // Reset the weights to their initial values.
    tch::no_grad(|| {
        linear.ws.init(nn::Init::Const(0.));
        linear.bs.as_mut().unwrap().init(nn::Init::Const(0.));
    });
    let initial_loss2: f64 = from(&xs.apply(&linear).mse_loss(&ys, Reduction::Mean));
    assert_eq!(initial_loss, initial_loss2);

    // Set the learning-rate to be very small and check that the loss does not change
    // much.
    opt.set_lr(1e-10);
    for _idx in 1..50 {
        let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
        opt.backward_step(&loss);
    }
    let loss = xs.apply(&linear).mse_loss(&ys, Reduction::Mean);
    let final_loss: f64 = from(&loss);
    assert!((final_loss - initial_loss) < 1e-5, "{}", "final loss {final_loss}")
}

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

#[test]
fn gradient_descent_test() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        let xs = Tensor::zeros([7], kind::FLOAT_CPU);
        let ys = Tensor::zeros([7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow_tensor_scalar(2).sum(Kind::Float);
        opt.backward_step(&loss);
    }
}

#[test]
fn gradient_descent_test_clip_norm() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        let xs = Tensor::zeros([7], kind::FLOAT_CPU);
        let ys = Tensor::ones([7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow_tensor_scalar(2).sum(Kind::Float);
        opt.backward_step_clip_norm(&loss, 0.1);
    }
}

fn round4(t: Tensor) -> Vec<f64> {
    let v = vec_f64_from(&t);
    v.iter().map(|x| (10000. * x).round() / 10000.).collect()
}

#[test]
fn gradient_clip_test() {
    let vs = nn::VarStore::new(Device::Cpu);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    let root = vs.root();
    let var1 = root.ones("v1", &[2]);
    let var2 = root.ones("v2", &[1]);
    let mut var3 = root.ones("v3", &[2]);
    tch::no_grad(|| {
        var3 *= -1;
    });
    let all = Tensor::cat(&[&var1, &(2 * &var2), &(2 * &var3)], 0);
    let loss = all.pow_tensor_scalar(2).sum(Kind::Float);
    opt.zero_grad();
    loss.backward();
    let g1 = var1.grad();
    let g2 = var2.grad();
    let g3 = var3.grad();
    assert_eq!(vec_f64_from(&g1), [2.0, 2.0]);
    assert_eq!(vec_f64_from(&g2), [8.0]);
    assert_eq!(vec_f64_from(&g3), [-8.0, -8.0]);
    // Test clipping the gradient by value.
    let loss = all.pow_tensor_scalar(2).sum(Kind::Float);
    opt.zero_grad();
    loss.backward();
    opt.clip_grad_value(4.0);
    let g1 = var1.grad();
    let g2 = var2.grad();
    let g3 = var3.grad();
    assert_eq!(vec_f64_from(&g1), [2.0, 2.0]);
    assert_eq!(vec_f64_from(&g2), [4.0]);
    assert_eq!(vec_f64_from(&g3), [-4.0, -4.0]);
    // Test clipping the gradient norm.
    let loss = all.pow_tensor_scalar(2).sum(Kind::Float);
    opt.zero_grad();
    loss.backward();
    opt.clip_grad_norm(1.0);
    let g1 = var1.grad();
    let g2 = var2.grad();
    let g3 = var3.grad();
    assert_eq!(round4(g1), [0.1414, 0.1414]);
    assert_eq!(round4(g2), [0.5657]);
    assert_eq!(round4(g3), [-0.5657, -0.5657]);
}

#[test]
fn bn_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let bn = nn::batch_norm1d(vs.root(), 40, Default::default());
    let x = Tensor::randn([10, 40], opts);
    let _y = x.apply_t(&bn, true);
    assert_eq!(vs.len(), 4);
}

#[test]
fn bn_test_no_affine() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let bn_cfg = nn::BatchNormConfig { affine: false, ..Default::default() };
    let bn = nn::batch_norm1d(vs.root(), 40, bn_cfg);
    let x = Tensor::randn([10, 40], opts);
    let _y = x.apply_t(&bn, true);
    assert_eq!(vs.len(), 2);
}

#[test]
fn layer_norm_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let ln = layer_norm(vs.root(), vec![5, 10, 10], Default::default());
    let x = Tensor::randn([20, 5, 10, 10], opts);
    let _y = x.apply(&ln);
}

#[test]
fn group_norm_test() {
    let opts = (tch::Kind::Float, tch::Device::Cpu);
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let ln = group_norm(vs.root(), 2, 10, Default::default());
    let x = Tensor::randn([1, 10, 7], opts);
    let _y = x.apply(&ln);
}

#[test]
fn layer_norm_parameters_test() {
    tch::manual_seed(42);
    // Create some linear data.
    let xs = Tensor::from_slice(&[42.0, 42.0, 42.0, 24.0]).to_kind(Kind::Float).view([-1, 2]);
    let ys = &xs * 0.42 + 1.337;

    // Fit a layer normalization layer (with deterministic initialization) on the data.
    let vs = nn::VarStore::new(Device::Cpu);
    let mut opt = nn::Sgd::default().build(&vs, 1.0).unwrap();
    let mut ln = layer_norm(vs.root(), vec![2], Default::default());

    let loss = xs.apply(&ln).mse_loss(&ys, Reduction::Mean);
    let initial_loss: f64 = from(&loss);
    assert!(initial_loss > 1.0, "{}", "initial loss {initial_loss}");

    // Optimization loop.
    for _idx in 1..50 {
        let loss = xs.apply(&ln).mse_loss(&ys, Reduction::Mean);
        opt.backward_step(&loss);
    }
    let loss = xs.apply(&ln).mse_loss(&ys, Reduction::Mean);
    let final_loss: f64 = from(&loss);
    assert!(final_loss < 0.25, "{}", "final loss {final_loss:?}");

    //     Reset the weights to their initial values.
    tch::no_grad(|| {
        if let Some(ws) = &mut ln.ws {
            ws.init(nn::Init::Const(1.));
        }
        if let Some(bs) = &mut ln.bs {
            bs.init(nn::Init::Const(0.));
        }
    });
    let initial_loss2: f64 = from(&xs.apply(&ln).mse_loss(&ys, Reduction::Mean));
    assert_eq!(initial_loss, initial_loss2)
}

fn gru_test(rnn_config: nn::RNNConfig) {
    use nn::RNN;
    let batch_dim = 5;
    let seq_len = 3;
    let input_dim = 2;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let gru = nn::gru(vs.root(), input_dim, output_dim, rnn_config);

    let num_directions = if rnn_config.bidirectional { 2 } else { 1 };
    let layer_dim = rnn_config.num_layers * num_directions;
    //
    // step test
    let input = Tensor::randn([batch_dim, input_dim], kind::FLOAT_CPU);
    let nn::GRUState(output) = gru.step(&input, &gru.zero_state(batch_dim));
    assert_eq!(output.size(), [layer_dim, batch_dim, output_dim]);

    // seq test
    let input = Tensor::randn([batch_dim, seq_len, input_dim], kind::FLOAT_CPU);
    let (output, _) = gru.seq(&input);
    assert_eq!(output.size(), [batch_dim, seq_len, output_dim * num_directions]);
}

#[test]
fn gru() {
    gru_test(Default::default());
    gru_test(nn::RNNConfig { has_biases: false, ..Default::default() });
    gru_test(nn::RNNConfig { bidirectional: true, ..Default::default() });
    gru_test(nn::RNNConfig { num_layers: 2, ..Default::default() });
    gru_test(nn::RNNConfig { num_layers: 2, bidirectional: true, ..Default::default() });
}

fn lstm_test(rnn_config: nn::RNNConfig) {
    use nn::RNN;
    let batch_dim = 5;
    let seq_len = 3;
    let input_dim = 2;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let lstm = nn::lstm(vs.root(), input_dim, output_dim, rnn_config);

    let num_directions = if rnn_config.bidirectional { 2 } else { 1 };
    let layer_dim = rnn_config.num_layers * num_directions;
    //
    // step test
    let input = Tensor::randn([batch_dim, input_dim], kind::FLOAT_CPU);
    let nn::LSTMState((h, c)) = lstm.step(&input, &lstm.zero_state(batch_dim));
    assert_eq!(h.size(), [layer_dim, batch_dim, output_dim]);
    assert_eq!(c.size(), [layer_dim, batch_dim, output_dim]);

    // seq test
    let input = Tensor::randn([batch_dim, seq_len, input_dim], kind::FLOAT_CPU);
    let (output, _) = lstm.seq(&input);
    assert_eq!(output.size(), [batch_dim, seq_len, output_dim * num_directions]);
}

#[test]
fn lstm() {
    lstm_test(Default::default());
    lstm_test(nn::RNNConfig { has_biases: false, ..Default::default() });
    lstm_test(nn::RNNConfig { bidirectional: true, ..Default::default() });
    lstm_test(nn::RNNConfig { num_layers: 2, ..Default::default() });
    lstm_test(nn::RNNConfig { num_layers: 2, bidirectional: true, ..Default::default() });
}

fn embedding_test(embedding_config: nn::EmbeddingConfig) {
    let batch_dim = 5;
    let seq_len = 7;
    let input_dim = 10;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let embeddings = nn::embedding(vs.root(), input_dim, output_dim, embedding_config);

    // forward test
    let input = Tensor::randint(10, [batch_dim, seq_len], kind::INT64_CPU);
    let output = embeddings.forward(&input);
    assert_eq!(output.size(), [batch_dim, seq_len, output_dim]);

    // padding test
    let padding_idx = if embedding_config.padding_idx < 0 {
        input_dim + embedding_config.padding_idx
    } else {
        embedding_config.padding_idx
    };
    let input = Tensor::from_slice(&[padding_idx]);
    let output = embeddings.forward(&input);
    assert_eq!(output.size(), [1, output_dim]);
    assert_eq!(output.get(0), embeddings.ws.get(padding_idx));
}

#[test]
fn embedding_default() {
    embedding_test(Default::default());
}

#[test]
fn embedding_neg_padding() {
    embedding_test(nn::EmbeddingConfig { padding_idx: -1, ..Default::default() });
}

#[test]
fn embedding_zero_padding() {
    embedding_test(nn::EmbeddingConfig { padding_idx: 0, ..Default::default() });
}

fn linear_test(linear_config: nn::LinearConfig) {
    let batch_dim = 5;
    let input_dim = 10;
    let output_dim = 4;
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(vs.root(), input_dim, output_dim, linear_config);

    // forward test
    let input = Tensor::randint(10, [batch_dim, input_dim], kind::FLOAT_CPU);
    let expected_var_store_size = if linear_config.bias { 2 } else { 1 };

    let output = linear.forward(&input);
    assert_eq!(output.size(), [batch_dim, output_dim]);

    assert_eq!(output.size(), [batch_dim, output_dim]);
    assert_eq!(vs.variables().len(), expected_var_store_size);
    assert!(vs.variables().contains_key("weight"));
    assert_eq!(vs.variables().contains_key("bias"), linear_config.bias);
}

#[test]
fn linear() {
    linear_test(Default::default());
    linear_test(nn::LinearConfig { bias: true, ..Default::default() });
    linear_test(nn::LinearConfig { bias: false, ..Default::default() });
}

#[test]
fn pad() {
    let xs = Tensor::from_slice(&[1., 2., 3.]);
    let padded = nn::PaddingMode::Zeros.pad(&xs, &[1, 1]);
    assert_eq!(vec_f32_from(&padded), [0., 1., 2., 3., 0.]);

    let xs = Tensor::from_slice(&[1., 2., 3.]).view([1, 3]);
    let padded = nn::PaddingMode::Zeros.pad(&xs, &[1, 1]);
    assert_eq!(vec_f32_from(&padded.reshape(-1)), [0., 1., 2., 3., 0.]);

    let xs = Tensor::from_slice(&[1., 2., 3., 4.]).view([1, 2, 2]);
    let padded = nn::PaddingMode::Reflect.pad(&xs, &[1, 1, 1, 1]);
    assert_eq!(
        vec_f32_from(&padded.reshape(-1)),
        &[4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0]
    );
    let padded = nn::PaddingMode::Reflect.pad(&xs, &[1, 1, 1, 1]);
    assert_eq!(
        vec_f32_from(&padded.reshape(-1)),
        &[4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0]
    );
    let padded = nn::PaddingMode::Reflect.pad(&xs, &[1, 1, 1, 1]);
    assert_eq!(
        vec_f32_from(&padded.reshape(-1)),
        &[4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 1.0]
    );
}

fn apply_conv(xs: &Tensor, padding_mode: nn::PaddingMode) -> Tensor {
    let vs = nn::VarStore::new(Device::Cpu);
    let conv_cfg = nn::ConvConfig { padding: 1, bias: false, padding_mode, ..Default::default() };
    let mut conv = nn::conv2d(vs.root(), 1, 1, 3, conv_cfg);
    tch::no_grad(|| {
        _ = conv.ws.fill_(1.);
    });
    conv.forward(xs)
}

#[test]
fn conv() {
    let xs = Tensor::from_slice(&[1f32, 2., 3., 4.]).view([1, 1, 2, 2]); // NCHW

    let conved = apply_conv(&xs, nn::PaddingMode::Zeros);
    assert_eq!(vec_f32_from(&conved.reshape(-1)), &[10.0, 10.0, 10.0, 10.0]);

    let conved = apply_conv(&xs, nn::PaddingMode::Reflect);
    assert_eq!(vec_f32_from(&conved.reshape(-1)), &[27.0, 24.0, 21.0, 18.0]);

    let conved = apply_conv(&xs, nn::PaddingMode::Circular);
    assert_eq!(vec_f32_from(&conved.reshape(-1)), &[27.0, 24.0, 21.0, 18.0]);

    let conved = apply_conv(&xs, nn::PaddingMode::Replicate);
    assert_eq!(vec_f32_from(&conved.reshape(-1)), &[18.0, 21.0, 24.0, 27.0]);
}

#[test]
fn seq() {
    let s = nn::seq().add_fn(|xs| xs.shallow_clone().relu_());
    let xs = Tensor::from_slice(&[1.0, -1.0, 2.0]);
    let ys = xs.apply(&s);
    assert_eq!(vec_f32_from(&xs), [1., 0., 2.]);
    assert_eq!(vec_f32_from(&ys), [1., 0., 2.]);
}
