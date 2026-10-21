// This should rearch 91.5% accuracy.

use anyhow::Result;
use tch::{kind, no_grad, vision, Kind, Tensor};

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

pub fn run() -> Result<()> {
    let m = vision::mnist::load_dir("data")?;
    println!("train-images: {:?}", m.train_images.size());
    println!("train-labels: {:?}", m.train_labels.size());
    println!("test-images: {:?}", m.test_images.size());
    println!("test-labels: {:?}", m.test_labels.size());
    let mut ws = Tensor::zeros([IMAGE_DIM, LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    let mut bs = Tensor::zeros([LABELS], kind::FLOAT_CPU).set_requires_grad(true);
    for epoch in 1..200 {
        let logits = m.train_images.mm(&ws) + &bs;
        let loss = logits.log_softmax(-1, Kind::Float).nll_loss(&m.train_labels);
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        no_grad(|| {
            ws += ws.grad() * (-1);
            bs += bs.grad() * (-1);
        });
        let test_logits = m.test_images.mm(&ws) + &bs;
        let test_accuracy = test_logits
            .argmax(Some(-1), false)
            .eq_tensor(&m.test_labels)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100. * test_accuracy
        );
    }
    Ok(())
}
