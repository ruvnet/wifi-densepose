// This should rearch 97% accuracy.

mod sparse_adam;

use anyhow::Result;
use tch::{nn, nn::Module, Device};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "layer1", IMAGE_DIM, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

pub fn run() -> Result<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root());

    // force a sparse update step (in order to test on dense problem)
    let force_sparse = false;
    // create a custom optimizer with learning rate `0.005`, beta_1 `0.9` and beta2 `0.999`
    let mut opt = sparse_adam::SparseAdam::new(&vs, 5e-3, 0.9, 0.999, 1e-8, force_sparse);

    for epoch in 1..200 {
        let loss = net.forward(&m.train_images).cross_entropy_for_logits(&m.train_labels);

        // call custom optimizer
        opt.zero_grad();
        loss.backward();
        opt.step();

        let test_accuracy = net.forward(&m.test_images).accuracy_for_logits(&m.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::try_from(&loss)?,
            100. * f64::try_from(&test_accuracy)?,
        );
    }
    Ok(())
}

fn main() {
    run().unwrap();
}
