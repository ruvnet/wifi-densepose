// This example illustrates how to use transfer learning to fine tune a pre-trained
// imagenet model on another dataset.
//
// The pre-trained weight files containing the pre-trained weights can be found here:
// https://github.com/LaurentMazare/tch-rs/releases/download/untagged-eb220e5c19f9bb250bd1/resnet18.ot
use anyhow::{bail, Result};
use tch::nn::{self, OptimizerConfig};
use tch::vision::{imagenet, resnet};

pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (weights, dataset_dir) = match args.as_slice() {
        [_, w, d] => (std::path::Path::new(w), d.to_owned()),
        _ => bail!("usage: main resnet18.ot dataset-path"),
    };
    // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    let dataset = imagenet::load_from_dir(dataset_dir)?;
    println!("{dataset:?}");

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = resnet::resnet18_no_final_layer(&vs.root());
    vs.load(weights)?;

    // Pre-compute the final activations.
    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    for epoch_idx in 1..1001 {
        let predicted = train_images.apply(&linear);
        let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
        sgd.backward_step(&loss);

        let test_accuracy = test_images.apply(&linear).accuracy_for_logits(&dataset.test_labels);
        println!("{} {:.2}%", epoch_idx, 100. * f64::try_from(test_accuracy)?);
    }
    Ok(())
}
