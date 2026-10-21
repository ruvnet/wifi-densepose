use anyhow::Result;
use tch::nn::{Adam, ModuleT, OptimizerConfig, VarStore};
use tch::vision::dataset::Dataset;
use tch::TrainableCModule;
use tch::{CModule, Device};

fn train_and_save_model(dataset: &Dataset, device: Device) -> Result<()> {
    let vs = VarStore::new(device);
    let mut trainable = TrainableCModule::load("model.pt", vs.root())?;
    trainable.set_train();
    let initial_acc = trainable.batch_accuracy_for_logits(
        &dataset.test_images,
        &dataset.test_labels,
        vs.device(),
        1024,
    );
    println!("Initial accuracy: {:5.2}%", 100. * initial_acc);

    let mut opt = Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..20 {
        for (images, labels) in dataset.train_iter(128).shuffle().to_device(vs.device()).take(50) {
            let loss = trainable.forward_t(&images, true).cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
        }
        let test_accuracy = trainable.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            vs.device(),
            1024,
        );
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
    trainable.save("trained_model.pt")?;
    Ok(())
}

fn load_trained_and_test_acc(dataset: &Dataset, device: Device) -> Result<()> {
    let mut module = CModule::load_on_device("trained_model.pt", device)?;
    module.set_eval();
    let accuracy =
        module.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, device, 1024);
    println!("Updated accuracy: {:5.2}%", 100. * accuracy);
    Ok(())
}

fn main() -> Result<()> {
    let dataset = tch::vision::mnist::load_dir("data")?;
    let dataset = Dataset {
        train_images: dataset.train_images.view([-1, 1, 28, 28]),
        test_images: dataset.test_images.view([-1, 1, 28, 28]),
        ..dataset
    };
    let device = Device::Cpu;
    train_and_save_model(&dataset, device)?;
    load_trained_and_test_acc(&dataset, device)?;
    Ok(())
}
