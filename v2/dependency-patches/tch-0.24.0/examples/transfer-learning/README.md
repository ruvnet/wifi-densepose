# Transfer Learning Tutorial

This tutorial follows the lines of the
[PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

We will use transfer learning to leverage a pretrained ResNet model on a small dataset.
This dataset is made of images of ants and bees that we want to classify,
there are roughly 240 training images and 150 validation images each of them of size
224x224. The dataset is so small that training even the simplest convolutional neural
network on it would be very difficult.

Instead the original tutorial proposes two alternatives to train the classifier.

- *Finetuning the pretrained model.* We start from a ResNet-18 model pretrained on ImageNet
1000 categories, replace the last layer by a binary classifier and train the resulting model
as usual.
- *Using a pretrained model as feature extractor.* The pretrained model weights are frozen and
we run this model and store the outputs of the last layer before the final classifier.
We then train a binary classifier on the resulting features.

We will focus on the second alternative but first we need to get the code
building and running and we also have to download the dataset and pretrained
weights.

## Installation Instructions
Run the following commands to download the latest tch-rs version
and run the tests, this installs the CPU version of libtorch if necessary.

```bash
git clone https://github.com/LaurentMazare/tch-rs.git
cd tch-rs
cargo test
```

The ants and bees dataset can be downloaded [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip).
You can download the weights for a ResNet-18 network pretrained on ImageNet,
[resnet18.ot](https://github.com/LaurentMazare/tch-rs/releases/download/untagged-eb220e5c19f9bb250bd1/resnet18.ot).

Once this is done and the dataset has been extracted we can build and run the code with:
```bash
cargo run --example transfer-learning resnet18.ot hymenoptera_data
```

## Loading the Data

Let us now have a look at the code from `main.rs`.
The dataset is loaded via some helper functions.

```rust
let dataset = imagenet::load_from_dir(dataset_dir)?;
println!("{:?}", dataset);
```

The `println!` macro prints the dimensions of the tensors that have
been created. For training the tensor has shape `211x3x224x224`, this
corresponds to 211 images of height and width both 224 with 3 channels
(PyTorch uses the NCHW ordering for image data). The testing image
tensor has dimensions `127` so there are 127 images with the
same size as used in training.


## Using a Pretrained ResNet as Feature Extractor

The pixel data from the dataset is converted to features by running
a pre-trained ResNet model.

```rust
let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
let net = resnet::resnet18_no_final_layer(&vs.root());
vs.load(weights)?;

let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));
```

This snippet performs the following steps:
- A variable store `vs` is created to hold the network weights.
- A ResNet-18 model is created using this variable store. At this point the
  model weights are randomly initialized.
- `vs.load(weights)` loads the weights stored in a given file and copy their values
  to some tensors. Tensors are named in the serialized file in a way that matches
  the names we used when creating the ResNet model.
- Finally for each tensor of the training and testing datasets, `apply`
  performs a forward pass on the model and returns the resulting tensor. In this
  case the result is a vector of 512 values per sample.
The `no_grad` closure informs PyTorch that there is no need to keep a graph of the
forward pass as we do not plan on asking for gradients.

## Training a Linear Layer on top of the Extracted Features

Now that we have precomputed the output of the ResNet model on our training and
testing images we will train a linear binary classifier to recognize ants vs bees.

We start by defining a model, for this we need a variable store to hold the
trainable variables.

```rust
let vs = tch::nn::VarStore::new(tch::Device::Cpu);
let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
```

We will use stochastic gradient descent to minimize the cross-entropy loss
on the classification task. To do this we create a `sgd` optimizer and then
iterate on the training dataset. After each epoch the accuracy is computed
on the testing set and printed.

```rust
let sgd = nn::Sgd::default().build(&vs, 1e-3)?;
for epoch_idx in 1..1001 {
    let predicted = train_images.apply(&linear);
    let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
    sgd.backward_step(&loss);

    let test_accuracy = test_images
        .apply(&linear)
        .accuracy_for_logits(&dataset.test_labels);
    println!("{} {:.2}%", epoch_idx, 100. * f64::from(test_accuracy));
}
```

On each training step the model output is computed through a forward pass. The
cross-entropy loss is then evaluated on the resulting logits using the training labels.
The backward pass then evaluates gradients for the trainable variables of our
model and these variables are updated by the optimizer.
```rust
    let predicted = train_images.apply(&linear);
    let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
    sgd.backward_step(&loss);
```

After each epoch the accuracy is evaluated on the testing set and printed out.
```rust
    let test_accuracy = test_images
        .apply(&linear)
        .accuracy_for_logits(&dataset.test_labels);
    println!("{} {:.2}%", epoch_idx, 100. * f64::from(test_accuracy));
```

This should result in a `94.5%` accuracy on the testing set.
The whole code for this example can be found in [main.rs](main.rs).

