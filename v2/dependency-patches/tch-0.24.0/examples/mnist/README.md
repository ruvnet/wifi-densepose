The examples in this directory have been adapted from the [TensorFlow
tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html).
To execute these examples, you will have to unzip the [MNIST data
files](http://yann.lecun.com/exdb/mnist/) in `data/`.

## Linear Classifier

The code can be found in `mnist_linear.ml`.

We first load the MNIST data. This is done using the MNIST helper module,
labels are returned as a vector of integer.  Train images and labels are used
when training the model.  Test images and labels are used to estimate the
validation error.

```rust
let m = vision::mnist::load_dir("data").unwrap();
```

After that two tensors are initialized to hold the weights and biases for the
linear classifier. `requires_grad` is used when creating the tensors to inform
torch that we will compute some gradients with respect to these tensors.

```rust
let mut ws = Tensor::zeros(&[IMAGE_DIM, LABELS], kind::FLOAT_CPU).set_requires_grad(true);
let mut bs = Tensor::zeros(&[LABELS], kind::FLOAT_CPU).set_requires_grad(true);
```

Using these the model is defined as multiplying an input by the weight matrix
and adding the bias. A softmax function is used to transform the output into a
probability distribution.

```rust
    let logits = m.train_images.mm(&ws) + &bs;
```

We use gradient descent to minimize cross-entropy with respect to variables
`ws` and `bs` and iterate this a couple hundred times.

Rather than using an optimizer we perform the gradient descent updates manually.
This is only to illustrate how gradients can be computed and used. Other examples
such as `mnist_nn.rs` or `mnist_conv.rs` use an Adam optimizer.

```rust
    for epoch in 1..200 {
        let logits = m.train_images.mm(&ws) + &bs;
        let loss = logits.log_softmax(-1).nll_loss(&m.train_labels);
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        no_grad(|| {
            ws += ws.grad() * (-1);
            bs += bs.grad() * (-1);
        });
```

Running this code should build a model that has ~92% accuracy.

## A Simple Neural-Network

The code can be found in `mnist_nn.rs`, accuracy should reach ~96%.

## Convolutional Neural-Network

The code can be found in `mnist_conv.rs`, accuracy should reach ~99%.

When building models with multiple weights and bias parameters we use
a variable store to keep track of these variables and let the optimizer
know about them. The variable store is created in the first line of the
following snippet, then the model is built using this variable store
and finally we create an optimizer that will performs gradient descent
over the parameters that have been added to the variable store when
creating the network.

```rust
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());
    let opt = nn::Optimizer::adam(&vs, 1e-4, Default::default());
```

Note that this will automatically run on a GPU when available.
For a convolutional model we cannot run on all the training images in
a single step as this would requires lots of memory. Instead we
run on mini-batches. An iterator makes it easy to loop over all
the training images shuffled and grouped by mini-batches.
This is done in the main training loop:

```rust
    for epoch in 1..100 {
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
```

