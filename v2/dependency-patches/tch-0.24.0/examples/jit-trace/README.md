To run this example, you will have to unzip the [MNIST data
files](http://yann.lecun.com/exdb/mnist/) in `data/`.

# Saving a trained model by tracing

This tutorial illustrates how to serialize a model trained in Rust as a TorchScript program.

## Training and serializing a TorchScript program in Rust.

By running the following command, you can train a model on the MNIST dataset and serialize it as a TorchScript program in Rust.

```bash
cargo run --example jit-trace
```

It will save the model as `model.pt`.

## Loading the model in Python

You can load and run the saved model in Python. 
`test.py` is an example Python script to load the model from `model.pt` and run it on the MNIST dataset.

You can run the script by the following command. 
Please ensure that `torch` and `torchvision` are installed in your environment before running.
```bash
python ./examples/jit-trace/test.py
```
