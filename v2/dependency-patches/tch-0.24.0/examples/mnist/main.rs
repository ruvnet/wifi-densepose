/* Some very simple models trained on the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz
*/

use anyhow::Result;

mod mnist_conv;
mod mnist_linear;
mod mnist_nn;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = if args.len() < 2 { None } else { Some(args[1].as_str()) };
    match model {
        None => mnist_nn::run(),
        Some("linear") => mnist_linear::run(),
        Some("conv") => mnist_conv::run(),
        Some(_) => mnist_nn::run(),
    }
}
