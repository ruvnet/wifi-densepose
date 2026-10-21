// This example illustrates how to use pre-trained vision models.
// model to get the imagenet label for some image.
//
// The pre-trained weight files containing the pre-trained weights can be found here:
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet34.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/densenet121.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg13.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg16.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/vgg19.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/squeezenet1_0.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/squeezenet1_1.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/alexnet.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/inception-v3.ot
// https://github.com/LaurentMazare/tch-rs/releases/download/mw/mobilenet-v2.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b0.safetensors
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b1.safetensors
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b2.safetensors
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b3.safetensors
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/efficientnet-b4.safetensors
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/convmixer1536_20.ot
// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/convmixer1024_20.ot
// In order to obtain the dinov2 weights, e.g. dinov2_vits14.safetensors, run the
// src/vision/export_dinov2.py
use anyhow::{bail, Context, Result};
use tch::nn::ModuleT;
use tch::vision::{
    alexnet, convmixer, densenet, dinov2, efficientnet, imagenet, inception, mobilenet, resnet,
    squeezenet, vgg,
};

pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (weights, image) = match args.as_slice() {
        [_, w, i] => (std::path::Path::new(w), i.to_owned()),
        _ => bail!("usage: main resnet18.ot image.jpg"),
    };
    // Load the image file and resize it to the usual imagenet dimension of 224x224.
    let image = imagenet::load_image_and_resize224(image)?;

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net: Box<dyn ModuleT> =
        match weights.file_stem().context("no stem")?.to_str().context("invalid stem")? {
            "resnet18" => Box::new(resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT)),
            "resnet34" => Box::new(resnet::resnet34(&vs.root(), imagenet::CLASS_COUNT)),
            "densenet121" => Box::new(densenet::densenet121(&vs.root(), imagenet::CLASS_COUNT)),
            "vgg13" => Box::new(vgg::vgg13(&vs.root(), imagenet::CLASS_COUNT)),
            "vgg16" => Box::new(vgg::vgg16(&vs.root(), imagenet::CLASS_COUNT)),
            "vgg19" => Box::new(vgg::vgg19(&vs.root(), imagenet::CLASS_COUNT)),
            "squeezenet1_0" => Box::new(squeezenet::v1_0(&vs.root(), imagenet::CLASS_COUNT)),
            "squeezenet1_1" => Box::new(squeezenet::v1_1(&vs.root(), imagenet::CLASS_COUNT)),
            "alexnet" => Box::new(alexnet::alexnet(&vs.root(), imagenet::CLASS_COUNT)),
            "inception-v3" => Box::new(inception::v3(&vs.root(), imagenet::CLASS_COUNT)),
            "mobilenet-v2" => Box::new(mobilenet::v2(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b0" => Box::new(efficientnet::b0(&vs.root(), imagenet::CLASS_COUNT)),
            // Maybe the higher resolution models should be handled differently.
            "efficientnet-b1" => Box::new(efficientnet::b1(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b2" => Box::new(efficientnet::b2(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b3" => Box::new(efficientnet::b3(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b4" => Box::new(efficientnet::b4(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b5" => Box::new(efficientnet::b5(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b6" => Box::new(efficientnet::b6(&vs.root(), imagenet::CLASS_COUNT)),
            "efficientnet-b7" => Box::new(efficientnet::b7(&vs.root(), imagenet::CLASS_COUNT)),
            "convmixer1536_20" => Box::new(convmixer::c1536_20(&vs.root(), imagenet::CLASS_COUNT)),
            "convmixer1024_20" => Box::new(convmixer::c1024_20(&vs.root(), imagenet::CLASS_COUNT)),
            "dinov2_vits14" => Box::new(dinov2::vit_small(&vs.root())),
            s => bail!("unknown model for {s}, use a weight file named e.g. resnet18.ot"),
        };
    vs.load(weights)?;

    // Apply the forward pass of the model to get the logits.
    let output =
        net.forward_t(&image.unsqueeze(0), /* train= */ false).softmax(-1, tch::Kind::Float); // Convert to probability.

    // Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}
