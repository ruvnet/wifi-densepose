//! The `vision` module groups functions and models related to
//! computer vision.
pub mod dataset;

pub mod image;

pub mod mnist;

pub mod cifar;

pub mod alexnet;

pub mod inception;

pub mod resnet;

pub mod densenet;

pub mod vgg;

pub mod squeezenet;

pub mod mobilenet;

pub mod imagenet;

pub mod efficientnet;

pub mod convmixer;

pub mod dinov2;

#[cfg(feature = "image")]
mod rust_image;
