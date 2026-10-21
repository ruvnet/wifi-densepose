use tch::{nn, vision, Tensor};

#[test]
fn mobilenet() {
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let net = vision::mobilenet::v2(&vs.root(), 1000);
    let img = Tensor::zeros([1, 3, 224, 224], tch::kind::FLOAT_CPU);
    let logits = img.apply_t(&net, false);
    assert_eq!(logits.size(), [1, 1000]);
}

#[test]
fn resize() {
    // Check that resizing returns a tensor with the appropriate dimensions.
    let img = Tensor::zeros([3, 8, 12], tch::kind::FLOAT_CPU);
    let resized_img = vision::image::resize(&img, 16, 16).unwrap();
    assert_eq!(resized_img.size(), [3, 16, 16]);
    let resized_img = vision::image::resize(&img, 4, 16).unwrap();
    assert_eq!(resized_img.size(), [3, 16, 4]);
    let resized_img = vision::image::resize(&img, 32, 8).unwrap();
    assert_eq!(resized_img.size(), [3, 8, 32]);
}

#[test]
fn resize_preserve_aspect_ratio() {
    // Check that resizing returns a tensor with the appropriate dimensions.
    let img = Tensor::zeros([3, 8, 12], tch::kind::FLOAT_CPU);
    let resized_img = vision::image::resize_preserve_aspect_ratio(&img, 16, 16).unwrap();
    assert_eq!(resized_img.size(), [3, 16, 16]);
    let resized_img = vision::image::resize(&img, 4, 16).unwrap();
    assert_eq!(resized_img.size(), [3, 16, 4]);
    let resized_img = vision::image::resize(&img, 32, 8).unwrap();
    assert_eq!(resized_img.size(), [3, 8, 32]);
}
