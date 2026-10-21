//! A simple dataset structure shared by various computer vision datasets.
use crate::data::Iter2;
use crate::{IndexOp, Tensor};
use rand::Rng;

#[derive(Debug)]
pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: i64,
}

impl Dataset {
    pub fn train_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.train_images, &self.train_labels, batch_size)
    }

    pub fn test_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.test_images, &self.test_labels, batch_size)
    }
}

/// Randomly applies horizontal flips
/// This expects a 4 dimension NCHW tensor and returns a tensor with
/// an identical shape.
pub fn random_flip(t: &Tensor) -> Tensor {
    let size = t.size();
    if size.len() != 4 {
        panic!("unexpected shape for tensor {t:?}")
    }
    let output = t.zeros_like();
    for batch_index in 0..size[0] {
        let mut output_view = output.i(batch_index);
        let t_view = t.i(batch_index);
        let src = if rand::random() { t_view } else { t_view.flip([2]) };
        output_view.copy_(&src)
    }
    output
}

/// Pad the image using reflections and take some random crops.
/// This expects a 4 dimension NCHW tensor and returns a tensor with
/// an identical shape.
pub fn random_crop(t: &Tensor, pad: i64) -> Tensor {
    let size = t.size();
    if size.len() != 4 {
        panic!("unexpected shape for tensor {t:?}")
    }
    let sz_h = size[2];
    let sz_w = size[3];
    let padded = t.reflection_pad2d([pad, pad, pad, pad]);
    let output = t.zeros_like();
    for bindex in 0..size[0] {
        let mut output_view = output.i(bindex);
        let start_w = rand::thread_rng().gen_range(0..2 * pad);
        let start_h = rand::thread_rng().gen_range(0..2 * pad);
        let src = padded.i((bindex, .., start_h..start_h + sz_h, start_w..start_w + sz_w));
        output_view.copy_(&src)
    }
    output
}

/// Applies cutout: randomly remove some square areas in the original images.
/// <https://arxiv.org/abs/1708.04552>
pub fn random_cutout(t: &Tensor, sz: i64) -> Tensor {
    let size = t.size();
    if size.len() != 4 || sz > size[2] || sz > size[3] {
        panic!("unexpected shape for tensor {t:?} {sz}")
    }
    let mut output = t.zeros_like();
    output.copy_(t);
    for bindex in 0..size[0] {
        let start_h = rand::thread_rng().gen_range(0..size[2] - sz + 1);
        let start_w = rand::thread_rng().gen_range(0..size[3] - sz + 1);
        let _output =
            output.i((bindex, .., start_h..start_h + sz, start_w..start_w + sz)).fill_(0.0);
    }
    output
}

pub fn augmentation(t: &Tensor, flip: bool, crop: i64, cutout: i64) -> Tensor {
    let mut t = t.shallow_clone();
    if flip {
        t = random_flip(&t);
    }
    if crop > 0 {
        t = random_crop(&t, crop);
    }
    if cutout > 0 {
        t = random_cutout(&t, cutout);
    }
    t
}
