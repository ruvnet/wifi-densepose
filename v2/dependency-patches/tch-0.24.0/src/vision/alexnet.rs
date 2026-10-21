//! AlexNet.
//! <https://arxiv.org/abs/1404.5997>
use crate::{nn, nn::Conv2D, nn::ModuleT, Tensor};

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig { stride, padding, ..Default::default() };
    nn::conv2d(p, c_in, c_out, ksize, conv2d_cfg)
}

fn max_pool2d(xs: Tensor, ksize: i64, stride: i64) -> Tensor {
    xs.max_pool2d([ksize, ksize], [stride, stride], [0, 0], [1, 1], false)
}

fn features(p: nn::Path) -> impl ModuleT {
    nn::seq_t()
        .add(conv2d(&p / "0", 3, 64, 11, 2, 4))
        .add_fn(|xs| max_pool2d(xs.relu(), 3, 2))
        .add(conv2d(&p / "3", 64, 192, 5, 1, 2))
        .add_fn(|xs| max_pool2d(xs.relu(), 3, 2))
        .add(conv2d(&p / "6", 192, 384, 3, 1, 1))
        .add_fn(|xs| xs.relu())
        .add(conv2d(&p / "8", 384, 256, 3, 1, 1))
        .add_fn(|xs| xs.relu())
        .add(conv2d(&p / "10", 256, 256, 3, 1, 1))
        .add_fn(|xs| max_pool2d(xs.relu(), 3, 2))
}

fn classifier(p: nn::Path, nclasses: i64) -> impl ModuleT {
    nn::seq_t()
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::linear(&p / "1", 256 * 6 * 6, 4096, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::linear(&p / "4", 4096, 4096, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&p / "6", 4096, nclasses, Default::default()))
}

pub fn alexnet(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    nn::seq_t()
        .add(features(p / "features"))
        .add_fn(|xs| xs.adaptive_avg_pool2d([6, 6]).flat_view())
        .add(classifier(p / "classifier", nclasses))
}
