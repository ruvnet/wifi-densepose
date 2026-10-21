//! SqueezeNet implementation.

use crate::{nn, nn::Module, nn::ModuleT, Tensor};

fn max_pool2d(xs: &Tensor) -> Tensor {
    xs.max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], true)
}

fn fire(p: nn::Path, c_in: i64, c_squeeze: i64, c_exp1: i64, c_exp3: i64) -> impl Module {
    let cfg3 = nn::ConvConfig { padding: 1, ..Default::default() };
    let squeeze = nn::conv2d(&p / "squeeze", c_in, c_squeeze, 1, Default::default());
    let exp1 = nn::conv2d(&p / "expand1x1", c_squeeze, c_exp1, 1, Default::default());
    let exp3 = nn::conv2d(&p / "expand3x3", c_squeeze, c_exp3, 3, cfg3);
    nn::func(move |xs| {
        let xs = xs.apply(&squeeze).relu();
        Tensor::cat(&[xs.apply(&exp1).relu(), xs.apply(&exp3).relu()], 1)
    })
}

fn squeezenet(p: &nn::Path, v1_0: bool, nclasses: i64) -> impl ModuleT {
    let f_p = p / "features";
    let c_p = p / "classifier";
    let initial_conv_cfg = nn::ConvConfig { stride: 2, ..Default::default() };
    let final_conv_cfg = nn::ConvConfig { stride: 1, ..Default::default() };
    let features = if v1_0 {
        nn::seq_t()
            .add(nn::conv2d(&f_p / "0", 3, 96, 7, initial_conv_cfg))
            .add_fn(|xs| xs.relu())
            .add_fn(max_pool2d)
            .add(fire(&f_p / "3", 96, 16, 64, 64))
            .add(fire(&f_p / "4", 128, 16, 64, 64))
            .add(fire(&f_p / "5", 128, 32, 128, 128))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "7", 256, 32, 128, 128))
            .add(fire(&f_p / "8", 256, 48, 192, 192))
            .add(fire(&f_p / "9", 384, 48, 192, 192))
            .add(fire(&f_p / "10", 384, 64, 256, 256))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "12", 512, 64, 256, 256))
    } else {
        nn::seq_t()
            .add(nn::conv2d(&f_p / "0", 3, 64, 3, initial_conv_cfg))
            .add_fn(|xs| xs.relu())
            .add_fn(max_pool2d)
            .add(fire(&f_p / "3", 64, 16, 64, 64))
            .add(fire(&f_p / "4", 128, 16, 64, 64))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "6", 128, 32, 128, 128))
            .add(fire(&f_p / "7", 256, 32, 128, 128))
            .add_fn(max_pool2d)
            .add(fire(&f_p / "9", 256, 48, 192, 192))
            .add(fire(&f_p / "10", 384, 48, 192, 192))
            .add(fire(&f_p / "11", 384, 64, 256, 256))
            .add(fire(&f_p / "12", 512, 64, 256, 256))
    };
    features
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(nn::conv2d(&c_p / "1", 512, nclasses, 1, final_conv_cfg))
        .add_fn(|xs| xs.relu().adaptive_avg_pool2d([1, 1]).flat_view())
}

pub fn v1_0(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    squeezenet(p, true, nclasses)
}

pub fn v1_1(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    squeezenet(p, false, nclasses)
}
