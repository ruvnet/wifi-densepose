//! InceptionV3.
use crate::{nn, nn::ModuleT, Tensor};

fn conv_bn(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, pad: i64, stride: i64) -> impl ModuleT {
    let conv2d_cfg = nn::ConvConfig { stride, padding: pad, bias: false, ..Default::default() };
    let bn_cfg = nn::BatchNormConfig { eps: 0.001, ..Default::default() };
    nn::seq_t()
        .add(nn::conv2d(&p / "conv", c_in, c_out, ksize, conv2d_cfg))
        .add(nn::batch_norm2d(&p / "bn", c_out, bn_cfg))
        .add_fn(|xs| xs.relu())
}

fn conv_bn2(p: nn::Path, c_in: i64, c_out: i64, ksize: [i64; 2], pad: [i64; 2]) -> impl ModuleT {
    let conv2d_cfg =
        nn::ConvConfigND::<[i64; 2]> { padding: pad, bias: false, ..Default::default() };
    let bn_cfg = nn::BatchNormConfig { eps: 0.001, ..Default::default() };
    nn::seq_t()
        .add(nn::conv(&p / "conv", c_in, c_out, ksize, conv2d_cfg))
        .add(nn::batch_norm2d(&p / "bn", c_out, bn_cfg))
        .add_fn(|xs| xs.relu())
}

fn max_pool2d(xs: &Tensor, ksize: i64, stride: i64) -> Tensor {
    xs.max_pool2d([ksize, ksize], [stride, stride], [0, 0], [1, 1], false)
}

fn inception_a(p: nn::Path, c_in: i64, c_pool: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 64, 1, 0, 1);
    let b2_1 = conv_bn(&p / "branch5x5_1", c_in, 48, 1, 0, 1);
    let b2_2 = conv_bn(&p / "branch5x5_2", 48, 64, 5, 2, 1);
    let b3_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 64, 1, 0, 1);
    let b3_2 = conv_bn(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let b3_3 = conv_bn(&p / "branch3x3dbl_3", 96, 96, 3, 1, 1);
    let bpool = conv_bn(&p / "branch_pool", c_in, c_pool, 1, 0, 1);
    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr);
        let b3 = xs.apply_t(&b3_1, tr).apply_t(&b3_2, tr).apply_t(&b3_3, tr);
        let bpool = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, 9).apply_t(&bpool, tr);
        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

fn inception_b(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch3x3", c_in, 384, 3, 0, 2);
    let b2_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 64, 1, 0, 1);
    let b2_2 = conv_bn(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let b2_3 = conv_bn(&p / "branch3x3dbl_3", 96, 96, 3, 0, 2);
    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr);
        let bpool = max_pool2d(xs, 3, 2);
        Tensor::cat(&[b1, b2, bpool], 1)
    })
}

fn inception_c(p: nn::Path, c_in: i64, c7: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 192, 1, 0, 1);

    let b2_1 = conv_bn(&p / "branch7x7_1", c_in, c7, 1, 0, 1);
    let b2_2 = conv_bn2(&p / "branch7x7_2", c7, c7, [1, 7], [0, 3]);
    let b2_3 = conv_bn2(&p / "branch7x7_3", c7, 192, [7, 1], [3, 0]);

    let b3_1 = conv_bn(&p / "branch7x7dbl_1", c_in, c7, 1, 0, 1);
    let b3_2 = conv_bn2(&p / "branch7x7dbl_2", c7, c7, [7, 1], [3, 0]);
    let b3_3 = conv_bn2(&p / "branch7x7dbl_3", c7, c7, [1, 7], [0, 3]);
    let b3_4 = conv_bn2(&p / "branch7x7dbl_4", c7, c7, [7, 1], [3, 0]);
    let b3_5 = conv_bn2(&p / "branch7x7dbl_5", c7, 192, [1, 7], [0, 3]);

    let bpool = conv_bn(&p / "branch_pool", c_in, 192, 1, 0, 1);

    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr);
        let b3 = xs
            .apply_t(&b3_1, tr)
            .apply_t(&b3_2, tr)
            .apply_t(&b3_3, tr)
            .apply_t(&b3_4, tr)
            .apply_t(&b3_5, tr);
        let bpool = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, 9).apply_t(&bpool, tr);
        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

fn inception_d(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1_1 = conv_bn(&p / "branch3x3_1", c_in, 192, 1, 0, 1);
    let b1_2 = conv_bn(&p / "branch3x3_2", 192, 320, 3, 0, 2);

    let b2_1 = conv_bn(&p / "branch7x7x3_1", c_in, 192, 1, 0, 1);
    let b2_2 = conv_bn2(&p / "branch7x7x3_2", 192, 192, [1, 7], [0, 3]);
    let b2_3 = conv_bn2(&p / "branch7x7x3_3", 192, 192, [7, 1], [3, 0]);
    let b2_4 = conv_bn(&p / "branch7x7x3_4", 192, 192, 3, 0, 2);

    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1_1, tr).apply_t(&b1_2, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr).apply_t(&b2_4, tr);
        let bpool = max_pool2d(xs, 3, 2);
        Tensor::cat(&[b1, b2, bpool], 1)
    })
}

fn inception_e(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 320, 1, 0, 1);

    let b2_1 = conv_bn(&p / "branch3x3_1", c_in, 384, 1, 0, 1);
    let b2_2a = conv_bn2(&p / "branch3x3_2a", 384, 384, [1, 3], [0, 1]);
    let b2_2b = conv_bn2(&p / "branch3x3_2b", 384, 384, [3, 1], [1, 0]);

    let b3_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 448, 1, 0, 1);
    let b3_2 = conv_bn(&p / "branch3x3dbl_2", 448, 384, 3, 1, 1);
    let b3_3a = conv_bn2(&p / "branch3x3dbl_3a", 384, 384, [1, 3], [0, 1]);
    let b3_3b = conv_bn2(&p / "branch3x3dbl_3b", 384, 384, [3, 1], [1, 0]);

    let bpool = conv_bn(&p / "branch_pool", c_in, 192, 1, 0, 1);

    nn::func_t(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);

        let b2 = xs.apply_t(&b2_1, tr);
        let b2 = Tensor::cat(&[b2.apply_t(&b2_2a, tr), b2.apply_t(&b2_2b, tr)], 1);

        let b3 = xs.apply_t(&b3_1, tr).apply_t(&b3_2, tr);
        let b3 = Tensor::cat(&[b3.apply_t(&b3_3a, tr), b3.apply_t(&b3_3b, tr)], 1);

        let bpool = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, true, 9).apply_t(&bpool, tr);

        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

pub fn v3(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    nn::seq_t()
        .add(conv_bn(p / "Conv2d_1a_3x3", 3, 32, 3, 0, 2))
        .add(conv_bn(p / "Conv2d_2a_3x3", 32, 32, 3, 0, 1))
        .add(conv_bn(p / "Conv2d_2b_3x3", 32, 64, 3, 1, 1))
        .add_fn(|xs| max_pool2d(&xs.relu(), 3, 2))
        .add(conv_bn(p / "Conv2d_3b_1x1", 64, 80, 1, 0, 1))
        .add(conv_bn(p / "Conv2d_4a_3x3", 80, 192, 3, 0, 1))
        .add_fn(|xs| max_pool2d(&xs.relu(), 3, 2))
        .add(inception_a(p / "Mixed_5b", 192, 32))
        .add(inception_a(p / "Mixed_5c", 256, 64))
        .add(inception_a(p / "Mixed_5d", 288, 64))
        .add(inception_b(p / "Mixed_6a", 288))
        .add(inception_c(p / "Mixed_6b", 768, 128))
        .add(inception_c(p / "Mixed_6c", 768, 160))
        .add(inception_c(p / "Mixed_6d", 768, 160))
        .add(inception_c(p / "Mixed_6e", 768, 192))
        .add(inception_d(p / "Mixed_7a", 768))
        .add(inception_e(p / "Mixed_7b", 1280))
        .add(inception_e(p / "Mixed_7c", 2048))
        .add_fn_t(|xs, train| xs.adaptive_avg_pool2d([1, 1]).dropout(0.5, train).flat_view())
        .add(nn::linear(p / "fc", 2048, nclasses, Default::default()))
}
