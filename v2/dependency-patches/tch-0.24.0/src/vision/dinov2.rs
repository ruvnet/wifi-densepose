//! DINOv2: Learning Robust Visual Features without Supervision
//! https://github.com/facebookresearch/dinov2
//! The weights can be extracted from pre-trained Python models
//! using `python src/vision/export_dinov2.py`.
// TODO: use swiglu.
use crate::{nn, IndexOp, Kind, Tensor};

const IMG_SIZE: i64 = 518;
const PATCH_SIZE: i64 = 14;
const NUM_CLASSES: i64 = 1000;

#[derive(Debug)]
struct Attention {
    qkv: nn::Linear,
    proj: nn::Linear,
    num_heads: i64,
    scale: f64,
}

impl Attention {
    fn new(vs: nn::Path, dim: i64, num_heads: i64, qkv_bias: bool, proj_bias: bool) -> Self {
        let qkv_config = nn::LinearConfig { bias: qkv_bias, ..Default::default() };
        let proj_config = nn::LinearConfig { bias: proj_bias, ..Default::default() };
        let qkv = nn::linear(&vs / "qkv", dim, dim * 3, qkv_config);
        let proj = nn::linear(&vs / "proj", dim, dim, proj_config);
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Self { qkv, proj, num_heads, scale }
    }
}

impl nn::Module for Attention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (b, n, c) = xs.size3().unwrap();
        let qkv = self
            .qkv
            .forward(xs)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let q = qkv.get(0) * self.scale;
        let k = qkv.get(1);
        let v = qkv.get(2);
        let attn = q.matmul(&k.transpose(-2, -1)).softmax(-1, Kind::Float);
        attn.matmul(&v).transpose(1, 2).reshape([b, n, c]).apply(&self.proj)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vs: nn::Path, dim: i64) -> Self {
        let gamma = vs.var("gamma", &[dim], nn::Init::Const(0.));
        Self { gamma }
    }
}

impl nn::Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs * &self.gamma
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Mlp {
    fn new(vs: nn::Path, in_features: i64, hidden_features: i64, bias: bool) -> Self {
        let out_features = in_features;
        let config = nn::LinearConfig { bias, ..Default::default() };
        let fc1 = nn::linear(&vs / "fc1", in_features, hidden_features, config);
        let fc2 = nn::linear(&vs / "fc2", hidden_features, out_features, config);
        Self { fc1, fc2 }
    }
}

impl nn::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).gelu("none").apply(&self.fc2)
    }
}

#[derive(Debug)]
struct Block {
    norm1: nn::LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: nn::LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vs: nn::Path, dim: i64, num_heads: i64) -> Self {
        let norm1 = nn::layer_norm(&vs / "norm1", vec![dim], Default::default());
        let attn = Attention::new(&vs / "attn", dim, num_heads, true, true);
        let ls1 = LayerScale::new(&vs / "ls1", dim);
        let norm2 = nn::layer_norm(&vs / "norm2", vec![dim], Default::default());
        let mlp = Mlp::new(&vs / "mlp", dim, dim * 4, true);
        let ls2 = LayerScale::new(&vs / "ls2", dim);
        Self { norm1, attn, ls1, norm2, mlp, ls2 }
    }
}

impl nn::Module for Block {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs + xs.apply(&self.norm1).apply(&self.attn).apply(&self.ls1);
        &xs + xs.apply(&self.norm2).apply(&self.mlp).apply(&self.ls2)
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: nn::Conv2D,
    patch_size: (i64, i64),
    num_patches: i64,
}

impl PatchEmbed {
    fn new(vs: nn::Path, img_size: i64, patch_size: i64, in_chans: i64, embed_dim: i64) -> Self {
        let config = nn::ConvConfig { stride: patch_size, ..Default::default() };
        let proj = nn::conv2d(vs / "proj", in_chans, embed_dim, patch_size, config);
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Self { proj, patch_size: (patch_size, patch_size), num_patches }
    }
}

impl nn::Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (_b, _c, h, w) = xs.size4().unwrap();
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            panic!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            panic!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = xs.apply(&self.proj);
        let (b, c, h, w) = xs.size4().unwrap();
        // flatten embeddings.
        xs.reshape([b, c, h * w]).transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: nn::LayerNorm,
    head: nn::Linear,
}

impl DinoVisionTransformer {
    pub fn new(vs: &nn::Path, depth: usize, embed_dim: i64, num_heads: i64) -> Self {
        let patch_embed = PatchEmbed::new(vs / "patch_embed", IMG_SIZE, PATCH_SIZE, 3, embed_dim);
        let cls_token = vs.var("cls_token", &[1, 1, embed_dim], nn::Init::Const(0.));
        let num_tokens = 1;
        let pos_embed = vs.var(
            "pos_embed",
            &[1, patch_embed.num_patches + num_tokens, embed_dim],
            nn::Init::Const(0.),
        );
        let head = nn::linear(vs / "head", 2 * embed_dim, NUM_CLASSES, Default::default());
        let norm = nn::layer_norm(vs / "norm", vec![embed_dim], Default::default());
        let blocks =
            (0..depth).map(|i| Block::new(vs / "blocks" / i, embed_dim, num_heads)).collect();
        Self { patch_embed, cls_token, pos_embed, blocks, norm, head }
    }

    pub fn interpolate_pos_encoding(&self, xs: &Tensor, w: i64, h: i64) -> Tensor {
        let npatch = xs.size()[1] - 1;
        let n = self.pos_embed.size()[1] - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return xs.shallow_clone();
        }
        let class_pos_embed = self.pos_embed.i((.., ..1));
        let patch_pos_embed = self.pos_embed.i((.., 1..));
        let dim = *xs.size().last().unwrap();
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape([1, sqrt_n as i64, sqrt_n as i64, dim])
            .permute([0, 3, 1, 2])
            .upsample_bicubic2d([w0 as i64, h0 as i64], false, w0 / sqrt_n, h0 / sqrt_n);
        let patch_pos_embed = patch_pos_embed.permute([0, 2, 3, 1]).reshape([1, -1, dim]);
        Tensor::cat(&[class_pos_embed, patch_pos_embed], 1)
    }

    pub fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Tensor {
        let (b, _nc, w, h) = xs.size4().unwrap();
        let xs = xs.apply(&self.patch_embed);
        let xs = Tensor::concat(&[self.cls_token.expand([b, -1, -1], false), xs], 1);
        &xs + &self.interpolate_pos_encoding(&xs, w, h)
    }

    pub fn extract_features(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.prepare_tokens_with_mask(xs);
        for blk in self.blocks.iter() {
            xs = xs.apply(blk)
        }
        let xs = xs.apply(&self.norm);
        let xs_norm_clstoken = xs.i((.., 0));
        let xs_norm_patchtokens = xs.i((.., 1..)).mean_dim(1, false, None);
        Tensor::concat(&[xs_norm_clstoken, xs_norm_patchtokens], -1)
    }
}

impl nn::Module for DinoVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.extract_features(xs).apply(&self.head)
    }
}

pub fn vit_small(vs: &nn::Path) -> DinoVisionTransformer {
    DinoVisionTransformer::new(vs, 12, 384, 6)
}

pub fn vit_base(vs: &nn::Path) -> DinoVisionTransformer {
    DinoVisionTransformer::new(vs, 12, 768, 12)
}

pub fn vit_large(vs: &nn::Path) -> DinoVisionTransformer {
    DinoVisionTransformer::new(vs, 24, 1024, 16)
}

pub fn vit_giant(vs: &nn::Path) -> DinoVisionTransformer {
    DinoVisionTransformer::new(vs, 40, 1536, 24)
}
