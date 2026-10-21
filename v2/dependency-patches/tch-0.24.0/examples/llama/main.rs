// An implementation of LLaMA https://github.com/facebookresearch/llama
// This only contains the inference part.
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/blob/main/tokenizer.json
//
// In order to convert the llama weights to a .safetensors file, run:
// python examples/llama/convert_checkpoint.py ..../LLaMA/7B/consolidated.00.pth
use anyhow::Result;
use clap::Parser;
use tch::nn::{self, Module};
use tch::{Device, Kind, Tensor};

mod sentencepiece;
use sentencepiece::Tokenizer;

const CONTEXT_SIZE: usize = 512;
const START_PROMPT: &str = r"
EDWARD:
I wonder how our princely father 'scaped,
Or whether he be 'scaped away or no
From Clifford's and Northumberland's pursuit:
Had he been ta'en, we should have heard the news;
Had he been slain, we should have heard the news;
Or had he 'scaped, methinks we should have heard
The happy tidings of his good escape.
How fares my brother? why is he so sad?

RICHARD:
I cannot joy, until I be resolved
Where our right valiant father is become.
I saw him in the battle range about;
And watch'd him how he singled Clifford forth.
Methought he bore him in the thickest troop
As doth a lion in a herd of neat;
Or as a bear, encompass'd round with dogs,
Who having pinch'd a few and made them cry,
The rest stand all aloof, and bark at him.
So fared our father with his enemies;
So fled his enemies my warlike father:
Methinks, 'tis prize enough to be his son.
See how the morning opes her golden gates,
And takes her farewell of the glorious sun!
How well resembles it the prime of youth,
Trimm'd like a younker prancing to his love!

EDWARD:
Dazzle mine eyes, or do I see three suns?

RICHARD:
Three glorious suns, each one a perfect sun;
Not separated with the racking clouds,
But sever'd in a pale clear-shining sky.
See, see! they join, embrace, and seem to kiss,
As if they vow'd some league inviolable:
Now are they but one lamp, one light, one sun.
In this the heaven figures some event.

EDWARD:
'Tis wondrous strange, the like yet never heard of.
I think it cites us, brother, to the field,
That we, the sons of brave Plantagenet,
Each one already blazing by our meeds,
Should notwithstanding join our lights together
And over-shine the earth as this the world.
Whate'er it bodes, henceforward will I bear
Upon my target three fair-shining suns.
";

#[allow(dead_code)]
struct Config {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

#[allow(dead_code)]
impl Config {
    fn config_7b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 32, n_head: 32, n_embd: 4096 }
    }

    fn config_13b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 40, n_head: 40, n_embd: 5120 }
    }

    fn config_30b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 60, n_head: 52, n_embd: 6656 }
    }

    fn config_65b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 80, n_head: 64, n_embd: 8192 }
    }
}

#[derive(Debug)]
struct RmsNorm {
    scale: Tensor,
    size: i64,
}

impl RmsNorm {
    fn new(vs: nn::Path, size: i64) -> Self {
        let scale = vs.zeros("scale", &[size]);
        Self { scale, size }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let norm_xs = (xs * xs).mean_dim(-1, true, Kind::Float);
        let xs_normed = xs * (norm_xs + 1e-5).rsqrt();
        let scale = self.scale.reshape([1, 1, self.size]);
        scale * xs_normed
    }
}

#[derive(Debug)]
struct Mlp {
    c_fc1: nn::Linear,
    c_fc2: nn::Linear,
    c_proj: nn::Linear,
}

impl Mlp {
    fn new(vs: nn::Path, n_embd: i64) -> Self {
        let n_hidden = 8 * n_embd / 3;
        let n_hidden = (n_hidden - 1) / 256 * 256 + 256;
        let c = nn::LinearConfig { bias: false, ..Default::default() };
        let c_fc1 = nn::linear(&vs / "c_fc1", n_embd, n_hidden, c);
        let c_fc2 = nn::linear(&vs / "c_fc2", n_embd, n_hidden, c);
        let c_proj = nn::linear(&vs / "c_proj", n_hidden, n_embd, c);
        Self { c_fc1, c_fc2, c_proj }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.c_fc1).silu() * xs.apply(&self.c_fc2);
        xs.apply(&self.c_proj)
    }
}

#[derive(Debug)]
struct CausalSelfAttention {
    c_attn: nn::Linear,
    c_proj: nn::Linear,
    n_head: i64,
    n_embd: i64,
    device: Device,
}

impl CausalSelfAttention {
    fn new(vs: nn::Path, n_head: i64, n_embd: i64) -> Self {
        let c = nn::LinearConfig { bias: false, ..Default::default() };
        let c_attn = nn::linear(&vs / "c_attn", n_embd, 3 * n_embd, c);
        let c_proj = nn::linear(&vs / "c_proj", n_embd, n_embd, c);
        Self { c_attn, c_proj, n_head, n_embd, device: vs.device() }
    }

    fn apply_rotary_emb(&self, x: &Tensor, freqs_cis: &Tensor) -> Tensor {
        let mut dims = x.size();
        let v = dims.pop().unwrap();
        dims.push(v / 2);
        dims.push(2);
        let x = x.reshape(&dims);
        let re_x = x.slice(-1, 0, 1, 1);
        let im_x = x.slice(-1, 1, 2, 1);
        let re_f = freqs_cis.slice(-1, 0, 1, 1);
        let im_f = freqs_cis.slice(-1, 1, 2, 1);
        let re = &re_x * &re_f - &im_x * &im_f;
        let im = &re_x * &im_f + &im_x * &re_f;
        let rope = Tensor::cat(&[&re, &im], -1);
        // TODO: Add the flatten op.
        let mut dims = rope.size();
        let v1 = dims.pop().unwrap();
        let v2 = dims.pop().unwrap();
        dims.push(v1 * v2);
        rope.reshape(&dims)
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Tensor {
        let (b, t, c) = x.size3().unwrap();
        let kind = x.kind();
        let qkv = self.c_attn.forward(x);
        let n_embd = self.n_embd;
        let q = qkv.slice(2, 0, n_embd, 1);
        let k = qkv.slice(2, n_embd, 2 * n_embd, 1);
        let v = qkv.slice(2, 2 * n_embd, 3 * n_embd, 1);
        let target_dim = [b, t, self.n_head, c / self.n_head];
        let k = k.reshape(target_dim).transpose(1, 2);
        let q = q.reshape(target_dim).transpose(1, 2);
        let v = v.reshape(target_dim).transpose(1, 2);
        let q = self.apply_rotary_emb(&q, freqs_cis);
        let k = self.apply_rotary_emb(&k, freqs_cis);
        let k_shape = k.size();
        let att: Tensor = q.matmul(&k.transpose(-2, -1)) / (*k_shape.last().unwrap() as f64).sqrt();
        let mask = Tensor::ones([t, t], (kind, self.device)).tril(0).reshape([1, 1, t, t]);
        let att = att.masked_fill(&mask.eq(0.), f64::NEG_INFINITY);
        let y = att.softmax(-1, kind).matmul(&v);
        let y = y.transpose(1, 2).reshape([b, t, c]);
        self.c_proj.forward(&y)
    }
}

#[derive(Debug)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vs: nn::Path, config: &Config) -> Self {
        let rms_1 = RmsNorm::new(&vs / "rms_1", config.n_embd as i64);
        let attn =
            CausalSelfAttention::new(&vs / "attn", config.n_head as i64, config.n_embd as i64);
        let rms_2 = RmsNorm::new(&vs / "rms_2", config.n_embd as i64);
        let mlp = Mlp::new(&vs / "mlp", config.n_embd as i64);
        Self { rms_1, attn, rms_2, mlp }
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Tensor {
        let x = self.attn.forward(&self.rms_1.forward(x), freqs_cis) + x;
        self.mlp.forward(&self.rms_2.forward(&x)) + x
    }
}

#[derive(Debug)]
struct Llama {
    wte: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: nn::Linear,
}

impl Llama {
    fn new(vs: nn::Path, config: &Config) -> Self {
        let c = nn::LinearConfig { bias: false, ..Default::default() };
        let lm_head =
            nn::linear(&vs / "lm_head", config.n_embd as i64, config.vocab_size as i64, c);
        let wte = nn::embedding(
            &vs / "transformer" / "wte",
            config.vocab_size as i64,
            config.n_embd as i64,
            Default::default(),
        );
        let blocks = (0..config.n_layer)
            .map(|i| Block::new(&vs / "transformer" / "h" / i, config))
            .collect::<Vec<_>>();
        let ln_f = RmsNorm::new(&vs / "transformer" / "ln_f", config.n_embd as i64);
        Self { wte, blocks, ln_f, lm_head }
    }

    fn forward(&self, x: &Tensor, freqs_cis: &Tensor) -> Tensor {
        let (_, t) = x.size2().unwrap();
        let mut x = self.wte.forward(x);
        for block in self.blocks.iter() {
            x = block.forward(&x, freqs_cis);
        }
        let x = self.ln_f.forward(&x);
        let x = x.slice(1, t - 1, t, 1);
        self.lm_head.forward(&x)
    }
}

fn precompute_freqs_cis(config: &Config) -> Tensor {
    let seq_len = CONTEXT_SIZE;
    let n_elem = config.n_embd / config.n_head;
    let theta: Vec<_> =
        (0..n_elem).step_by(2).map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32)).collect();
    let arange: Vec<_> = (0..seq_len).map(|c| c as f32).collect();
    let theta = Tensor::from_slice(&theta);
    let arange = Tensor::from_slice(&arange);
    let idx_theta = arange.outer(&theta);
    let shape = [1, 1, seq_len as i64, n_elem as i64 / 2, 1];
    let idx_theta_cos = idx_theta.cos().reshape(shape);
    let idx_theta_sin = idx_theta.sin().reshape(shape);
    Tensor::cat(&[&idx_theta_cos, &idx_theta_sin], -1)
}

fn llama(vs: nn::Path, args: Args) -> impl Module {
    let config = Config::config_7b();
    let freqs_cis = precompute_freqs_cis(&config).to_device(vs.device());
    let llama = Llama::new(vs, &config);
    nn::func(move |xs| {
        let logits = llama.forward(xs, &freqs_cis);
        (logits / args.temperature).softmax(-1, Kind::Float)
    })
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CompKind {
    Float,
    Half,
    #[clap(name = "bfloat16")]
    BFloat16,
    #[clap(name = "qint8")]
    QInt8,
}

impl CompKind {
    fn to_kind(self) -> Kind {
        match self {
            Self::Float => Kind::Float,
            Self::Half => Kind::Half,
            Self::BFloat16 => Kind::BFloat16,
            Self::QInt8 => Kind::QInt8,
        }
    }
}

#[derive(clap::Parser, Debug, Clone, Copy)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use this type of float for computations, float/half/bfloat16/...
    #[arg(long, value_enum, default_value = "float")]
    kind: CompKind,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,
}

fn main() -> Result<()> {
    let _no_grad = tch::no_grad_guard();
    let args = Args::parse();
    let tokenizer = Tokenizer::from_file("llama-tokenizer.json")?;
    let mut tokens = tokenizer.encode(START_PROMPT)?;
    let mut new_tokens = vec![];
    let start_build = std::time::Instant::now();
    let device = if args.cpu {
        Device::Cpu
    } else if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::Cuda(0)
    };
    let mut vs = nn::VarStore::new(device);
    let llama = llama(vs.root(), args);
    // When loading the weights, assume that they are in the float16
    // format.
    vs.set_kind(Kind::Half);
    println!("generated the model in {:?}", start_build.elapsed());

    let start_load = std::time::Instant::now();
    // Instead of the standard `VarStore::load`, we use a mmap version that
    // should improve memory efficiency.
    {
        let file = std::fs::File::open("llama.safetensors")?;
        let content = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let safetensors = safetensors::SafeTensors::deserialize(&content)?;

        let mut variables = vs.variables_.lock().unwrap();
        for (name, var) in variables.named_variables.iter_mut() {
            let view = safetensors.tensor(name)?;
            let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
            let kind: Kind = view.dtype().try_into()?;
            // Using from_blob here instead of from_data_size avoids some unnecessary copy.
            let src_tensor =
                unsafe { Tensor::from_blob(view.data().as_ptr(), &size, &[], kind, Device::Cpu) };
            var.f_copy_(&src_tensor)?;
        }
    }
    println!("loaded weights in {:?}", start_load.elapsed());

    vs.set_kind(args.kind.to_kind());
    println!("switched to {:?}", args.kind);

    for index in 0..args.sample_len {
        let ctxt: Vec<_> =
            tokens[tokens.len().saturating_sub(CONTEXT_SIZE)..].iter().map(|c| *c as i64).collect();
        let ctxt = Tensor::from_slice(&ctxt).reshape([1, -1]);
        let logits = llama.forward(&ctxt);
        let sampled_y = logits.get(0).get(0).multinomial(1, true);
        let next_token = i64::try_from(&sampled_y)? as usize;
        tokens.push(next_token);
        new_tokens.push(next_token);
        println!("{} token: {} '{}'", index + 1, next_token, tokenizer.decode(&[next_token]));
    }
    println!("----\n{}\n----", tokenizer.decode(&new_tokens));
    Ok(())
}
