// Stable Diffusion implementation inspired:
// - Huggingface's amazing diffuser Python api: https://huggingface.co/blog/annotated-diffusion
// - Huggingface's (also amazing) blog post: https://huggingface.co/blog/annotated-diffusion
// - The "Grokking Stable Diffusion" notebook by Jonathan Whitaker.
// https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing
//
// In order to run this, first download the following and extract the file in data/
//
// mkdir -p data && cd data
// wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
// gunzip bpe_simple_vocab_16e6.txt.gz
//
// Download and convert the weights:
//
// 1. Clip Encoding Weights
// wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
// From python, extract the weights and save them as a .npz file.
//   import numpy as np
//   import torch
//   model = torch.load("./pytorch_model.bin")
//   np.savez("./pytorch_model.npz", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
//
// Then use tensor_tools to convert this to a .ot file that tch can use.
//   cargo run --release --example tensor-tools cp ./data/pytorch_model.npz ./data/pytorch_model.ot
//
// 2. VAE and Unet Weights
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/diffusion_pytorch_model.bin
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/diffusion_pytorch_model.bin
//
//   import numpy as np
//   import torch
//   model = torch.load("./vae.bin")
//   np.savez("./vae.npz", **{k: v.numpy() for k, v in model.items()})
//   model = torch.load("./unet.bin")
//   np.savez("./unet.npz", **{k: v.numpy() for k, v in model.items()})
//
//   cargo run --release --example tensor-tools cp ./data/vae.npz ./data/vae.ot
//   cargo run --release --example tensor-tools cp ./data/unet.npz ./data/unet.ot
// TODO: fix tensor_tools so that it works properly there.
// TODO: Split this file, probably in a way similar to huggingface/diffusers.
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use tch::{kind, nn, nn::Module, Device, Kind, Tensor};

// The config details can be found in the "text_config" section of this json file:
// https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
//   "hidden_act": "quick_gelu"
const VOCAB_SIZE: i64 = 49408;
const EMBED_DIM: i64 = 768; // a.k.a. config.hidden_size
const INTERMEDIATE_SIZE: i64 = 3072;
const MAX_POSITION_EMBEDDINGS: usize = 77;
const NUM_HIDDEN_LAYERS: i64 = 12;
const NUM_ATTENTION_HEADS: i64 = 12;

const HEIGHT: i64 = 512;
const WIDTH: i64 = 512;
const GUIDANCE_SCALE: f64 = 7.5;

const BYTES_TO_UNICODE: [(u8, char); 256] = [
    (33, '!'),
    (34, '"'),
    (35, '#'),
    (36, '$'),
    (37, '%'),
    (38, '&'),
    (39, '\''),
    (40, '('),
    (41, ')'),
    (42, '*'),
    (43, '+'),
    (44, ','),
    (45, '-'),
    (46, '.'),
    (47, '/'),
    (48, '0'),
    (49, '1'),
    (50, '2'),
    (51, '3'),
    (52, '4'),
    (53, '5'),
    (54, '6'),
    (55, '7'),
    (56, '8'),
    (57, '9'),
    (58, ':'),
    (59, ';'),
    (60, '<'),
    (61, '='),
    (62, '>'),
    (63, '?'),
    (64, '@'),
    (65, 'A'),
    (66, 'B'),
    (67, 'C'),
    (68, 'D'),
    (69, 'E'),
    (70, 'F'),
    (71, 'G'),
    (72, 'H'),
    (73, 'I'),
    (74, 'J'),
    (75, 'K'),
    (76, 'L'),
    (77, 'M'),
    (78, 'N'),
    (79, 'O'),
    (80, 'P'),
    (81, 'Q'),
    (82, 'R'),
    (83, 'S'),
    (84, 'T'),
    (85, 'U'),
    (86, 'V'),
    (87, 'W'),
    (88, 'X'),
    (89, 'Y'),
    (90, 'Z'),
    (91, '['),
    (92, '\\'),
    (93, ']'),
    (94, '^'),
    (95, '_'),
    (96, '`'),
    (97, 'a'),
    (98, 'b'),
    (99, 'c'),
    (100, 'd'),
    (101, 'e'),
    (102, 'f'),
    (103, 'g'),
    (104, 'h'),
    (105, 'i'),
    (106, 'j'),
    (107, 'k'),
    (108, 'l'),
    (109, 'm'),
    (110, 'n'),
    (111, 'o'),
    (112, 'p'),
    (113, 'q'),
    (114, 'r'),
    (115, 's'),
    (116, 't'),
    (117, 'u'),
    (118, 'v'),
    (119, 'w'),
    (120, 'x'),
    (121, 'y'),
    (122, 'z'),
    (123, '{'),
    (124, '|'),
    (125, '}'),
    (126, '~'),
    (161, '¡'),
    (162, '¢'),
    (163, '£'),
    (164, '¤'),
    (165, '¥'),
    (166, '¦'),
    (167, '§'),
    (168, '¨'),
    (169, '©'),
    (170, 'ª'),
    (171, '«'),
    (172, '¬'),
    (174, '®'),
    (175, '¯'),
    (176, '°'),
    (177, '±'),
    (178, '²'),
    (179, '³'),
    (180, '´'),
    (181, 'µ'),
    (182, '¶'),
    (183, '·'),
    (184, '¸'),
    (185, '¹'),
    (186, 'º'),
    (187, '»'),
    (188, '¼'),
    (189, '½'),
    (190, '¾'),
    (191, '¿'),
    (192, 'À'),
    (193, 'Á'),
    (194, 'Â'),
    (195, 'Ã'),
    (196, 'Ä'),
    (197, 'Å'),
    (198, 'Æ'),
    (199, 'Ç'),
    (200, 'È'),
    (201, 'É'),
    (202, 'Ê'),
    (203, 'Ë'),
    (204, 'Ì'),
    (205, 'Í'),
    (206, 'Î'),
    (207, 'Ï'),
    (208, 'Ð'),
    (209, 'Ñ'),
    (210, 'Ò'),
    (211, 'Ó'),
    (212, 'Ô'),
    (213, 'Õ'),
    (214, 'Ö'),
    (215, '×'),
    (216, 'Ø'),
    (217, 'Ù'),
    (218, 'Ú'),
    (219, 'Û'),
    (220, 'Ü'),
    (221, 'Ý'),
    (222, 'Þ'),
    (223, 'ß'),
    (224, 'à'),
    (225, 'á'),
    (226, 'â'),
    (227, 'ã'),
    (228, 'ä'),
    (229, 'å'),
    (230, 'æ'),
    (231, 'ç'),
    (232, 'è'),
    (233, 'é'),
    (234, 'ê'),
    (235, 'ë'),
    (236, 'ì'),
    (237, 'í'),
    (238, 'î'),
    (239, 'ï'),
    (240, 'ð'),
    (241, 'ñ'),
    (242, 'ò'),
    (243, 'ó'),
    (244, 'ô'),
    (245, 'õ'),
    (246, 'ö'),
    (247, '÷'),
    (248, 'ø'),
    (249, 'ù'),
    (250, 'ú'),
    (251, 'û'),
    (252, 'ü'),
    (253, 'ý'),
    (254, 'þ'),
    (255, 'ÿ'),
    (0, 'Ā'),
    (1, 'ā'),
    (2, 'Ă'),
    (3, 'ă'),
    (4, 'Ą'),
    (5, 'ą'),
    (6, 'Ć'),
    (7, 'ć'),
    (8, 'Ĉ'),
    (9, 'ĉ'),
    (10, 'Ċ'),
    (11, 'ċ'),
    (12, 'Č'),
    (13, 'č'),
    (14, 'Ď'),
    (15, 'ď'),
    (16, 'Đ'),
    (17, 'đ'),
    (18, 'Ē'),
    (19, 'ē'),
    (20, 'Ĕ'),
    (21, 'ĕ'),
    (22, 'Ė'),
    (23, 'ė'),
    (24, 'Ę'),
    (25, 'ę'),
    (26, 'Ě'),
    (27, 'ě'),
    (28, 'Ĝ'),
    (29, 'ĝ'),
    (30, 'Ğ'),
    (31, 'ğ'),
    (32, 'Ġ'),
    (127, 'ġ'),
    (128, 'Ģ'),
    (129, 'ģ'),
    (130, 'Ĥ'),
    (131, 'ĥ'),
    (132, 'Ħ'),
    (133, 'ħ'),
    (134, 'Ĩ'),
    (135, 'ĩ'),
    (136, 'Ī'),
    (137, 'ī'),
    (138, 'Ĭ'),
    (139, 'ĭ'),
    (140, 'Į'),
    (141, 'į'),
    (142, 'İ'),
    (143, 'ı'),
    (144, 'Ĳ'),
    (145, 'ĳ'),
    (146, 'Ĵ'),
    (147, 'ĵ'),
    (148, 'Ķ'),
    (149, 'ķ'),
    (150, 'ĸ'),
    (151, 'Ĺ'),
    (152, 'ĺ'),
    (153, 'Ļ'),
    (154, 'ļ'),
    (155, 'Ľ'),
    (156, 'ľ'),
    (157, 'Ŀ'),
    (158, 'ŀ'),
    (159, 'Ł'),
    (160, 'ł'),
    (173, 'Ń'),
];

const PAT: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

// This is mostly a Rust rewrite of the original Python CLIP code.
// https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
struct Tokenizer {
    re: regex::Regex,
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    start_of_text_token: usize,
    end_of_text_token: usize,
}

impl Tokenizer {
    fn create<T: AsRef<std::path::Path>>(bpe_path: T) -> anyhow::Result<Tokenizer> {
        let bpe_file = std::fs::File::open(bpe_path)?;
        let bpe_lines: Result<Vec<String>, _> = std::io::BufReader::new(bpe_file).lines().collect();
        let bpe_lines = bpe_lines?;
        let bpe_lines: Result<Vec<_>, _> = bpe_lines[1..49152 - 256 - 2 + 1]
            .iter()
            .map(|line| {
                let vs: Vec<_> = line.split_whitespace().collect();
                if vs.len() != 2 {
                    anyhow::bail!("expected two items got {} '{}'", vs.len(), line)
                }
                Ok((vs[0].to_string(), vs[1].to_string()))
            })
            .collect();
        let bpe_lines = bpe_lines?;
        let mut vocab: Vec<String> = Vec::new();
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(elem.into())
        }
        for (_index, elem) in BYTES_TO_UNICODE {
            vocab.push(format!("{elem}</w>"));
        }
        for elem in bpe_lines.iter() {
            vocab.push(format!("{}{}", elem.0, elem.1))
        }
        let start_of_text_token = vocab.len();
        vocab.push("<|startoftext|>".to_string());
        let end_of_text_token = vocab.len();
        vocab.push("<|endoftext|>".to_string());
        let encoder: HashMap<_, _> = vocab.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let decoder: HashMap<_, _> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bpe_ranks: HashMap<_, _> =
            bpe_lines.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let re = regex::Regex::new(PAT)?;
        let tokenizer =
            Tokenizer { encoder, re, bpe_ranks, decoder, start_of_text_token, end_of_text_token };
        Ok(tokenizer)
    }

    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, token: &str) -> Vec<usize> {
        let mut word: Vec<String> = token.chars().map(|x| x.to_string()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        let last_index = word.len() - 1;
        word[last_index] = format!("{}</w>", word[last_index]);
        while word.len() > 1 {
            let mut current_min = None;
            let pairs = Self::get_pairs(&word);
            for p in pairs.iter() {
                match self.bpe_ranks.get(p) {
                    None => {}
                    Some(v) => {
                        let should_replace = match current_min {
                            None => true,
                            Some((current_min, _)) => v < current_min,
                        };
                        if should_replace {
                            current_min = Some((v, p))
                        }
                    }
                }
            }
            let (first, second) = match current_min {
                None => break,
                Some((_v, (first, second))) => (first, second),
            };
            let mut new_word = vec![];
            let mut index = 0;
            while index < word.len() {
                let w = &word[index];
                if index + 1 < word.len() && w == first && &word[index + 1] == second {
                    new_word.push(format!("{first}{second}"));
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter().map(|x| *self.encoder.get(x).unwrap()).collect()
    }

    fn encode(&self, s: &str, pad_size_to: Option<usize>) -> anyhow::Result<Vec<usize>> {
        let s = s.to_lowercase();
        let mut bpe_tokens: Vec<usize> = vec![self.start_of_text_token];
        for token in self.re.captures_iter(&s) {
            let token = token.get(0).unwrap().as_str();
            bpe_tokens.extend(self.bpe(token))
        }
        match pad_size_to {
            None => bpe_tokens.push(self.end_of_text_token),
            Some(pad_size_to) => {
                bpe_tokens.resize_with(
                    std::cmp::min(bpe_tokens.len(), pad_size_to - 1),
                    Default::default,
                );
                while bpe_tokens.len() < pad_size_to {
                    bpe_tokens.push(self.end_of_text_token)
                }
            }
        }
        Ok(bpe_tokens)
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let s: String = tokens.iter().map(|token| self.decoder[token].as_str()).collect();
        s.replace("</w>", " ")
    }
}

// CLIP Text Model
// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py
#[derive(Debug)]
struct ClipTextEmbeddings {
    token_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    position_ids: Tensor,
}

impl ClipTextEmbeddings {
    fn new(vs: nn::Path) -> Self {
        let token_embedding =
            nn::embedding(&vs / "token_embedding", VOCAB_SIZE, EMBED_DIM, Default::default());
        let position_embedding = nn::embedding(
            &vs / "position_embedding",
            MAX_POSITION_EMBEDDINGS as i64,
            EMBED_DIM,
            Default::default(),
        );
        let position_ids =
            Tensor::arange(MAX_POSITION_EMBEDDINGS as i64, (Kind::Int64, vs.device()))
                .expand([1, -1], false);
        ClipTextEmbeddings { token_embedding, position_embedding, position_ids }
    }
}

impl Module for ClipTextEmbeddings {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let token_embedding = self.token_embedding.forward(xs);
        let position_embedding = self.position_embedding.forward(&self.position_ids);
        token_embedding + position_embedding
    }
}

fn quick_gelu(xs: &Tensor) -> Tensor {
    xs * (xs * 1.702).sigmoid()
}

#[derive(Debug)]
struct ClipAttention {
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    head_dim: i64,
    scale: f64,
}

impl ClipAttention {
    fn new(vs: nn::Path) -> Self {
        let k_proj = nn::linear(&vs / "k_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let v_proj = nn::linear(&vs / "v_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let q_proj = nn::linear(&vs / "q_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let out_proj = nn::linear(&vs / "out_proj", EMBED_DIM, EMBED_DIM, Default::default());
        let head_dim = EMBED_DIM / NUM_ATTENTION_HEADS;
        let scale = (head_dim as f64).powf(-0.5);
        ClipAttention { k_proj, v_proj, q_proj, out_proj, head_dim, scale }
    }

    fn shape(&self, xs: &Tensor, seq_len: i64, bsz: i64) -> Tensor {
        xs.view((bsz, seq_len, NUM_ATTENTION_HEADS, self.head_dim)).transpose(1, 2).contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let (bsz, tgt_len, embed_dim) = xs.size3().unwrap();
        let query_states = xs.apply(&self.q_proj) * self.scale;
        let proj_shape = (bsz * NUM_ATTENTION_HEADS, -1, self.head_dim);
        let query_states = self.shape(&query_states, tgt_len, bsz).view(proj_shape);
        let key_states = self.shape(&xs.apply(&self.k_proj), -1, bsz).view(proj_shape);
        let value_states = self.shape(&xs.apply(&self.v_proj), -1, bsz).view(proj_shape);
        let attn_weights = query_states.bmm(&key_states.transpose(1, 2));

        let src_len = key_states.size()[1];
        let attn_weights =
            attn_weights.view((bsz, NUM_ATTENTION_HEADS, tgt_len, src_len)) + causal_attention_mask;
        let attn_weights = attn_weights.view((bsz * NUM_ATTENTION_HEADS, tgt_len, src_len));
        let attn_weights = attn_weights.softmax(-1, Kind::Float);

        let attn_output = attn_weights.bmm(&value_states);
        attn_output
            .view((bsz, NUM_ATTENTION_HEADS, tgt_len, self.head_dim))
            .transpose(1, 2)
            .reshape([bsz, tgt_len, embed_dim])
            .apply(&self.out_proj)
    }
}

#[derive(Debug)]
struct ClipMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ClipMlp {
    fn new(vs: nn::Path) -> Self {
        let fc1 = nn::linear(&vs / "fc1", EMBED_DIM, INTERMEDIATE_SIZE, Default::default());
        let fc2 = nn::linear(&vs / "fc2", INTERMEDIATE_SIZE, EMBED_DIM, Default::default());
        ClipMlp { fc1, fc2 }
    }
}

impl Module for ClipMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs.apply(&self.fc1);
        quick_gelu(&xs).apply(&self.fc2)
    }
}

#[derive(Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: nn::Path) -> Self {
        let self_attn = ClipAttention::new(&vs / "self_attn");
        let layer_norm1 = nn::layer_norm(&vs / "layer_norm1", vec![EMBED_DIM], Default::default());
        let mlp = ClipMlp::new(&vs / "mlp");
        let layer_norm2 = nn::layer_norm(&vs / "layer_norm2", vec![EMBED_DIM], Default::default());
        ClipEncoderLayer { self_attn, layer_norm1, mlp, layer_norm2 }
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs);
        let xs = self.self_attn.forward(&xs, causal_attention_mask);
        let xs = xs + residual;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs);
        let xs = self.mlp.forward(&xs);
        xs + residual
    }
}

#[derive(Debug)]
struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    fn new(vs: nn::Path) -> Self {
        let vs = &vs / "layers";
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..NUM_HIDDEN_LAYERS {
            let layer = ClipEncoderLayer::new(&vs / index);
            layers.push(layer)
        }
        ClipEncoder { layers }
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)
        }
        xs
    }
}

#[derive(Debug)]
struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: nn::LayerNorm,
}

impl ClipTextTransformer {
    fn new(vs: nn::Path) -> Self {
        let vs = &vs / "text_model";
        let embeddings = ClipTextEmbeddings::new(&vs / "embeddings");
        let encoder = ClipEncoder::new(&vs / "encoder");
        let final_layer_norm =
            nn::layer_norm(&vs / "final_layer_norm", vec![EMBED_DIM], Default::default());
        ClipTextTransformer { embeddings, encoder, final_layer_norm }
    }

    // https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
    fn build_causal_attention_mask(bsz: i64, seq_len: i64, device: Device) -> Tensor {
        let mut mask = Tensor::ones([bsz, seq_len, seq_len], (Kind::Float, device));
        mask.fill_(f32::MIN as f64).triu_(1).unsqueeze(1)
    }
}

impl Module for ClipTextTransformer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (bsz, seq_len) = xs.size2().unwrap();
        let xs = self.embeddings.forward(xs);
        let causal_attention_mask = Self::build_causal_attention_mask(bsz, seq_len, xs.device());
        let xs = self.encoder.forward(&xs, &causal_attention_mask);
        xs.apply(&self.final_layer_norm)
    }
}

#[derive(Debug)]
struct GeGlu {
    proj: nn::Linear,
}

impl GeGlu {
    fn new(vs: nn::Path, dim_in: i64, dim_out: i64) -> Self {
        let proj = nn::linear(&vs / "proj", dim_in, dim_out * 2, Default::default());
        Self { proj }
    }
}

impl Module for GeGlu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let hidden_states_and_gate = xs.apply(&self.proj).chunk(2, -1);
        &hidden_states_and_gate[0] * hidden_states_and_gate[1].gelu("none")
    }
}

#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear: nn::Linear,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    fn new(vs: nn::Path, dim: i64, dim_out: Option<i64>, mult: i64) -> Self {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = &vs / "net";
        let project_in = GeGlu::new(&vs / 0, dim, inner_dim);
        let linear = nn::linear(&vs / 2, inner_dim, dim_out, Default::default());
        Self { project_in, linear }
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.project_in).apply(&self.linear)
    }
}

#[derive(Debug)]
struct CrossAttention {
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    heads: i64,
    scale: f64,
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    fn new(
        vs: nn::Path,
        query_dim: i64,
        context_dim: Option<i64>,
        heads: i64,
        dim_head: i64,
    ) -> Self {
        let no_bias = nn::LinearConfig { bias: false, ..Default::default() };
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = nn::linear(&vs / "to_q", query_dim, inner_dim, no_bias);
        let to_k = nn::linear(&vs / "to_k", context_dim, inner_dim, no_bias);
        let to_v = nn::linear(&vs / "to_v", context_dim, inner_dim, no_bias);
        let to_out = nn::linear(&vs / "to_out" / 0, inner_dim, query_dim, Default::default());
        Self { to_q, to_k, to_v, to_out, heads, scale }
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Tensor {
        let (batch_size, seq_len, dim) = xs.size3().unwrap();
        xs.reshape([batch_size, seq_len, self.heads, dim / self.heads])
            .permute([0, 2, 1, 3])
            .reshape([batch_size * self.heads, seq_len, dim / self.heads])
    }

    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Tensor {
        let (batch_size, seq_len, dim) = xs.size3().unwrap();
        xs.reshape([batch_size / self.heads, self.heads, seq_len, dim])
            .permute([0, 2, 1, 3])
            .reshape([batch_size / self.heads, seq_len, dim * self.heads])
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        let xs = query
            .matmul(&(key.transpose(-1, -2) * self.scale))
            .softmax(-1, Kind::Float)
            .matmul(value);
        self.reshape_batch_dim_to_heads(&xs)
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Tensor {
        let query = xs.apply(&self.to_q);
        let context = context.unwrap_or(xs);
        let key = context.apply(&self.to_k);
        let value = context.apply(&self.to_v);
        let query = self.reshape_heads_to_batch_dim(&query);
        let key = self.reshape_heads_to_batch_dim(&key);
        let value = self.reshape_heads_to_batch_dim(&value);
        self.attention(&query, &key, &value).apply(&self.to_out)
    }
}

#[derive(Debug)]
struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
}

impl BasicTransformerBlock {
    fn new(vs: nn::Path, dim: i64, n_heads: i64, d_head: i64, context_dim: Option<i64>) -> Self {
        let attn1 = CrossAttention::new(&vs / "attn1", dim, None, n_heads, d_head);
        let ff = FeedForward::new(&vs / "ff", dim, None, 4);
        let attn2 = CrossAttention::new(&vs / "attn2", dim, context_dim, n_heads, d_head);
        let norm1 = nn::layer_norm(&vs / "norm1", vec![dim], Default::default());
        let norm2 = nn::layer_norm(&vs / "norm2", vec![dim], Default::default());
        let norm3 = nn::layer_norm(&vs / "norm3", vec![dim], Default::default());
        Self { attn1, ff, attn2, norm1, norm2, norm3 }
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Tensor {
        let xs = self.attn1.forward(&xs.apply(&self.norm1), None) + xs;
        let xs = self.attn2.forward(&xs.apply(&self.norm2), context) + xs;
        xs.apply(&self.norm3).apply(&self.ff) + xs
    }
}

#[derive(Debug, Clone, Copy)]
struct SpatialTransformerConfig {
    depth: i64,
    num_groups: i64,
    context_dim: Option<i64>,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self { depth: 1, num_groups: 32, context_dim: None }
    }
}

#[derive(Debug)]
struct SpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: nn::Conv2D,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: nn::Conv2D,
    #[allow(dead_code)]
    config: SpatialTransformerConfig,
}

impl SpatialTransformer {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        n_heads: i64,
        d_head: i64,
        config: SpatialTransformerConfig,
    ) -> Self {
        let inner_dim = n_heads * d_head;
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, affine: true, ..Default::default() };
        let norm = nn::group_norm(&vs / "norm", config.num_groups, in_channels, group_cfg);
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 0, ..Default::default() };
        let proj_in = nn::conv2d(&vs / "proj_in", in_channels, inner_dim, 1, conv_cfg);
        let mut transformer_blocks = vec![];
        let vs_tb = &vs / "transformer_blocks";
        for index in 0..config.depth {
            let tb = BasicTransformerBlock::new(
                &vs_tb / index,
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
            );
            transformer_blocks.push(tb)
        }
        let proj_out = nn::conv2d(&vs / "proj_out", inner_dim, in_channels, 1, conv_cfg);
        Self { norm, proj_in, transformer_blocks, proj_out, config }
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Tensor {
        let (batch, _channel, height, weight) = xs.size4().unwrap();
        let residual = xs;
        let xs = xs.apply(&self.norm).apply(&self.proj_in);
        let inner_dim = xs.size()[1];
        let mut xs = xs.permute([0, 2, 3, 1]).view((batch, height * weight, inner_dim));
        for block in self.transformer_blocks.iter() {
            xs = block.forward(&xs, context)
        }
        let xs =
            xs.view((batch, height, weight, inner_dim)).permute([0, 3, 1, 2]).apply(&self.proj_out);
        xs + residual
    }
}

#[derive(Debug, Clone, Copy)]
struct AttentionBlockConfig {
    num_head_channels: Option<i64>,
    num_groups: i64,
    rescale_output_factor: f64,
    eps: f64,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self { num_head_channels: None, num_groups: 32, rescale_output_factor: 1., eps: 1e-5 }
    }
}

#[derive(Debug)]
struct AttentionBlock {
    group_norm: nn::GroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    proj_attn: nn::Linear,
    channels: i64,
    num_heads: i64,
    config: AttentionBlockConfig,
}

impl AttentionBlock {
    fn new(vs: nn::Path, channels: i64, config: AttentionBlockConfig) -> Self {
        let num_head_channels = config.num_head_channels.unwrap_or(channels);
        let num_heads = channels / num_head_channels;
        let group_cfg = nn::GroupNormConfig { eps: config.eps, affine: true, ..Default::default() };
        let group_norm = nn::group_norm(&vs / "group_norm", config.num_groups, channels, group_cfg);
        let query = nn::linear(&vs / "query", channels, channels, Default::default());
        let key = nn::linear(&vs / "key", channels, channels, Default::default());
        let value = nn::linear(&vs / "value", channels, channels, Default::default());
        let proj_attn = nn::linear(&vs / "proj_attn", channels, channels, Default::default());
        Self { group_norm, query, key, value, proj_attn, channels, num_heads, config }
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Tensor {
        let (batch, t, _h_times_d) = xs.size3().unwrap();
        xs.view((batch, t, self.num_heads, -1)).permute([0, 2, 1, 3])
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = xs;
        let (batch, channel, height, width) = xs.size4().unwrap();
        let xs = xs.apply(&self.group_norm).view((batch, channel, height * width)).transpose(1, 2);

        let query_proj = xs.apply(&self.query);
        let key_proj = xs.apply(&self.key);
        let value_proj = xs.apply(&self.value);

        let query_states = self.transpose_for_scores(query_proj);
        let key_states = self.transpose_for_scores(key_proj);
        let value_states = self.transpose_for_scores(value_proj);

        let scale = f64::powf((self.channels as f64) / (self.num_heads as f64), -0.25);
        let attention_scores =
            (query_states * scale).matmul(&(key_states.transpose(-1, -2) * scale));
        let attention_probs = attention_scores.softmax(-1, Kind::Float);

        let xs = attention_probs.matmul(&value_states);
        let xs = xs.permute([0, 2, 1, 3]).contiguous();
        let mut new_xs_shape = xs.size();
        new_xs_shape.pop();
        new_xs_shape.pop();
        new_xs_shape.push(self.channels);

        let xs = xs
            .view(new_xs_shape.as_slice())
            .apply(&self.proj_attn)
            .transpose(-1, -2)
            .view((batch, channel, height, width));
        (xs + residual) / self.config.rescale_output_factor
    }
}

#[derive(Debug)]
struct Downsample2D {
    conv: Option<nn::Conv2D>,
    padding: i64,
}

impl Downsample2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        use_conv: bool,
        out_channels: i64,
        padding: i64,
    ) -> Self {
        let conv = if use_conv {
            let config = nn::ConvConfig { stride: 2, padding, ..Default::default() };
            let conv = nn::conv2d(&vs / "conv", in_channels, out_channels, 3, config);
            Some(conv)
        } else {
            None
        };
        Downsample2D { conv, padding }
    }
}

impl Module for Downsample2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match &self.conv {
            None => xs.avg_pool2d([2, 2], [2, 2], [0, 0], false, true, None),
            Some(conv) => {
                if self.padding == 0 {
                    xs.pad([0, 1, 0, 1], "constant", Some(0.)).apply(conv)
                } else {
                    xs.apply(conv)
                }
            }
        }
    }
}

// This does not support the conv-transpose mode.
#[derive(Debug)]
struct Upsample2D {
    conv: nn::Conv2D,
}

impl Upsample2D {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let config = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv = nn::conv2d(&vs / "conv", in_channels, out_channels, 3, config);
        Self { conv }
    }
}

impl Upsample2D {
    fn forward(&self, xs: &Tensor, size: Option<(i64, i64)>) -> Tensor {
        let xs = match size {
            None => {
                // The following does not work and it's tricky to pass no fixed
                // dimensions so hack our way around this.
                // xs.upsample_nearest2d(&[], Some(2.), Some(2.)
                let (_bsize, _channels, h, w) = xs.size4().unwrap();
                xs.upsample_nearest2d([2 * h, 2 * w], Some(2.), Some(2.))
            }
            Some((h, w)) => xs.upsample_nearest2d([h, w], None, None),
        };
        xs.apply(&self.conv)
    }
}

#[derive(Debug, Clone, Copy)]
struct ResnetBlock2DConfig {
    out_channels: Option<i64>,
    temb_channels: Option<i64>,
    groups: i64,
    groups_out: Option<i64>,
    eps: f64,
    use_in_shortcut: Option<bool>,
    // non_linearity: silu
    output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            temb_channels: Some(512),
            groups: 32,
            groups_out: None,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2D,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2D,
    time_emb_proj: Option<nn::Linear>,
    conv_shortcut: Option<nn::Conv2D>,
    config: ResnetBlock2DConfig,
}

impl ResnetBlock2D {
    fn new(vs: nn::Path, in_channels: i64, config: ResnetBlock2DConfig) -> Self {
        let out_channels = config.out_channels.unwrap_or(in_channels);
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let group_cfg = nn::GroupNormConfig { eps: config.eps, affine: true, ..Default::default() };
        let norm1 = nn::group_norm(&vs / "norm1", config.groups, in_channels, group_cfg);
        let conv1 = nn::conv2d(&vs / "conv1", in_channels, out_channels, 3, conv_cfg);
        let groups_out = config.groups_out.unwrap_or(config.groups);
        let norm2 = nn::group_norm(&vs / "norm2", groups_out, out_channels, group_cfg);
        let conv2 = nn::conv2d(&vs / "conv2", out_channels, out_channels, 3, conv_cfg);
        let use_in_shortcut = config.use_in_shortcut.unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = nn::ConvConfig { stride: 1, padding: 0, ..Default::default() };
            Some(nn::conv2d(&vs / "conv_shortcut", in_channels, out_channels, 1, conv_cfg))
        } else {
            None
        };
        let time_emb_proj = config.temb_channels.map(|temb_channels| {
            nn::linear(&vs / "time_emb_proj", temb_channels, out_channels, Default::default())
        });
        Self { norm1, conv1, norm2, conv2, time_emb_proj, config, conv_shortcut }
    }

    fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Tensor {
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => xs.apply(conv_shortcut),
            None => xs.shallow_clone(),
        };
        let xs = xs.apply(&self.norm1).silu().apply(&self.conv1);
        let xs = match (temb, &self.time_emb_proj) {
            (Some(temb), Some(time_emb_proj)) => {
                temb.silu().apply(time_emb_proj).unsqueeze(-1).unsqueeze(-1) + xs
            }
            _ => xs,
        };
        let xs = xs.apply(&self.norm2).silu().apply(&self.conv2);
        (shortcut_xs + xs) / self.config.output_scale_factor
    }
}

#[derive(Debug, Clone, Copy)]
struct DownEncoderBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: i64,
    output_scale_factor: f64,
    add_downsample: bool,
    downsample_padding: i64,
}

impl Default for DownEncoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    #[allow(dead_code)]
    config: DownEncoderBlock2DConfig,
}

impl DownEncoderBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: DownEncoderBlock2DConfig,
    ) -> Self {
        let resnets: Vec<_> = {
            let vs = &vs / "resnets";
            let conv_cfg = ResnetBlock2DConfig {
                eps: config.resnet_eps,
                out_channels: Some(out_channels),
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                temb_channels: None,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(&vs / i, in_channels, conv_cfg)
                })
                .collect()
        };
        let downsampler = if config.add_downsample {
            let downsample = Downsample2D::new(
                &(&vs / "downsamplers") / 0,
                out_channels,
                true,
                out_channels,
                config.downsample_padding,
            );
            Some(downsample)
        } else {
            None
        };
        Self { resnets, downsampler, config }
    }
}

impl Module for DownEncoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, None)
        }
        match &self.downsampler {
            Some(downsampler) => xs.apply(downsampler),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UpDecoderBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: i64,
    output_scale_factor: f64,
    add_upsample: bool,
}

impl Default for UpDecoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
struct UpDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    #[allow(dead_code)]
    config: UpDecoderBlock2DConfig,
}

impl UpDecoderBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: UpDecoderBlock2DConfig,
    ) -> Self {
        let resnets: Vec<_> = {
            let vs = &vs / "resnets";
            let conv_cfg = ResnetBlock2DConfig {
                out_channels: Some(out_channels),
                eps: config.resnet_eps,
                groups: config.resnet_groups,
                output_scale_factor: config.output_scale_factor,
                temb_channels: None,
                ..Default::default()
            };
            (0..(config.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 { in_channels } else { out_channels };
                    ResnetBlock2D::new(&vs / i, in_channels, conv_cfg)
                })
                .collect()
        };
        let upsampler = if config.add_upsample {
            let upsample = Upsample2D::new(&vs / "upsamplers" / 0, out_channels, out_channels);
            Some(upsample)
        } else {
            None
        };
        Self { resnets, upsampler, config }
    }
}

impl Module for UpDecoderBlock2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, None)
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&xs, None),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct UNetMidBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: Option<i64>,
    attn_num_head_channels: Option<i64>,
    // attention_type "default"
    output_scale_factor: f64,
}

impl Default for UNetMidBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: Some(1),
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
struct UNetMidBlock2D {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(AttentionBlock, ResnetBlock2D)>,
    #[allow(dead_code)]
    config: UNetMidBlock2DConfig,
}

impl UNetMidBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        temb_channels: Option<i64>,
        config: UNetMidBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or_else(|| i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let attn_cfg = AttentionBlockConfig {
            num_head_channels: config.attn_num_head_channels,
            num_groups: resnet_groups,
            rescale_output_factor: config.output_scale_factor,
            eps: config.resnet_eps,
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = AttentionBlock::new(&vs_attns / index, in_channels, attn_cfg);
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }

    fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Tensor {
        let mut xs = self.resnet.forward(xs, temb);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&xs.apply(attn), temb)
        }
        xs
    }
}

#[derive(Debug, Clone, Copy)]
struct UNetMidBlock2DCrossAttnConfig {
    num_layers: i64,
    resnet_eps: f64,
    resnet_groups: Option<i64>,
    attn_num_head_channels: i64,
    // attention_type "default"
    output_scale_factor: f64,
    cross_attn_dim: i64,
}

impl Default for UNetMidBlock2DCrossAttnConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: Some(32),
            attn_num_head_channels: 1,
            output_scale_factor: 1.,
            cross_attn_dim: 1280,
        }
    }
}

#[derive(Debug)]
struct UNetMidBlock2DCrossAttn {
    resnet: ResnetBlock2D,
    attn_resnets: Vec<(SpatialTransformer, ResnetBlock2D)>,
    #[allow(dead_code)]
    config: UNetMidBlock2DCrossAttnConfig,
}

impl UNetMidBlock2DCrossAttn {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        temb_channels: Option<i64>,
        config: UNetMidBlock2DCrossAttnConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let vs_attns = &vs / "attentions";
        let resnet_groups = config.resnet_groups.unwrap_or_else(|| i64::min(in_channels / 4, 32));
        let resnet_cfg = ResnetBlock2DConfig {
            eps: config.resnet_eps,
            groups: resnet_groups,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnet = ResnetBlock2D::new(&vs_resnets / "0", in_channels, resnet_cfg);
        let n_heads = config.attn_num_head_channels;
        let attn_cfg = SpatialTransformerConfig {
            depth: 1,
            num_groups: resnet_groups,
            context_dim: Some(config.cross_attn_dim),
        };
        let mut attn_resnets = vec![];
        for index in 0..config.num_layers {
            let attn = SpatialTransformer::new(
                &vs_attns / index,
                in_channels,
                n_heads,
                in_channels / n_heads,
                attn_cfg,
            );
            let resnet = ResnetBlock2D::new(&vs_resnets / (index + 1), in_channels, resnet_cfg);
            attn_resnets.push((attn, resnet))
        }
        Self { resnet, attn_resnets, config }
    }

    fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Tensor {
        let mut xs = self.resnet.forward(xs, temb);
        for (attn, resnet) in self.attn_resnets.iter() {
            xs = resnet.forward(&attn.forward(&xs, encoder_hidden_states), temb)
        }
        xs
    }
}

#[derive(Debug, Clone)]
struct EncoderConfig {
    // down_block_types: DownEncoderBlock2D
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    norm_num_groups: i64,
    double_z: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            double_z: true,
        }
    }
}

#[derive(Debug)]
struct Encoder {
    conv_in: nn::Conv2D,
    down_blocks: Vec<DownEncoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    #[allow(dead_code)]
    config: EncoderConfig,
}

impl Encoder {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: EncoderConfig) -> Self {
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, config.block_out_channels[0], 3, conv_cfg);
        let mut down_blocks = vec![];
        let vs_down_blocks = &vs / "down_blocks";
        for index in 0..config.block_out_channels.len() {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index > 0 {
                config.block_out_channels[index - 1]
            } else {
                config.block_out_channels[0]
            };
            let is_final = index + 1 == config.block_out_channels.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            let down_block =
                DownEncoderBlock2D::new(&vs_down_blocks / index, in_channels, out_channels, cfg);
            down_blocks.push(down_block)
        }
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, None, mid_cfg);
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            last_block_out_channels,
            group_cfg,
        );
        let conv_out_channels = if config.double_z { 2 * out_channels } else { out_channels };
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", last_block_out_channels, conv_out_channels, 3, conv_cfg);
        Self { conv_in, down_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.conv_in);
        for down_block in self.down_blocks.iter() {
            xs = xs.apply(down_block)
        }
        self.mid_block.forward(&xs, None).apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct DecoderConfig {
    // up_block_types: UpDecoderBlock2D
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    norm_num_groups: i64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self { block_out_channels: vec![64], layers_per_block: 2, norm_num_groups: 32 }
    }
}

#[derive(Debug)]
struct Decoder {
    conv_in: nn::Conv2D,
    up_blocks: Vec<UpDecoderBlock2D>,
    mid_block: UNetMidBlock2D,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    #[allow(dead_code)]
    config: DecoderConfig,
}

impl Decoder {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: DecoderConfig) -> Self {
        let n_block_out_channels = config.block_out_channels.len();
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in =
            nn::conv2d(&vs / "conv_in", in_channels, last_block_out_channels, 3, conv_cfg);
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block =
            UNetMidBlock2D::new(&vs / "mid_block", last_block_out_channels, None, mid_cfg);
        let mut up_blocks = vec![];
        let vs_up_blocks = &vs / "up_blocks";
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().copied().rev().collect();
        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;
            let cfg = UpDecoderBlock2DConfig {
                num_layers: config.layers_per_block + 1,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_upsample: !is_final,
                ..Default::default()
            };
            let up_block =
                UpDecoderBlock2D::new(&vs_up_blocks / index, in_channels, out_channels, cfg);
            up_blocks.push(up_block)
        }
        let group_cfg = nn::GroupNormConfig { eps: 1e-6, ..Default::default() };
        let conv_norm_out = nn::group_norm(
            &vs / "conv_norm_out",
            config.norm_num_groups,
            config.block_out_channels[0],
            group_cfg,
        );
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let conv_out =
            nn::conv2d(&vs / "conv_out", config.block_out_channels[0], out_channels, 3, conv_cfg);
        Self { conv_in, up_blocks, mid_block, conv_norm_out, conv_out, config }
    }
}

impl Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.mid_block.forward(&xs.apply(&self.conv_in), None);
        for up_block in self.up_blocks.iter() {
            xs = xs.apply(up_block)
        }
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

#[derive(Debug, Clone)]
struct AutoEncoderKLConfig {
    block_out_channels: Vec<i64>,
    layers_per_block: i64,
    latent_channels: i64,
    norm_num_groups: i64,
}

impl Default for AutoEncoderKLConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 32,
        }
    }
}

// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/vae.py#L485
// This implementation is specific to the config used in stable-diffusion-v1-4
// https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
#[derive(Debug)]
struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: nn::Conv2D,
    post_quant_conv: nn::Conv2D,
    #[allow(dead_code)]
    config: AutoEncoderKLConfig,
}

impl AutoEncoderKL {
    fn new(vs: nn::Path, in_channels: i64, out_channels: i64, config: AutoEncoderKLConfig) -> Self {
        let latent_channels = config.latent_channels;
        let encoder_cfg = EncoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            double_z: true,
        };
        let encoder = Encoder::new(&vs / "encoder", in_channels, latent_channels, encoder_cfg);
        let decoder_cfg = DecoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
        };
        let decoder = Decoder::new(&vs / "decoder", latent_channels, out_channels, decoder_cfg);
        let conv_cfg = Default::default();
        let quant_conv =
            nn::conv2d(&vs / "quant_conv", 2 * latent_channels, 2 * latent_channels, 1, conv_cfg);
        let post_quant_conv =
            nn::conv2d(&vs / "post_quant_conv", latent_channels, latent_channels, 1, conv_cfg);
        Self { encoder, decoder, quant_conv, post_quant_conv, config }
    }

    // Returns the parameters of the latent distribution.
    #[allow(dead_code)]
    fn encode(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.encoder).apply(&self.quant_conv)
    }

    /// Takes as input some sampled values.
    fn decode(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.post_quant_conv).apply(&self.decoder)
    }
}

#[derive(Debug, Clone, Copy)]
struct DownBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    resnet_groups: i64,
    output_scale_factor: f64,
    add_downsample: bool,
    downsample_padding: i64,
}

impl Default for DownBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
struct DownBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    #[allow(dead_code)]
    config: DownBlock2DConfig,
}

impl DownBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: DownBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(out_channels),
            eps: config.resnet_eps,
            output_scale_factor: config.output_scale_factor,
            temb_channels,
            ..Default::default()
        };
        let resnets = (0..config.num_layers)
            .map(|i| {
                let in_channels = if i == 0 { in_channels } else { out_channels };
                ResnetBlock2D::new(&vs_resnets / i, in_channels, resnet_cfg)
            })
            .collect();
        let downsampler = if config.add_downsample {
            let downsampler = Downsample2D::new(
                &vs / "downsamplers" / 0,
                out_channels,
                true,
                out_channels,
                config.downsample_padding,
            );
            Some(downsampler)
        } else {
            None
        };
        Self { resnets, downsampler, config }
    }

    fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> (Tensor, Vec<Tensor>) {
        let mut xs = xs.shallow_clone();
        let mut output_states = vec![];
        for resnet in self.resnets.iter() {
            xs = resnet.forward(&xs, temb);
            output_states.push(xs.shallow_clone());
        }
        let xs = match &self.downsampler {
            Some(downsampler) => {
                let xs = xs.apply(downsampler);
                output_states.push(xs.shallow_clone());
                xs
            }
            None => xs,
        };
        (xs, output_states)
    }
}

#[derive(Debug, Clone, Copy)]
struct CrossAttnDownBlock2DConfig {
    downblock: DownBlock2DConfig,
    attn_num_head_channels: i64,
    cross_attention_dim: i64,
    // attention_type: "default"
}

impl Default for CrossAttnDownBlock2DConfig {
    fn default() -> Self {
        Self { downblock: Default::default(), attn_num_head_channels: 1, cross_attention_dim: 1280 }
    }
}

#[derive(Debug)]
struct CrossAttnDownBlock2D {
    downblock: DownBlock2D,
    attentions: Vec<SpatialTransformer>,
    #[allow(dead_code)]
    config: CrossAttnDownBlock2DConfig,
}

impl CrossAttnDownBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: CrossAttnDownBlock2DConfig,
    ) -> Self {
        let downblock = DownBlock2D::new(
            vs.clone(),
            in_channels,
            out_channels,
            temb_channels,
            config.downblock,
        );
        let n_heads = config.attn_num_head_channels;
        let cfg = SpatialTransformerConfig {
            depth: 1,
            context_dim: Some(config.cross_attention_dim),
            num_groups: config.downblock.resnet_groups,
        };
        let vs_attn = &vs / "attentions";
        let attentions = (0..config.downblock.num_layers)
            .map(|i| {
                SpatialTransformer::new(
                    &vs_attn / i,
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    cfg,
                )
            })
            .collect();
        Self { downblock, attentions, config }
    }

    fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> (Tensor, Vec<Tensor>) {
        let mut output_states = vec![];
        let mut xs = xs.shallow_clone();
        for (resnet, attn) in self.downblock.resnets.iter().zip(self.attentions.iter()) {
            xs = resnet.forward(&xs, temb);
            xs = attn.forward(&xs, encoder_hidden_states);
            output_states.push(xs.shallow_clone());
        }
        let xs = match &self.downblock.downsampler {
            Some(downsampler) => {
                let xs = xs.apply(downsampler);
                output_states.push(xs.shallow_clone());
                xs
            }
            None => xs,
        };
        (xs, output_states)
    }
}

#[derive(Debug, Clone, Copy)]
struct UpBlock2DConfig {
    num_layers: i64,
    resnet_eps: f64,
    // resnet_time_scale_shift: "default"
    // resnet_act_fn: "swish"
    resnet_groups: i64,
    output_scale_factor: f64,
    add_upsample: bool,
}

impl Default for UpBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_upsample: true,
        }
    }
}

#[derive(Debug)]
struct UpBlock2D {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Upsample2D>,
    #[allow(dead_code)]
    config: UpBlock2DConfig,
}

impl UpBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        prev_output_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: UpBlock2DConfig,
    ) -> Self {
        let vs_resnets = &vs / "resnets";
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(out_channels),
            temb_channels,
            eps: config.resnet_eps,
            output_scale_factor: config.output_scale_factor,
            ..Default::default()
        };
        let resnets = (0..config.num_layers)
            .map(|i| {
                let res_skip_channels =
                    if i == config.num_layers - 1 { in_channels } else { out_channels };
                let resnet_in_channels = if i == 0 { prev_output_channels } else { out_channels };
                let in_channels = resnet_in_channels + res_skip_channels;
                ResnetBlock2D::new(&vs_resnets / i, in_channels, resnet_cfg)
            })
            .collect();
        let upsampler = if config.add_upsample {
            let upsampler = Upsample2D::new(&vs / "upsamplers" / 0, out_channels, out_channels);
            Some(upsampler)
        } else {
            None
        };
        Self { resnets, upsampler, config }
    }

    fn forward(
        &self,
        xs: &Tensor,
        res_xs: &[Tensor],
        temb: Option<&Tensor>,
        upsample_size: Option<(i64, i64)>,
    ) -> Tensor {
        let mut xs = xs.shallow_clone();
        for (index, resnet) in self.resnets.iter().enumerate() {
            xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1);
            xs = resnet.forward(&xs, temb);
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(&xs, upsample_size),
            None => xs,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CrossAttnUpBlock2DConfig {
    upblock: UpBlock2DConfig,
    attn_num_head_channels: i64,
    cross_attention_dim: i64,
    // attention_type: "default"
}

impl Default for CrossAttnUpBlock2DConfig {
    fn default() -> Self {
        Self { upblock: Default::default(), attn_num_head_channels: 1, cross_attention_dim: 1280 }
    }
}

#[derive(Debug)]
struct CrossAttnUpBlock2D {
    upblock: UpBlock2D,
    attentions: Vec<SpatialTransformer>,
    #[allow(dead_code)]
    config: CrossAttnUpBlock2DConfig,
}

impl CrossAttnUpBlock2D {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        prev_output_channels: i64,
        out_channels: i64,
        temb_channels: Option<i64>,
        config: CrossAttnUpBlock2DConfig,
    ) -> Self {
        let upblock = UpBlock2D::new(
            vs.clone(),
            in_channels,
            prev_output_channels,
            out_channels,
            temb_channels,
            config.upblock,
        );
        let n_heads = config.attn_num_head_channels;
        let cfg = SpatialTransformerConfig {
            depth: 1,
            context_dim: Some(config.cross_attention_dim),
            num_groups: config.upblock.resnet_groups,
        };
        let vs_attn = &vs / "attentions";
        let attentions = (0..config.upblock.num_layers)
            .map(|i| {
                SpatialTransformer::new(
                    &vs_attn / i,
                    out_channels,
                    n_heads,
                    out_channels / n_heads,
                    cfg,
                )
            })
            .collect();
        Self { upblock, attentions, config }
    }

    fn forward(
        &self,
        xs: &Tensor,
        res_xs: &[Tensor],
        temb: Option<&Tensor>,
        upsample_size: Option<(i64, i64)>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Tensor {
        let mut xs = xs.shallow_clone();
        for (index, resnet) in self.upblock.resnets.iter().enumerate() {
            xs = Tensor::cat(&[&xs, &res_xs[res_xs.len() - index - 1]], 1);
            xs = resnet.forward(&xs, temb);
            xs = self.attentions[index].forward(&xs, encoder_hidden_states);
        }
        match &self.upblock.upsampler {
            Some(upsampler) => upsampler.forward(&xs, upsample_size),
            None => xs,
        }
    }
}

#[derive(Debug)]
struct Timesteps {
    num_channels: i64,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    device: Device,
}

impl Timesteps {
    fn new(
        num_channels: i64,
        flip_sin_to_cos: bool,
        downscale_freq_shift: f64,
        device: Device,
    ) -> Self {
        Self { num_channels, flip_sin_to_cos, downscale_freq_shift, device }
    }
}

impl Module for Timesteps {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let half_dim = self.num_channels / 2;
        let exponent = Tensor::arange(half_dim, (Kind::Float, self.device)) * -f64::ln(10000.);
        let exponent = exponent / (half_dim as f64 - self.downscale_freq_shift);
        let emb = exponent.exp();
        // emb = timesteps[:, None].float() * emb[None, :]
        let emb = xs.unsqueeze(-1) * emb.unsqueeze(0);
        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[emb.cos(), emb.sin()], -1)
        } else {
            Tensor::cat(&[emb.sin(), emb.cos()], -1)
        };
        if self.num_channels % 2 == 1 {
            emb.pad([0, 1, 0, 0], "constant", None)
        } else {
            emb
        }
    }
}

#[derive(Debug)]
struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    // act_fn: "silu"
    fn new(vs: nn::Path, channel: i64, time_embed_dim: i64) -> Self {
        let linear_cfg = Default::default();
        let linear_1 = nn::linear(&vs / "linear_1", channel, time_embed_dim, linear_cfg);
        let linear_2 = nn::linear(&vs / "linear_2", time_embed_dim, time_embed_dim, linear_cfg);
        Self { linear_1, linear_2 }
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.linear_1).silu().apply(&self.linear_2)
    }
}

#[derive(Debug, Clone, Copy)]
struct BlockConfig {
    out_channels: i64,
    use_cross_attn: bool,
}

#[derive(Debug, Clone)]
struct UNet2DConditionModelConfig {
    center_input_sample: bool,
    flip_sin_to_cos: bool,
    freq_shift: f64,
    blocks: Vec<BlockConfig>,
    layers_per_block: i64,
    downsample_padding: i64,
    mid_block_scale_factor: f64,
    norm_num_groups: i64,
    norm_eps: f64,
    cross_attention_dim: i64,
    attention_head_dim: i64,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.,
            blocks: vec![
                BlockConfig { out_channels: 320, use_cross_attn: true },
                BlockConfig { out_channels: 640, use_cross_attn: true },
                BlockConfig { out_channels: 1280, use_cross_attn: true },
                BlockConfig { out_channels: 1280, use_cross_attn: false },
            ],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 1280,
            attention_head_dim: 8,
        }
    }
}

#[derive(Debug)]
enum UNetDownBlock {
    Basic(DownBlock2D),
    CrossAttn(CrossAttnDownBlock2D),
}

#[derive(Debug)]
enum UNetUpBlock {
    Basic(UpBlock2D),
    CrossAttn(CrossAttnUpBlock2D),
}

#[derive(Debug)]
struct UNet2DConditionModel {
    conv_in: nn::Conv2D,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetDownBlock>,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: Vec<UNetUpBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2D,
    config: UNet2DConditionModelConfig,
}

impl UNet2DConditionModel {
    fn new(
        vs: nn::Path,
        in_channels: i64,
        out_channels: i64,
        config: UNet2DConditionModelConfig,
    ) -> Self {
        let n_blocks = config.blocks.len();
        let b_channels = config.blocks[0].out_channels;
        let bl_channels = config.blocks.last().unwrap().out_channels;
        let time_embed_dim = b_channels * 4;
        let conv_cfg = nn::ConvConfig { stride: 1, padding: 1, ..Default::default() };
        let conv_in = nn::conv2d(&vs / "conv_in", in_channels, b_channels, 3, conv_cfg);

        let time_proj =
            Timesteps::new(b_channels, config.flip_sin_to_cos, config.freq_shift, vs.device());
        let time_embedding =
            TimestepEmbedding::new(&vs / "time_embedding", b_channels, time_embed_dim);

        let vs_db = &vs / "down_blocks";
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn } = config.blocks[i];
                let in_channels =
                    if i > 0 { config.blocks[i - 1].out_channels } else { b_channels };
                let db_cfg = DownBlock2DConfig {
                    num_layers: config.layers_per_block,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_downsample: i < n_blocks - 1,
                    downsample_padding: config.downsample_padding,
                    ..Default::default()
                };
                if use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig {
                        downblock: db_cfg,
                        attn_num_head_channels: config.attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                    };
                    let block = CrossAttnDownBlock2D::new(
                        &vs_db / i,
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        config,
                    );
                    UNetDownBlock::CrossAttn(block)
                } else {
                    let block = DownBlock2D::new(
                        &vs_db / i,
                        in_channels,
                        out_channels,
                        Some(time_embed_dim),
                        db_cfg,
                    );
                    UNetDownBlock::Basic(block)
                }
            })
            .collect();

        let mid_cfg = UNetMidBlock2DCrossAttnConfig {
            resnet_eps: config.norm_eps,
            output_scale_factor: config.mid_block_scale_factor,
            cross_attn_dim: config.cross_attention_dim,
            attn_num_head_channels: config.attention_head_dim,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block = UNetMidBlock2DCrossAttn::new(
            &vs / "mid_block",
            bl_channels,
            Some(time_embed_dim),
            mid_cfg,
        );

        let vs_ub = &vs / "up_blocks";
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let BlockConfig { out_channels, use_cross_attn } = config.blocks[n_blocks - 1 - i];
                let prev_out_channels =
                    if i > 0 { config.blocks[n_blocks - i].out_channels } else { bl_channels };
                let in_channels = {
                    let index = if i == n_blocks - 1 { 0 } else { n_blocks - i - 2 };
                    config.blocks[index].out_channels
                };
                let ub_cfg = UpBlock2DConfig {
                    num_layers: config.layers_per_block + 1,
                    resnet_eps: config.norm_eps,
                    resnet_groups: config.norm_num_groups,
                    add_upsample: i < n_blocks - 1,
                    ..Default::default()
                };
                if use_cross_attn {
                    let config = CrossAttnUpBlock2DConfig {
                        upblock: ub_cfg,
                        attn_num_head_channels: config.attention_head_dim,
                        cross_attention_dim: config.cross_attention_dim,
                    };
                    let block = CrossAttnUpBlock2D::new(
                        &vs_ub / i,
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        config,
                    );
                    UNetUpBlock::CrossAttn(block)
                } else {
                    let block = UpBlock2D::new(
                        &vs_ub / i,
                        in_channels,
                        prev_out_channels,
                        out_channels,
                        Some(time_embed_dim),
                        ub_cfg,
                    );
                    UNetUpBlock::Basic(block)
                }
            })
            .collect();

        let group_cfg = nn::GroupNormConfig { eps: config.norm_eps, ..Default::default() };
        let conv_norm_out =
            nn::group_norm(&vs / "conv_norm_out", config.norm_num_groups, b_channels, group_cfg);
        let conv_out = nn::conv2d(&vs / "conv_out", b_channels, out_channels, 3, conv_cfg);
        Self {
            conv_in,
            time_proj,
            time_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            config,
        }
    }
}

impl UNet2DConditionModel {
    fn forward(&self, xs: &Tensor, timestep: f64, encoder_hidden_states: &Tensor) -> Tensor {
        let (bsize, _channels, height, width) = xs.size4().unwrap();
        let device = xs.device();
        let n_blocks = self.config.blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2i64.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;
        // 0. center input if necessary
        let xs = if self.config.center_input_sample { xs * 2.0 - 1.0 } else { xs.shallow_clone() };
        // 1. time
        let emb = (Tensor::ones([bsize], (Kind::Float, device)) * timestep)
            .apply(&self.time_proj)
            .apply(&self.time_embedding);
        // 2. pre-process
        let xs = xs.apply(&self.conv_in);
        // 3. down
        let mut down_block_res_xs = vec![xs.shallow_clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(&xs, Some(&emb)),
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(&xs, Some(&emb), Some(encoder_hidden_states))
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = _xs;
        }
        // 4. mid
        let xs = self.mid_block.forward(&xs, Some(&emb), Some(encoder_hidden_states));
        // 5. up
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs = down_block_res_xs.split_off(down_block_res_xs.len() - n_resnets);
            if i < n_blocks - 1 && forward_upsample_size {
                let (_, _, h, w) = down_block_res_xs.last().unwrap().size4().unwrap();
                upsample_size = Some((h, w))
            }
            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(&xs, &res_xs, Some(&emb), upsample_size),
                UNetUpBlock::CrossAttn(b) => {
                    b.forward(&xs, &res_xs, Some(&emb), upsample_size, Some(encoder_hidden_states))
                }
            };
        }
        // 6. post-process
        xs.apply(&self.conv_norm_out).silu().apply(&self.conv_out)
    }
}

// TODO: LMSDiscreteScheduler
// https://github.com/huggingface/diffusers/blob/32bf4fdc4386809c870528cb261028baae012d27/src/diffusers/schedulers/scheduling_lms_discrete.py#L47

fn build_clip_transformer(device: Device) -> anyhow::Result<ClipTextTransformer> {
    let mut vs = nn::VarStore::new(device);
    let text_model = ClipTextTransformer::new(vs.root());
    vs.load("data/pytorch_model.ot")?;
    Ok(text_model)
}

fn build_vae(device: Device) -> anyhow::Result<AutoEncoderKL> {
    let mut vs_ae = nn::VarStore::new(device);
    // https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/vae/config.json
    let autoencoder_cfg = AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 4,
        norm_num_groups: 32,
    };
    let autoencoder = AutoEncoderKL::new(vs_ae.root(), 3, 3, autoencoder_cfg);
    vs_ae.load("data/vae.ot")?;
    Ok(autoencoder)
}

fn build_unet(device: Device) -> anyhow::Result<UNet2DConditionModel> {
    let mut vs_unet = nn::VarStore::new(device);
    // https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json
    let unet_cfg = UNet2DConditionModelConfig {
        attention_head_dim: 8,
        blocks: vec![
            BlockConfig { out_channels: 320, use_cross_attn: true },
            BlockConfig { out_channels: 640, use_cross_attn: true },
            BlockConfig { out_channels: 1280, use_cross_attn: true },
            BlockConfig { out_channels: 1280, use_cross_attn: false },
        ],
        center_input_sample: false,
        cross_attention_dim: 768,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
    };
    let unet = UNet2DConditionModel::new(vs_unet.root(), 4, 4, unet_cfg);
    vs_unet.load("data/unet.ot")?;
    Ok(unet)
}

#[derive(Debug, Clone, Copy)]
enum BetaSchedule {
    #[allow(dead_code)]
    Linear,
    ScaledLinear,
}

#[derive(Debug, Clone, Copy)]
struct DDIMSchedulerConfig {
    beta_start: f64,
    beta_end: f64,
    beta_schedule: BetaSchedule,
    eta: f64,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085f64,
            beta_end: 0.012f64,
            beta_schedule: BetaSchedule::ScaledLinear,
            eta: 0.,
        }
    }
}

#[derive(Debug, Clone)]
struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
    #[allow(dead_code)]
    config: DDIMSchedulerConfig,
}

// clip_sample: False, set_alpha_to_one: False
impl DDIMScheduler {
    fn new(inference_steps: usize, train_timesteps: usize, config: DDIMSchedulerConfig) -> Self {
        let step_ratio = train_timesteps / inference_steps;
        // TODO: Remove this hack which aimed at matching the behavior of diffusers==0.2.4
        let timesteps = (0..(inference_steps + 1)).map(|s| s * step_ratio).rev().collect();
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => Tensor::linspace(
                config.beta_start.sqrt(),
                config.beta_end.sqrt(),
                train_timesteps as i64,
                kind::FLOAT_CPU,
            )
            .square(),
            BetaSchedule::Linear => Tensor::linspace(
                config.beta_start,
                config.beta_end,
                train_timesteps as i64,
                kind::FLOAT_CPU,
            ),
        };
        let alphas: Tensor = 1.0 - betas;
        let alphas_cumprod = Vec::<f64>::try_from(alphas.cumprod(0, Kind::Double)).unwrap();
        Self { alphas_cumprod, timesteps, step_ratio, config }
    }

    // https://github.com/huggingface/diffusers/blob/6e099e2c8ce4c4f5c7318e970a8c093dc5c7046e/src/diffusers/schedulers/scheduling_ddim.py#L195
    fn step(&self, model_output: &Tensor, timestep: usize, sample: &Tensor) -> Tensor {
        let prev_timestep = timestep.saturating_sub(self.step_ratio);

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = self.alphas_cumprod[prev_timestep];
        let beta_prod_t = 1. - alpha_prod_t;
        let beta_prod_t_prev = 1. - alpha_prod_t_prev;

        let pred_original_sample =
            (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt();

        let variance = (beta_prod_t_prev / beta_prod_t) * (1. - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.sqrt();

        let pred_sample_direction =
            (1. - alpha_prod_t_prev - std_dev_t * std_dev_t).sqrt() * model_output;
        let prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction;
        if self.config.eta > 0. {
            &prev_sample + Tensor::randn_like(&prev_sample) * std_dev_t
        } else {
            prev_sample
        }
    }
}

fn main() -> anyhow::Result<()> {
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    // TODO: Switch to using claps to allow more flags?
    let mut prompt = "A rusty robot holding a fire torch in its hand".to_string();
    let mut device = Device::cuda_if_available();
    for arg in std::env::args().skip(1) {
        if arg.as_str() == "cpu" {
            device = Device::Cpu;
        } else {
            prompt = arg;
        }
    }
    let n_steps = 30;
    let scheduler = DDIMScheduler::new(n_steps, 1000, Default::default());

    let tokenizer = Tokenizer::create("data/bpe_simple_vocab_16e6.txt")?;
    let tokens = tokenizer.encode(&prompt, Some(MAX_POSITION_EMBEDDINGS))?;
    let str = tokenizer.decode(&tokens);
    println!("Str: {str}");
    let tokens: Vec<i64> = tokens.iter().map(|x| *x as i64).collect();
    let tokens = Tensor::from_slice(&tokens).view((1, -1)).to(device);
    let uncond_tokens = tokenizer.encode("", Some(MAX_POSITION_EMBEDDINGS))?;
    let uncond_tokens: Vec<i64> = uncond_tokens.iter().map(|x| *x as i64).collect();
    let uncond_tokens = Tensor::from_slice(&uncond_tokens).view((1, -1)).to(device);
    println!("Tokens: {tokens:?}");

    let no_grad_guard = tch::no_grad_guard();
    println!("Building the Clip transformer.");
    let text_model = build_clip_transformer(device)?;
    let text_embeddings = text_model.forward(&tokens);
    let uncond_embeddings = text_model.forward(&uncond_tokens);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0);
    println!("Text embeddings: {text_embeddings:?}");

    println!("Building the autoencoder.");
    let vae = build_vae(device)?;
    println!("Building the unet.");
    let unet = build_unet(device)?;

    let bsize = 1;
    // DETERMINISTIC SEEDING
    tch::manual_seed(32);
    let mut latents = Tensor::randn([bsize, 4, HEIGHT / 8, WIDTH / 8], (Kind::Float, device));

    for (timestep_index, &timestep) in scheduler.timesteps.iter().enumerate() {
        println!("Timestep {timestep_index} {timestep} {latents:?}");
        let latent_model_input = Tensor::cat(&[&latents, &latents], 0);
        let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
        let noise_pred = noise_pred.chunk(2, 0);
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        let noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
        latents = scheduler.step(&noise_pred, timestep, &latents);

        let image = vae.decode(&(&latents / 0.18215));
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        let image = (image * 255.).to_kind(Kind::Uint8);
        tch::vision::image::save(&image, format!("sd_{timestep_index}.png"))?
    }

    drop(no_grad_guard);
    Ok(())
}
