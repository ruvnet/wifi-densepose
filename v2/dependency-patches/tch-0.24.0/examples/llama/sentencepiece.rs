// A very naive sentencepiece encoder/decoder, this only supports the BPE model and not the unigram
// one.
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
pub struct Tokenizer {
    encoder: HashMap<Vec<u8>, usize>,
    // TODO: Maybe use a vec instead of a hashmap?
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
}

const DELIM: char = '▁';

impl Tokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let config: serde_json::Value = serde_json::from_reader(reader)?;
        let model = config.get("model").context("no model key")?;
        let type_ =
            model.get("type").context("no model.type key")?.as_str().context("not a string")?;
        if type_ != "BPE" {
            bail!(format!("model type is not BPE: {type_}"))
        }
        let vocab = model
            .get("vocab")
            .context("no model.vocab key")?
            .as_object()
            .context("model.vocab not an object")?;
        let single_chars: HashSet<u8> = vocab
            .iter()
            .filter_map(|(key, _)| {
                let b = key.as_bytes();
                if b.len() == 1 {
                    Some(b[0])
                } else {
                    None
                }
            })
            .collect();
        let encoder = vocab
            .iter()
            .rev()
            .map(|(key, value)| {
                let key = key
                    .strip_prefix("<0x")
                    .and_then(|s| s.strip_suffix('>'))
                    .and_then(|s| u8::from_str_radix(s, 16).ok())
                    .and_then(|s| if single_chars.contains(&s) { None } else { Some(s) })
                    .map_or_else(|| key.as_bytes().to_vec(), |s| vec![s]);
                value.as_i64().context("not an int").map(|v| (key, v as usize))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let bpe_ranks = model
            .get("merges")
            .context("no model.merges key")?
            .as_array()
            .context("model.merges not an array")?
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let value = value.as_str().context("not a string")?;
                match value.split_once(' ') {
                    Some((v1, v2)) => {
                        let key = (v1.as_bytes().to_vec(), v2.as_bytes().to_vec());
                        Ok((key, i))
                    }
                    None => bail!(format!("no space in merge '{value}'")),
                }
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let decoder = encoder
            .iter()
            .map(|(k, v)| (*v, String::from_utf8_lossy(k).replace(DELIM, " ")))
            .collect();
        Ok(Self { encoder, decoder, bpe_ranks })
    }

    fn get_pairs(word: &[Vec<u8>]) -> HashSet<(Vec<u8>, Vec<u8>)> {
        let mut pairs = HashSet::new();
        for (i, v) in word.iter().enumerate() {
            if i > 0 {
                pairs.insert((word[i - 1].clone(), v.clone()));
            }
        }
        pairs
    }

    fn bpe(&self, s: &str) -> Vec<usize> {
        let mut buffer = [0u8; 4];
        let mut word: Vec<Vec<u8>> = vec![];
        for c in s.chars() {
            let buffer = c.encode_utf8(&mut buffer);
            word.push(buffer.as_bytes().to_vec())
        }
        if word.is_empty() {
            return Vec::new();
        }
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
                    let mut first_and_second = first.clone();
                    first_and_second.extend_from_slice(second);
                    new_word.push(first_and_second);
                    index += 2
                } else {
                    new_word.push(w.clone());
                    index += 1
                }
            }
            word = new_word
        }
        word.iter().filter_map(|x| self.encoder.get(x)).copied().collect()
    }

    // Run bpe on the whole string, very very inefficient but should be good enough for
    // prompts. The original string should first be split on whitespace/...
    pub fn encode(&self, s: &str) -> Result<Vec<usize>> {
        let mut buffer = [0u8; 4];
        let s = format!("{DELIM}{}", s.replace(' ', DELIM.encode_utf8(&mut buffer)));
        Ok(self.bpe(&s))
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|token| self.decoder[token].as_str()).collect()
    }
}
