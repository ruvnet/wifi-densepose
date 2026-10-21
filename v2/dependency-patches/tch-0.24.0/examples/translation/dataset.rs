use super::lang;
use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};

const PREFIXES: [&str; 12] = [
    "i am ",
    "i m ",
    "you are ",
    "you re",
    "he is ",
    "he s ",
    "she is ",
    "she s ",
    "we are ",
    "we re ",
    "they are ",
    "they re ",
];

#[derive(Debug)]
pub struct Dataset {
    input_lang: lang::Lang,
    output_lang: lang::Lang,
    pairs: Vec<(String, String)>,
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| match c {
            '!' => " !".to_string(),
            '.' => " .".to_string(),
            '?' => " ?".to_string(),
            _ => {
                if c.is_alphanumeric() {
                    c.to_string()
                } else {
                    " ".to_string()
                }
            }
        })
        .collect()
}

fn to_indexes(s: &str, lang: &lang::Lang) -> Vec<usize> {
    let mut res = s.split_whitespace().filter_map(|x| lang.get_index(x)).collect::<Vec<_>>();
    res.push(lang.eos_token());
    res
}

fn filter_prefix(s: &str) -> bool {
    PREFIXES.iter().any(|prefix| s.starts_with(prefix))
}

fn read_pairs(ilang: &str, olang: &str, max_length: usize) -> Result<Vec<(String, String)>> {
    let file = File::open(format!("data/{ilang}-{olang}.txt"))?;
    let mut res: Vec<(String, String)> = vec![];
    for line in BufReader::new(file).lines() {
        let line = line?;
        match line.split('\t').collect::<Vec<_>>().as_slice() {
            [lhs, rhs] => {
                let lhs = normalize(lhs);
                let rhs = normalize(rhs);
                if lhs.split_whitespace().count() < max_length
                    && rhs.split_whitespace().count() < max_length
                    && (filter_prefix(&lhs) || filter_prefix(&rhs))
                {
                    res.push((lhs, rhs))
                }
            }
            _ => bail!("a line does not contain a single tab {}", line),
        }
    }
    Ok(res)
}

impl Dataset {
    pub fn new(ilang: &str, olang: &str, max_length: usize) -> Result<Dataset> {
        let pairs = read_pairs(ilang, olang, max_length)?;
        let mut input_lang = lang::Lang::new(ilang);
        let mut output_lang = lang::Lang::new(olang);
        for (lhs, rhs) in pairs.iter() {
            input_lang.add_sentence(lhs);
            output_lang.add_sentence(rhs);
        }
        let dataset = Dataset { input_lang, output_lang, pairs };
        Ok(dataset)
    }

    pub fn input_lang(&self) -> &lang::Lang {
        &self.input_lang
    }

    pub fn output_lang(&self) -> &lang::Lang {
        &self.output_lang
    }

    pub fn reverse(self) -> Self {
        Dataset {
            input_lang: self.output_lang,
            output_lang: self.input_lang,
            pairs: self.pairs.into_iter().map(|(x, y)| (y, x)).collect(),
        }
    }

    pub fn pairs(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        self.pairs
            .iter()
            .map(|(lhs, rhs)| {
                (to_indexes(lhs, &self.input_lang), to_indexes(rhs, &self.output_lang))
            })
            .collect::<Vec<_>>()
    }
}
