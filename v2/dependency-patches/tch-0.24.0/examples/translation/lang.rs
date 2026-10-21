use std::collections::HashMap;

const SOS_TOKEN: &str = "SOS";
const EOS_TOKEN: &str = "EOS";

#[derive(Debug)]
pub struct Lang {
    name: String,
    word_to_index_and_count: HashMap<String, (usize, usize)>,
    index_to_word: HashMap<usize, String>,
}

impl Lang {
    fn add_word(&mut self, word: &str) {
        if !word.is_empty() {
            match self.word_to_index_and_count.get_mut(word) {
                None => {
                    let length = self.word_to_index_and_count.len();
                    self.word_to_index_and_count.insert(word.to_string(), (length, 1));
                    self.index_to_word.insert(length, word.to_owned());
                }
                Some((_, cnt)) => {
                    *cnt += 1;
                }
            }
        }
    }

    pub fn new(name: &str) -> Lang {
        let mut lang = Lang {
            name: name.to_string(),
            word_to_index_and_count: HashMap::new(),
            index_to_word: HashMap::new(),
        };
        lang.add_word(SOS_TOKEN);
        lang.add_word(EOS_TOKEN);
        lang
    }

    pub fn add_sentence(&mut self, sentence: &str) {
        for word in sentence.split_whitespace() {
            self.add_word(word);
        }
    }

    pub fn len(&self) -> usize {
        self.index_to_word.len()
    }

    pub fn sos_token(&self) -> usize {
        self.word_to_index_and_count.get(SOS_TOKEN).unwrap().0
    }

    pub fn eos_token(&self) -> usize {
        self.word_to_index_and_count.get(EOS_TOKEN).unwrap().0
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_index(&self, word: &str) -> Option<usize> {
        self.word_to_index_and_count.get(word).map(|x| x.0)
    }

    pub fn seq_to_string(&self, seq: &[usize]) -> String {
        let mut res: Vec<String> = vec![];
        for s in seq.iter() {
            res.push(self.index_to_word.get(s).unwrap().to_string());
        }
        res.join(" ")
    }
}
