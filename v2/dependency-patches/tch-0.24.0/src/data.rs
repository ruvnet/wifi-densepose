//! Dataset iterators.
use crate::{kind, kind::Kind, Device, IndexOp, TchError, Tensor};
use std::collections::HashMap;

/// An iterator over a pair of tensors which have the same first dimension
/// size.
/// The typical use case is to iterate over batches. Each batch is a pair
/// containing a (potentially random) slice of each of the two input
/// tensors.
#[derive(Debug)]
pub struct Iter2 {
    xs: Tensor,
    ys: Tensor,
    batch_index: i64,
    batch_size: i64,
    total_size: i64,
    device: Device,
    return_smaller_last_batch: bool,
}

impl Iter2 {
    /// Returns a new iterator.
    ///
    /// This takes as input two tensors which first dimension must match. The
    /// returned iterator can be used to range over mini-batches of data of
    /// specified size.
    /// An error is returned if `xs` and `ys` have different first dimension
    /// sizes.
    ///
    /// # Arguments
    ///
    /// * `xs` - the features to be used by the model.
    /// * `ys` - the targets that the model attempts to predict.
    /// * `batch_size` - the size of batches to be returned.
    pub fn f_new(xs: &Tensor, ys: &Tensor, batch_size: i64) -> Result<Iter2, TchError> {
        let total_size = xs.size()[0];
        if ys.size()[0] != total_size {
            return Err(TchError::Shape(format!(
                "different dimension for the two inputs {xs:?} {ys:?}"
            )));
        }
        Ok(Iter2 {
            xs: xs.shallow_clone(),
            ys: ys.shallow_clone(),
            batch_index: 0,
            batch_size,
            total_size,
            device: Device::Cpu,
            return_smaller_last_batch: false,
        })
    }

    /// Returns a new iterator.
    ///
    /// This takes as input two tensors which first dimension must match. The
    /// returned iterator can be used to range over mini-batches of data of
    /// specified size.
    /// Panics if `xs` and `ys` have different first dimension sizes.
    ///
    /// # Arguments
    ///
    /// * `xs` - the features to be used by the model.
    /// * `ys` - the targets that the model attempts to predict.
    /// * `batch_size` - the size of batches to be returned.
    pub fn new(xs: &Tensor, ys: &Tensor, batch_size: i64) -> Iter2 {
        Iter2::f_new(xs, ys, batch_size).unwrap()
    }

    /// Shuffles the dataset.
    ///
    /// The iterator would still run over the whole dataset but the order in
    /// which elements are grouped in mini-batches is randomized.
    pub fn shuffle(&mut self) -> &mut Iter2 {
        let index = Tensor::randperm(self.total_size, (Kind::Int64, self.device));
        self.xs = self.xs.index_select(0, &index);
        self.ys = self.ys.index_select(0, &index);
        self
    }

    /// Transfers the mini-batches to a specified device.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, device: Device) -> &mut Iter2 {
        self.device = device;
        self
    }

    /// When set, returns the last batch even if smaller than the batch size.
    pub fn return_smaller_last_batch(&mut self) -> &mut Iter2 {
        self.return_smaller_last_batch = true;
        self
    }
}

impl Iterator for Iter2 {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let size = std::cmp::min(self.batch_size, self.total_size - start);
        if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
            None
        } else {
            self.batch_index += 1;
            Some((
                self.xs.i(start..start + size).to_device(self.device),
                self.ys.i(start..start + size).to_device(self.device),
            ))
        }
    }
}

/// Text data holder.
#[derive(Debug)]
pub struct TextData {
    data: Tensor,
    char_for_label: Vec<char>,
    label_for_char: HashMap<u8, u8>,
}

/// Text data iterator.
#[derive(Debug)]
pub struct TextDataIter {
    data: Tensor,
    seq_len: i64,
    batch_index: i64,
    batch_size: i64,
    indexes: Tensor,
    indexes_len: i64,
}

impl TextData {
    /// Creates a text dataset from a file.
    pub fn new<P: AsRef<std::path::Path>>(filename: P) -> Result<TextData, TchError> {
        let mut buffer = std::fs::read(&filename).map_err(|err| {
            std::io::Error::new(err.kind(), format!("{:?} {err}", filename.as_ref()))
        })?;

        let mut label_for_char = HashMap::<u8, u8>::new();
        let mut char_for_label = Vec::<char>::new();
        for c in buffer.iter_mut() {
            *c = *label_for_char.entry(*c).or_insert_with(|| {
                let label = char_for_label.len() as u8;
                char_for_label.push(*c as char);
                label
            })
        }

        Ok(TextData { data: Tensor::from_slice(&buffer), char_for_label, label_for_char })
    }

    /// Returns the number of different characters/labels used by the dataset.
    pub fn labels(&self) -> i64 {
        self.char_for_label.len() as i64
    }

    /// Returns a shallow copy of the data.
    pub fn data(&self) -> Tensor {
        self.data.shallow_clone()
    }

    pub fn label_to_char(&self, label: i64) -> char {
        self.char_for_label[label as usize]
    }

    pub fn char_to_label(&self, c: char) -> Result<u8, TchError> {
        match self.label_for_char.get(&(c as u8)) {
            None => Err(TchError::Convert(format!("cannot find char {c}"))),
            Some(v) => Ok(*v),
        }
    }

    /// Returns a batch iterator over the dataset.
    /// Each sample is made of seq_len characters.
    pub fn iter_shuffle(&self, seq_len: i64, batch_size: i64) -> TextDataIter {
        let indexes_len = self.data.size()[0] - seq_len + 1;
        TextDataIter {
            data: self.data.shallow_clone(),
            seq_len,
            batch_index: 0,
            batch_size,
            indexes: Tensor::randperm(indexes_len, kind::INT64_CPU),
            indexes_len,
        }
    }
}

impl Iterator for TextDataIter {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let size = std::cmp::min(self.batch_size, self.indexes_len - start);
        if size < self.batch_size {
            None
        } else {
            self.batch_index += 1;
            let indexes = Vec::<i64>::try_from(&self.indexes.i(start..start + size)).unwrap();
            let batch: Vec<_> = indexes.iter().map(|&i| self.data.i(i..i + self.seq_len)).collect();
            let batch: Vec<_> = batch.iter().collect();
            Some(Tensor::stack(&batch, 0))
        }
    }
}
