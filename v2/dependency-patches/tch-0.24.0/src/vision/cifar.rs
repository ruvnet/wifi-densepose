//! The CIFAR-10 dataset.
//!
//! The files can be downloaded from the following page:
//! <https://www.cs.toronto.edu/~kriz/cifar.html>
//! The binary version of the dataset is used.
use super::dataset::Dataset;
use crate::{kind, IndexOp, Kind, Tensor};
use std::fs::File;
use std::io::{BufReader, Read, Result};

const W: i64 = 32;
const H: i64 = 32;
const C: i64 = 3;
const BYTES_PER_IMAGE: i64 = W * H * C + 1;
const SAMPLES_PER_FILE: i64 = 10000;

fn read_file_(filename: &std::path::Path) -> Result<(Tensor, Tensor)> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    let mut data = vec![0u8; (SAMPLES_PER_FILE * BYTES_PER_IMAGE) as usize];
    buf_reader.read_exact(&mut data)?;
    let content = Tensor::from_slice(&data);
    let images = Tensor::zeros([SAMPLES_PER_FILE, C, H, W], kind::FLOAT_CPU);
    let labels = Tensor::zeros([SAMPLES_PER_FILE], kind::INT64_CPU);
    for index in 0..SAMPLES_PER_FILE {
        let content_offset = BYTES_PER_IMAGE * index;
        labels.i(index).copy_(&content.i(content_offset));
        images.i(index).copy_(
            &content
                .narrow(0, 1 + content_offset, BYTES_PER_IMAGE - 1)
                .view((C, H, W))
                .to_kind(Kind::Float),
        );
    }
    Ok((images.to_kind(Kind::Float) / 255.0, labels))
}

fn read_file(filename: &std::path::Path) -> Result<(Tensor, Tensor)> {
    read_file_(filename)
        .map_err(|err| std::io::Error::new(err.kind(), format!("{filename:?} {err}")))
}

pub fn load_dir<T: AsRef<std::path::Path>>(dir: T) -> Result<Dataset> {
    let dir = dir.as_ref();
    let (test_images, test_labels) = read_file(&dir.join("test_batch.bin"))?;
    let train_images_and_labels = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ]
    .iter()
    .map(|x| read_file(&dir.join(x)))
    .collect::<Result<Vec<_>>>()?;
    let (train_images, train_labels): (Vec<_>, Vec<_>) =
        train_images_and_labels.into_iter().unzip();
    Ok(Dataset {
        train_images: Tensor::cat(&train_images, 0),
        train_labels: Tensor::cat(&train_labels, 0),
        test_images,
        test_labels,
        labels: 10,
    })
}
