use std::io::Write;
use tch::{data, IndexOp, Tensor};

mod test_utils;
use test_utils::*;

#[test]
fn iter2() {
    let bsize: usize = 4;
    let vs: Vec<i64> = (0..1337).collect();
    let xs = Tensor::from_slice(&vs);
    let ys = Tensor::from_slice(&vs.iter().map(|x| x * 2).collect::<Vec<_>>());
    for (batch_xs, batch_ys) in data::Iter2::new(&xs, &ys, bsize as i64) {
        let xs = vec_i64_from(&batch_xs);
        let ys = vec_i64_from(&batch_ys);
        assert_eq!(xs.len(), bsize);
        assert_eq!(ys.len(), bsize);
        for i in 0..bsize {
            assert_eq!(ys[i], 2 * xs[i]);
            if i > 0 {
                assert_eq!(xs[i - 1] + 1, xs[i])
            }
        }
    }
    let mut all_in_order = true;
    for (batch_xs, batch_ys) in data::Iter2::new(&xs, &ys, bsize as i64).shuffle() {
        let xs = vec_i64_from(&batch_xs);
        let ys = vec_i64_from(&batch_ys);
        assert_eq!(xs.len(), bsize);
        assert_eq!(ys.len(), bsize);
        for i in 0..bsize {
            assert_eq!(ys[i], 2 * xs[i]);
            if i > 0 && xs[i - 1] + 1 != xs[i] {
                all_in_order = false
            }
        }
    }
    assert!(!all_in_order)
}

#[test]
fn text() {
    let filename = std::env::temp_dir().join(format!("tch-{}.txt", std::process::id()));
    {
        let mut file = std::fs::File::create(&filename).unwrap();
        file.write_all(b"01234567890123456789").unwrap();
    }
    let text_data = data::TextData::new(&filename).unwrap();
    for i in 0..10 {
        assert_eq!(text_data.label_to_char(i), i.to_string().chars().next().unwrap());
    }
    for xs in text_data.iter_shuffle(2, 5) {
        let first_column_plus_one = (xs.i((.., ..1)) + 1).fmod(10);
        let second_column = xs.i((.., 1..=1));
        let err: i64 = from(
            &(first_column_plus_one - second_column).pow_tensor_scalar(2).sum(tch::Kind::Float),
        );
        assert_eq!(err, 0)
    }
}
