use tch::{kind, Tensor};

#[test]
fn display_scalar() {
    let t = Tensor::ones([], kind::INT64_CPU) * (-1234);
    let s = format!("{t}");
    assert_eq!(&s, "[-1234]\nTensor[[], Int64]");
    let s = format!("{}", &t / 10.0);
    assert_eq!(&s, "[-123.4000]\nTensor[[], Float]");
    let s = format!("{}", &t / 1e8);
    assert_eq!(&s, "[-1.2340e-5]\nTensor[[], Float]");
    let s = format!("{}", &t * 1e8);
    assert_eq!(&s, "[-1.2340e11]\nTensor[[], Float]");
    let s = format!("{}", &t * 0.);
    assert_eq!(&s, "[-0.]\nTensor[[], Float]");
    let s = format!("{}", &t * (-0.));
    assert_eq!(&s, "[0.]\nTensor[[], Float]");
    let s = format!("{}", &t.eq_tensor(&t));
    assert_eq!(&s, "[true]\nTensor[[], Bool]");
    let s = format!("{}", &t.not_equal_tensor(&t));
    assert_eq!(&s, "[false]\nTensor[[], Bool]");
}

#[test]
fn display_vector() {
    let t = Tensor::from_slice::<i64>(&[]);
    let s = format!("{t}");
    assert_eq!(&s, "[]\nTensor[[0], Int64]");
    let t = Tensor::from_slice(&[0.1234567, 1.0, -1.2, 4.1, f64::NAN]);
    let s = format!("{t}");
    assert_eq!(&s, "[ 0.1235,  1.0000, -1.2000,  4.1000,     NaN]\nTensor[[5], Double]");
    let t = Tensor::ones([50], kind::FLOAT_CPU) * 42;
    let s = format!("\n{t}");
    let expected = r#"
[42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,
 42., 42.]
Tensor[[50], Float]"#;
    assert_eq!(&s, expected);
    let t = Tensor::ones([11000], kind::FLOAT_CPU) * 42;
    let s = format!("{t}");
    assert_eq!(&s, "[42., 42., 42., ..., 42., 42., 42.]\nTensor[[11000], Float]");
}

#[test]
fn display_multi_dim() {
    let t = Tensor::ones([200, 100], kind::FLOAT_CPU) * 42;
    let s = format!("\n{t}");
    let expected = r#"
[[42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 ...
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.],
 [42., 42., 42., ..., 42., 42., 42.]]
Tensor[[200, 100], Float]"#;
    assert_eq!(&s, expected);
    let t = t.reshape([2, 1, 1, 100, 100]);
    let t = format!("\n{t}");
    let expected = r#"
[[[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]],
 [[[[42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    ...
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.],
    [42., 42., 42., ..., 42., 42., 42.]]]]]
Tensor[[2, 1, 1, 100, 100], Float]"#;
    assert_eq!(&t, expected);
}
