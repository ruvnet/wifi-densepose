#![allow(clippy::unnecessary_fallible_conversions)]
use anyhow::Result;
use half::f16;
use std::convert::{TryFrom, TryInto};
use std::f32;
use tch::{Device, TchError, Tensor};

mod test_utils;
use test_utils::*;

#[test]
#[cfg(feature = "cuda-tests")]
fn amp_non_finite_check_and_unscale() {
    let mut u = Tensor::from_slice(&[10f32, 20f32]).to_device(Device::Cuda(0));
    let mut found_inf = Tensor::from_slice(&[0f32]).to_device(Device::Cuda(0));
    let inv_scale = Tensor::from_slice(&[0.1f32]).to_device(Device::Cuda(0));
    u.internal_amp_non_finite_check_and_unscale(&mut found_inf, &inv_scale);
    assert_eq!(vec_f32_from(&u), &[1f32, 2f32]);
    assert_eq!(vec_f32_from(&found_inf), [0f32]);

    let mut v = Tensor::from_slice(&[1f32, f32::INFINITY]).to_device(Device::Cuda(0));
    v.internal_amp_non_finite_check_and_unscale(&mut found_inf, &inv_scale);
    assert_eq!(vec_f32_from(&v), &[0.1, f32::INFINITY]);
    assert_eq!(vec_f32_from(&found_inf), [1f32]);

    u.internal_amp_non_finite_check_and_unscale(&mut found_inf, &inv_scale);
    assert_eq!(vec_f32_from(&u), &[0.1, 0.2]);
    // found_inf is sticky
    assert_eq!(vec_f32_from(&found_inf), [1f32]);
}

#[test]
fn assign_ops() {
    let mut t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    t += 1;
    t *= 2;
    t -= 1;
    assert_eq!(vec_i64_from(&t), [7, 3, 9, 3, 11]);
}

#[test]
fn constant_ops() {
    let mut t = Tensor::from_slice(&[7i64, 3, 9, 3, 11]);
    t = -t;
    assert_eq!(vec_i64_from(&t), [-7, -3, -9, -3, -11]);
    t = 1 - t;
    assert_eq!(vec_i64_from(&t), [8, 4, 10, 4, 12]);
    t = 2 * t;
    assert_eq!(vec_i64_from(&t), [16, 8, 20, 8, 24]);

    let mut t = Tensor::from_slice(&[0.2f64, 0.1]);
    t = 2 / t;
    assert_eq!(vec_f64_from(&t), [10.0, 20.0]);
}

#[test]
fn iter() {
    let t = Tensor::from_slice(&[7i64, 3, 9, 3, 11]);
    let v = t.iter::<i64>().unwrap().collect::<Vec<_>>();
    assert_eq!(v, [7, 3, 9, 3, 11]);
    let t = Tensor::from_slice(&[std::f64::consts::PI, 15.926, 5.3589, 79.0]);
    let v = t.iter::<f64>().unwrap().collect::<Vec<_>>();
    assert_eq!(v, [std::f64::consts::PI, 15.926, 5.3589, 79.0]);
}

#[test]
fn array_conversion() {
    let vec: Vec<_> = (0..6).map(|x| (x * x) as f64).collect();
    let t = Tensor::from_slice(&vec);
    assert_eq!(vec_f64_from(&t), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
    let t = t.view([3, 2]);
    assert_eq!(from::<Vec::<Vec<f64>>>(&t), [[0.0, 1.0], [4.0, 9.0], [16.0, 25.0]]);
    let t = t.view([2, 3]);
    assert_eq!(from::<Vec::<Vec<f64>>>(&t), [[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]])
}

#[test]
fn binary_ops() {
    let t = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let t = (&t * &t) + &t - 1.5;
    assert_eq!(vec_f64_from(&t), [10.5, 0.5, 18.5, 0.5, 28.5]);
}

#[test]
fn grad() {
    let mut x = Tensor::from(2.0).set_requires_grad(true);
    let y = &x * &x + &x + 36;
    x.zero_grad();
    y.backward();
    let dy_over_dx = x.grad();
    assert_eq!(vec_f64_from(&dy_over_dx), [5.0]);
}

#[test]
fn grad_grad() {
    // Compute a second order derivative using run_backward.
    let mut x = Tensor::from(42.0).set_requires_grad(true);
    let y = &x * &x * &x + &x + &x * &x;
    x.zero_grad();
    let dy_over_dx = Tensor::run_backward(&[y], &[&x], true, true);
    assert_eq!(dy_over_dx.len(), 1);
    let dy_over_dx = &dy_over_dx[0];
    dy_over_dx.backward();
    let dy_over_dx2 = x.grad();
    assert_eq!(from::<f64>(&dy_over_dx2), 254.0);
}

#[test]
#[should_panic(expected = "one of the input tensor does not use set_requires_grad")]
fn grad_without_requires() {
    let x = Tensor::from(2.0);
    let y = &x * &x + &x + 36;
    let _dy_over_dx = Tensor::run_backward(&[y], &[&x], true, true);
}

#[test]
fn cat_and_stack() {
    let t = Tensor::from_slice(&[13.0, 37.0]);
    let t = Tensor::cat(&[&t, &t, &t], 0);
    assert_eq!(t.size(), [6]);
    assert_eq!(vec_f64_from(&t), [13.0, 37.0, 13.0, 37.0, 13.0, 37.0]);

    let t = Tensor::from_slice(&[13.0, 37.0]);
    let t = Tensor::stack(&[&t, &t, &t], 0);
    assert_eq!(t.size(), [3, 2]);
    assert_eq!(vec_f64_from(&t), [13.0, 37.0, 13.0, 37.0, 13.0, 37.0]);

    let t = Tensor::from_slice(&[13.0, 37.0]);
    let t = Tensor::stack(&[&t, &t, &t], 1);
    assert_eq!(t.size(), [2, 3]);
    assert_eq!(vec_f64_from(&t), [13.0, 13.0, 13.0, 37.0, 37.0, 37.0]);
}

#[test]
fn onehot() {
    let xs = Tensor::from_slice(&[0, 1, 2, 3]);
    let onehot = xs.onehot(4);
    assert_eq!(
        vec_f64_from(&onehot),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(onehot.device(), xs.device());
    assert_eq!(onehot.size(), vec![4, 4])
}

#[test]
fn fallible() {
    // Try to compare two tensors with incompatible dimensions and check that this returns an
    // error.
    let xs = Tensor::from_slice(&[0, 1, 2, 3]);
    let ys = Tensor::from_slice(&[0, 1, 2, 3, 4]);
    assert!(xs.f_eq_tensor(&ys).is_err())
}

#[test]
fn chunk() {
    let xs = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let tensors = xs.chunk(3, 0);
    assert_eq!(tensors.len(), 3);
    assert_eq!(vec_i64_from(&tensors[0]), vec![0, 1, 2, 3]);
    assert_eq!(vec_i64_from(&tensors[1]), vec![4, 5, 6, 7]);
    assert_eq!(vec_i64_from(&tensors[2]), vec![8, 9]);
}

#[test]
fn broadcast() {
    let xs = Tensor::from_slice(&[4, 5, 3]);
    let ys = Tensor::from(42);
    let tensors = Tensor::broadcast_tensors(&[xs, ys]);
    assert_eq!(tensors.len(), 2);
    assert_eq!(vec_i64_from(&tensors[0]), vec![4, 5, 3]);
    assert_eq!(vec_i64_from(&tensors[1]), vec![42, 42, 42]);
}

#[test]
fn eq() {
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let u = &t + 1 - 1;
    assert_eq!(t, u);
    assert!(t == u);
    assert!(t != u - 1);

    let t = Tensor::from_slice(&[std::f64::consts::PI]);
    let u = Tensor::from(std::f64::consts::PI);
    // The tensor shape is important for equality.
    assert!(t != u);
    assert!(t.size() != u.size());

    let u = u.reshape([1]);
    assert_eq!(t, u);
    assert!(t == u);
    assert!(t != u - 1)
}

#[test]
fn values_at_index() {
    let t = Tensor::from(42);
    assert_eq!(t.int64_value(&[]), 42);
    assert_eq!(t.double_value(&[]), 42.0);
    assert!(t.f_int64_value(&[0]).is_err());
    assert!(t.f_double_value(&[0]).is_err());
}

#[test]
fn into_ndarray_f64() {
    let tensor = Tensor::from_slice(&[1., 2., 3., 4.]);
    let nd: ndarray::ArrayD<f64> = (&tensor).try_into().unwrap();
    assert_eq!(vec_f64_from(&tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn into_ndarray_i64() {
    let tensor = Tensor::from_slice(&[1, 2, 3, 4]);
    let nd: ndarray::ArrayD<i64> = (&tensor).try_into().unwrap();
    assert_eq!(vec_i64_from(&tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn from_ndarray_f64() {
    let nd = ndarray::arr2(&[[1f64, 2.], [3., 4.]]);
    let tensor = Tensor::try_from(nd.clone()).unwrap();
    assert_eq!(vec_f64_from(&tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn from_ndarray_i64() {
    let nd = ndarray::arr2(&[[1i64, 2], [3, 4]]);
    let tensor = Tensor::try_from(nd.clone()).unwrap();
    assert_eq!(vec_i64_from(&tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn from_ndarray_bool() {
    let nd = ndarray::arr2(&[[true, false], [true, true]]);
    let tensor = Tensor::try_from(nd.clone()).unwrap();
    assert_eq!(vec_bool_from(&tensor).as_slice(), nd.as_slice().unwrap());
}

#[test]
fn from_primitive() -> Result<()> {
    assert_eq!(vec_i32_from(&Tensor::try_from(1_i32)?), vec![1]);
    assert_eq!(vec_i64_from(&Tensor::try_from(1_i64)?), vec![1]);
    assert_eq!(vec_f16_from(&Tensor::try_from(f16::from_f64(1.0))?), vec![f16::from_f64(1.0)]);
    assert_eq!(vec_f32_from(&Tensor::try_from(1_f32)?), vec![1.0]);
    assert_eq!(vec_f64_from(&Tensor::try_from(1_f64)?), vec![1.0]);
    assert_eq!(vec_bool_from(&Tensor::try_from(true)?), vec![true]);
    Ok(())
}

#[test]
fn from_vec() -> Result<()> {
    assert_eq!(vec_i32_from(&Tensor::try_from(vec![-1_i32, 0, 1])?), vec![-1, 0, 1]);
    assert_eq!(vec_i64_from(&Tensor::try_from(vec![-1_i64, 0, 1])?), vec![-1, 0, 1]);
    assert_eq!(
        from::<Vec<f16>>(&Tensor::try_from(vec![
            f16::from_f64(-1.0),
            f16::from_f64(0.0),
            f16::from_f64(1.0)
        ])?),
        vec![f16::from_f64(-1.0), f16::from_f64(0.0), f16::from_f64(1.0)]
    );
    assert_eq!(vec_f32_from(&Tensor::try_from(vec![-1_f32, 0.0, 1.0])?), vec![-1.0, 0.0, 1.0]);
    assert_eq!(vec_f64_from(&Tensor::try_from(vec![-1_f64, 0.0, 1.0])?), vec![-1.0, 0.0, 1.0]);
    assert_eq!(vec_bool_from(&Tensor::try_from(vec![true, false])?), vec![true, false]);
    Ok(())
}

#[test]
fn from_slice() -> Result<()> {
    assert_eq!(vec_i32_from(&Tensor::try_from(&[-1_i32, 0, 1] as &[_])?), vec![-1, 0, 1]);
    assert_eq!(vec_i64_from(&Tensor::try_from(&[-1_i64, 0, 1] as &[_])?), vec![-1, 0, 1]);
    assert_eq!(
        vec_f16_from(&Tensor::try_from(&[
            f16::from_f64(-1.0),
            f16::from_f64(0.0),
            f16::from_f64(1.0)
        ] as &[_])?),
        vec![f16::from_f64(-1.0), f16::from_f64(0.0), f16::from_f64(1.0)]
    );
    assert_eq!(vec_f32_from(&Tensor::try_from(&[-1_f32, 0.0, 1.0] as &[_])?), vec![-1.0, 0.0, 1.0]);
    assert_eq!(vec_f64_from(&Tensor::try_from(&[-1_f64, 0.0, 1.0] as &[_])?), vec![-1.0, 0.0, 1.0]);
    assert_eq!(vec_bool_from(&Tensor::try_from(&[true, false] as &[_])?), vec![true, false]);
    Ok(())
}

#[test]
fn test_device() {
    let x = Tensor::from(1);
    assert_eq!(x.device(), Device::Cpu);
    let x = Tensor::from(1).to_device(Device::Cpu);
    assert_eq!(x.device(), Device::Cpu);
    if tch::Cuda::device_count() > 0 {
        let x = Tensor::from(1).to_device(Device::Cuda(0));
        assert_eq!(x.device(), Device::Cuda(0));
        let x = Tensor::from(1).to_device(Device::Cuda(0)).to_device(Device::Cpu);
        assert_eq!(x.device(), Device::Cpu);
    }
}

#[test]
fn where_() {
    let t1 = Tensor::from_slice(&[3, 1, 4, 1, 5, 9]);
    let t2 = Tensor::from_slice(&[2, 7, 1, 8, 2, 8]);
    let t = t1.where_self(&t1.lt(4), &t2);
    assert_eq!(vec_i64_from(&t), [3, 1, 1, 1, 2, 8]);
}

#[test]
fn bool_tensor() {
    let t1 = Tensor::from_slice(&[true, true, false]);
    assert_eq!(vec_i64_from(&t1), [1, 1, 0]);
    assert_eq!(vec_bool_from(&t1), [true, true, false]);
    let t1 = Tensor::from_slice(&[0, 1, 0]).to_kind(tch::Kind::Bool);
    let t2 = Tensor::from_slice(&[1, 1, 1]).to_kind(tch::Kind::Bool);
    let t1_any = t1.any();
    let t2_any = t2.any();
    let t1_all = t1.all();
    let t2_all = t2.all();
    assert!(from::<bool>(&t1_any));
    assert!(!from::<bool>(&t1_all));
    assert!(from::<bool>(&t2_any));
    assert!(from::<bool>(&t2_all));
}

#[test]
fn copy_overflow() {
    let mut s = [f32::consts::PI];
    let r = Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu)).f_copy_data(&mut s, 1);
    assert!(r.is_err());

    let mut s: [i8; 0] = [];
    let r = Tensor::zeros([10000], (tch::Kind::Int8, Device::Cpu)).f_copy_data(&mut s, 10000);
    assert!(r.is_err());
}

#[test]
fn mkldnn() {
    if tch::utils::has_mkldnn() {
        let t = Tensor::randn([5, 5, 5], (tch::Kind::Float, Device::Cpu));
        assert!(!t.is_mkldnn());
        assert!(t.to_mkldnn().is_mkldnn());
    }
}

#[test]
fn sparse() {
    let t = Tensor::from_slice(&[1, 2, 3]);
    assert!(!t.is_sparse());
}

#[test]
fn einsum() {
    // Element-wise squaring of a vector.
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let t = Tensor::einsum("i, i -> i", &[&t, &t], None::<i64>);
    assert_eq!(vec_f64_from(&t), [1.0, 4.0, 9.0]);
    // Matrix transpose
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);
    let t = Tensor::einsum("ij -> ji", &[t], None::<i64>);
    assert_eq!(vec_f64_from(&t), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    // Sum all elements
    let t = Tensor::einsum("ij -> ", &[t], None::<i64>);
    assert_eq!(vec_f64_from(&t), [21.0]);
}

#[test]
fn vec2() {
    let tensor = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]).reshape([2, 3]);
    assert_eq!(Vec::<Vec::<f64>>::try_from(tensor).unwrap(), [[1., 2., 3.], [4., 5., 6.]])
}

#[test]
fn upsample1d() {
    let tensor = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]).reshape([2, 3, 1]);
    let up1 = tensor.upsample_linear1d([2], false, 1.);
    assert_eq!(
        // Exclude the last element because of some numerical instability.
        vec_f64_from(&up1)[0..11],
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]
    );
    let up1 = tensor.upsample_linear1d([2], false, None);
    assert_eq!(vec_f64_from(&up1), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]);
}

#[test]
fn argmax() {
    let tensor = Tensor::from_slice(&[7., 2., 3., 4., 5., 6.]).reshape([2, 3]);
    let argmax = tensor.argmax(None, false);
    assert_eq!(vec_i64_from(&argmax), [0],);
    let argmax = tensor.argmax(0, false);
    assert_eq!(vec_i64_from(&argmax), [0, 1, 1],);
    let argmax = tensor.argmax(-1, false);
    assert_eq!(vec_i64_from(&argmax), [0, 2],);
}

#[test]
fn strides() {
    fn check_stride(t: &Tensor) {
        let shape = t.size();
        let ndim = shape.len();
        let mut c = 1;
        let mut strides = vec![0i64; ndim];
        strides[ndim - 1] = c;
        for i in (1..ndim).rev() {
            c *= shape[i];
            strides[i - 1] = c;
        }

        assert_eq!(t.stride(), strides);
    }

    let tensor = Tensor::zeros([2, 3, 4], tch::kind::FLOAT_CPU);
    check_stride(&tensor);

    let tensor: Tensor = Tensor::ones([3, 4, 5, 6, 7, 8], tch::kind::FLOAT_CPU);
    check_stride(&tensor);
}

#[test]
fn nested_tensor() {
    let vec: Vec<Vec<i32>> = vec![vec![1, 2], vec![1, 2], vec![4, 5]];
    let t = Tensor::from_slice2(&vec);
    assert_eq!(t.size(), [3, 2]);
    assert_eq!(vec_i32_from(&t.view([-1])), [1, 2, 1, 2, 4, 5]);
}

#[test]
fn quantized() {
    let t = Tensor::from_slice(&[-1f32, 0., 1., 2., 120., 0.42]);
    let t = t.quantize_per_tensor(0.1, 10, tch::Kind::QUInt8);
    let t = t.dequantize();
    assert_eq!(vec_f32_from(&t), [-1f32, 0., 1., 2., 24.5, 0.4]);
}

#[test]
fn nll_loss() {
    let input = Tensor::randn([3, 5], (tch::Kind::Float, Device::Cpu)).set_requires_grad(true);
    let target = Tensor::from_slice(&[1i64, 0, 4]);
    let output = input.nll_loss(&target);
    output.backward();

    let weights = Tensor::from_slice(&[1f32, 2.0, 2.0, 1.0, 1.0]);
    // This used to segfault, see https://github.com/LaurentMazare/tch-rs/issues/366
    let _output = input.g_nll_loss(&target, Some(weights), tch::Reduction::Mean, -100);
}

#[test]
fn allclose() {
    let t = Tensor::from_slice(&[-1f32, 0., 1., 2., 120., 0.42]);
    let t = t.quantize_per_tensor(0.1, 10, tch::Kind::QUInt8);
    let t = t.dequantize();
    assert!(!t.allclose(&(&t + 0.1), 1e-5, 1e-8, false));
    assert!(t.allclose(&(&t + 1e-9), 1e-5, 1e-8, false));
}

#[test]
fn set_data() {
    let mut t = Tensor::from_slice(&[-1f32, 0., 1., 2., 120., 0.42]);
    t.set_data(&t.to_kind(tch::Kind::BFloat16));
    assert_eq!(t.kind(), tch::Kind::BFloat16);
}

#[test]
fn convert_vec() {
    let t_1d = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]);
    let vec: Vec<i64> = Vec::try_from(t_1d).unwrap();
    assert_eq!(vec, vec![0, 1, 2, 3, 4, 5]);

    let t_2d = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).view((2, 3));
    let vec: Result<Vec<i64>, TchError> = Vec::try_from(t_2d);
    assert!(matches!(vec, Err(TchError::Convert(msg)) if msg==
             "Attempting to convert a Tensor with 2 dimensions to flat vector"));

    let t_2d = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).view((2, 3));
    let vec: Vec<Vec<i64>> = Vec::try_from(t_2d).unwrap();
    assert_eq!(vec, vec![vec![0, 1, 2], vec![3, 4, 5]]);
}

#[test]
fn convert_ndarray() {
    let t_1d = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]);
    let array_1d: ndarray::ArrayD<i64> = t_1d.as_ref().try_into().unwrap();
    assert_eq!(array_1d.as_slice(), ndarray::array![0, 1, 2, 3, 4, 5].as_slice());

    let t_2d = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).view((2, 3));
    let array_2d: ndarray::ArrayD<i64> = t_2d.as_ref().try_into().unwrap();
    assert_eq!(array_2d.as_slice(), ndarray::array![[0, 1, 2], [3, 4, 5]].as_slice());

    let t_3d = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]).view((2, 2, 2));
    let array_3d: ndarray::ArrayD<i64> = t_3d.as_ref().try_into().unwrap();
    assert_eq!(array_3d.as_slice(), ndarray::array![[[0, 1], [2, 3]], [[4, 5], [6, 7]]].as_slice());
}

#[test]
fn fp8_tensor() {
    let float8_types = [
        tch::Kind::Float8e5m2,
        tch::Kind::Float8e4m3fn,
        tch::Kind::Float8e5m2fnuz,
        tch::Kind::Float8e4m3fnuz,
    ];

    for &kind in &float8_types {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_kind(kind);
        assert_eq!(t.kind(), kind);
        assert_eq!(t.kind().elt_size_in_bytes(), 1);
    }
}
