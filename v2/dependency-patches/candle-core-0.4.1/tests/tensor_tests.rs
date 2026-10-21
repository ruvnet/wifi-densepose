use candle_core::{test_device, test_utils, DType, Device, IndexOp, Result, Tensor, D};

fn zeros(device: &Device) -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}

fn ones(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::ones((2, 3), DType::U8, device)?.to_vec2::<u8>()?,
        [[1, 1, 1], [1, 1, 1]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::U32, device)?.to_vec2::<u32>()?,
        [[1, 1, 1], [1, 1, 1]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::I64, device)?.to_vec2::<i64>()?,
        [[1, 1, 1], [1, 1, 1]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::F32, device)?.to_vec2::<f32>()?,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    );
    assert_eq!(
        Tensor::ones((2, 3), DType::F64, device)?.to_vec2::<f64>()?,
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    );
    Ok(())
}

fn full(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::full(42u32, (2, 3), device)?.to_vec2::<u32>()?,
        [[42, 42, 42], [42, 42, 42]],
    );
    Ok(())
}

fn arange(device: &Device) -> Result<()> {
    assert_eq!(
        Tensor::arange(0u8, 5u8, device)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4],
    );
    assert_eq!(
        Tensor::arange_step(0u8, 5u8, 2, device)?.to_vec1::<u8>()?,
        [0, 2, 4],
    );
    assert_eq!(
        Tensor::arange_step(0u8, 5u8, 3, device)?.to_vec1::<u8>()?,
        [0, 3],
    );
    assert_eq!(
        Tensor::arange_step(5i64, 0i64, -1, device)?.to_vec1::<i64>()?,
        [5, 4, 3, 2, 1],
    );
    Ok(())
}

fn add_mul(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], device)?;
    let dim1 = tensor.dims1()?;
    assert_eq!(dim1, 3);
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [3., 1., 4.]);
    let tensor = Tensor::add(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [6., 2., 8.]);
    let tensor = Tensor::mul(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [36., 4., 64.]);
    Ok(())
}

fn tensor_2d(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content, data);
    Ok(())
}

fn clamp(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?;
    let tensor = tensor.clamp(1.5, 6.2)?;
    assert_eq!(
        tensor.to_vec2::<f32>()?,
        [[3.0, 1.5, 4.0, 1.5, 5.0], [2.0, 1.5, 6.2, 6.2, 2.0]],
    );
    Ok(())
}

fn unary_op(device: &Device) -> Result<()> {
    let data = &[[-3f32, 1., 4., -0.1, 0.5], [2.7, -1.8, -0.28, 1.8, 2.8]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        test_utils::to_vec2_round(&tensor.gelu()?, 4)?,
        [
            [-0.0036, 0.8412, 3.9999, -0.046, 0.3457],
            [2.6911, -0.0647, -0.1091, 1.7353, 2.7933]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.gelu_erf()?, 4)?,
        [
            [-0.004, 0.8413, 3.9999, -0.046, 0.3457],
            [2.6906, -0.0647, -0.1091, 1.7353, 2.7928]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.erf()?, 4)?,
        [
            [-1.0, 0.8427, 1.0, -0.1125, 0.5205],
            [0.9999, -0.9891, -0.3079, 0.9891, 0.9999]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.silu()?, 4)?,
        [
            [-0.1423, 0.7311, 3.9281, -0.0475, 0.3112],
            [2.53, -0.2553, -0.1205, 1.5447, 2.6395]
        ]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.ceil()?, 4)?,
        [[-3.0, 1.0, 4.0, -0.0, 1.0], [3.0, -1.0, -0.0, 2.0, 3.0]]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.floor()?, 4)?,
        [[-3.0, 1.0, 4.0, -1.0, 0.0], [2.0, -2.0, -1.0, 1.0, 2.0]]
    );
    assert_eq!(
        test_utils::to_vec2_round(&tensor.round()?, 4)?,
        [[-3.0, 1.0, 4.0, -0.0, 1.0], [3.0, -2.0, -0.0, 2.0, 3.0]]
    );
    let tensor = Tensor::new(&[2997.9246, 314.15926f32], device)?;
    assert_eq!(
        test_utils::to_vec1_round(&tensor.round_to(2)?, 4)?,
        [2997.92, 314.16]
    );
    assert_eq!(
        test_utils::to_vec1_round(&tensor.round_to(-2)?, 4)?,
        [3000.0, 300.]
    );
    Ok(())
}

fn binary_op(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, device)?;
    let tensor = (&tensor1 + (&tensor1 * &tensor1)? / (&tensor1 + &tensor2))?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    #[allow(clippy::eq_op)]
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);

    let min = tensor1.minimum(&(&tensor2 * 0.5)?)?;
    let max = tensor1.maximum(&(&tensor2 * 0.5)?)?;
    assert_eq!(
        min.to_vec2::<f32>()?,
        [[2.5, 1.0, 2.5, 1.0, 2.5], [1.0, 0.5, 3.5, 4.0, 1.0]],
    );
    assert_eq!(
        max.to_vec2::<f32>()?,
        [[3.0, 2.5, 4.0, 2.5, 5.0], [2.0, 1.0, 7.0, 8.0, 2.0]]
    );
    Ok(())
}

fn transpose(device: &Device) -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, device)?.t()?;
    let dims = tensor.dims2()?;
    assert_eq!(dims, (5, 2));
    assert_eq!(
        tensor.to_vec2::<f32>()?,
        &[[3f32, 2.], [1., 1.], [4., 7.], [1., 8.], [5., 2.]]
    );
    assert_eq!(tensor.t()?.to_vec2::<f32>()?, data);
    assert_eq!(tensor.contiguous()?.t()?.to_vec2::<f32>()?, data);
    assert_eq!(((tensor + 1.)?.t()? - 1.)?.to_vec2::<f32>()?, data);
    Ok(())
}

fn var(device: &Device) -> Result<()> {
    // Values taken from https://pytorch.org/docs/stable/generated/torch.var.html
    let data = &[
        [0.2035f32, 1.2959, 1.8101, -0.4644],
        [1.5027, -0.3270, 0.5905, 0.6538],
        [-1.5745, 1.3330, -0.5596, -0.6548],
        [0.1264, -0.5080, 1.6420, 0.1992],
    ];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        test_utils::to_vec2_round(&tensor.var_keepdim(1)?, 4)?,
        &[[1.0631], [0.559], [1.4893], [0.8258]]
    );
    Ok(())
}

fn sum(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.sum_keepdim(2)?.to_vec3::<u32>()?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec3::<u32>()?,
        &[[[5, 2, 11], [9, 7, 17]]],
    );
    assert_eq!(tensor.sum_keepdim((0, 2, 1))?.to_vec3::<u32>()?, &[[[51]]],);
    assert_eq!(
        tensor.t()?.sum_keepdim(1)?.t()?.to_vec3::<u32>()?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum_keepdim((2, 1))?.to_vec3::<u32>()?,
        &[[[8 + 15]], [[10 + 18]]]
    );
    let data: Vec<u32> = (0..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.sum_keepdim(0)?.to_vec1::<u32>()?, &[7998000]);
    let tensor = tensor.reshape((2000, 2))?;
    assert_eq!(tensor.sum_keepdim((0, 1))?.to_vec2::<u32>()?, &[[7998000]]);
    assert_eq!(
        tensor.sum_keepdim(0)?.sum_keepdim(1)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[3998000, 4000000]]
    );

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(tensor.sum_keepdim((0, 1))?.to_vec2::<u32>()?, &[[7998000]]);
    assert_eq!(
        tensor.sum_keepdim(0)?.sum_keepdim(1)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(1)?.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[7998000]]
    );
    assert_eq!(
        tensor.sum_keepdim(0)?.to_vec2::<u32>()?,
        &[[3998000, 4000000]]
    );

    let t1 = tensor.reshape((200, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor.sum_keepdim((0, 1, 2))?.to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(0)?
                .sum_keepdim(2)?
                .sum_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(0)?
                .sum_keepdim((1, 2))?
                .to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor
                .sum_keepdim(1)?
                .sum_keepdim((0, 2))?
                .to_vec3::<u32>()?,
            &[[[7998000]]]
        );
        assert_eq!(
            tensor.sum_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [398000, 398200, 398400, 398600],
                [398800, 399000, 399200, 399400],
                [399600, 399800, 400000, 400200],
                [400400, 400600, 400800, 401000],
                [401200, 401400, 401600, 401800]
            ]]
        );
    }
    Ok(())
}

fn min(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.min_keepdim(2)?.to_vec3::<u32>()?,
        &[[[1], [1]], [[1], [2]]]
    );
    assert_eq!(
        tensor.min_keepdim(0)?.to_vec3::<u32>()?,
        &[[[2, 1, 4], [1, 2, 8]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.min_keepdim(0)?.to_vec1::<u32>()?, &[200]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.min_keepdim(0)?.min_keepdim(1)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1)?.min_keepdim(0)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(tensor.min_keepdim(0)?.to_vec2::<u32>()?, &[[200, 201]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.min_keepdim(0)?.min_keepdim(1)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(
        tensor.min_keepdim(1)?.min_keepdim(0)?.to_vec2::<u32>()?,
        &[[200]]
    );
    assert_eq!(tensor.min_keepdim(0)?.to_vec2::<u32>()?, &[[200, 201]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .min_keepdim(0)?
                .min_keepdim(2)?
                .min_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[200]]]
        );
        assert_eq!(
            tensor.min_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [200, 201, 202, 203],
                [204, 205, 206, 207],
                [208, 209, 210, 211],
                [212, 213, 214, 215],
                [216, 217, 218, 219]
            ]]
        );
    }
    Ok(())
}

fn max(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.max_keepdim(2)?.to_vec3::<u32>()?,
        &[[[4], [9]], [[7], [8]]]
    );
    assert_eq!(
        tensor.max_keepdim(0)?.to_vec3::<u32>()?,
        &[[[3, 1, 7], [8, 5, 9]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.max_keepdim(0)?.to_vec1::<u32>()?, &[3999]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor.max_keepdim(0)?.max_keepdim(1)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1)?.max_keepdim(0)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(tensor.max_keepdim(0)?.to_vec2::<u32>()?, &[[3998, 3999]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor.max_keepdim(0)?.max_keepdim(1)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(
        tensor.max_keepdim(1)?.max_keepdim(0)?.to_vec2::<u32>()?,
        &[[3999]]
    );
    assert_eq!(tensor.max_keepdim(0)?.to_vec2::<u32>()?, &[[3998, 3999]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .max_keepdim(0)?
                .max_keepdim(2)?
                .max_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[3999]]]
        );
        assert_eq!(
            tensor.max_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [3980, 3981, 3982, 3983],
                [3984, 3985, 3986, 3987],
                [3988, 3989, 3990, 3991],
                [3992, 3993, 3994, 3995],
                [3996, 3997, 3998, 3999]
            ]]
        );
    }
    Ok(())
}

fn argmin(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmin_keepdim(2)?.to_vec3::<u32>()?,
        &[[[1], [0]], [[1], [1]]]
    );
    assert_eq!(
        tensor.argmin_keepdim(0)?.to_vec3::<u32>()?,
        &[[[1, 0, 0], [0, 1, 1]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec1::<u32>()?, &[0]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor
            .argmin_keepdim(0)?
            .argmin_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmin_keepdim(1)?
            .argmin_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec2::<u32>()?, &[[0, 0]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor
            .argmin_keepdim(0)?
            .argmin_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmin_keepdim(1)?
            .argmin_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmin_keepdim(0)?.to_vec2::<u32>()?, &[[0, 0]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .argmin_keepdim(0)?
                .argmin_keepdim(2)?
                .argmin_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[0]]]
        );
        assert_eq!(
            tensor.argmin_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]]
        );
    }
    Ok(())
}

fn argmax(device: &Device) -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.argmax_keepdim(2)?.to_vec3::<u32>()?,
        &[[[2], [2]], [[2], [0]]]
    );
    assert_eq!(
        tensor.argmax_keepdim(0)?.to_vec3::<u32>()?,
        &[[[0, 0, 1], [1, 0, 0]]],
    );
    let data: Vec<u32> = (200..4000u32).collect();
    let tensor = Tensor::new(data.as_slice(), device)?;
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec1::<u32>()?, &[3799]);
    let tensor = tensor.reshape((1900, 2))?;
    assert_eq!(
        tensor
            .argmax_keepdim(0)?
            .argmax_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmax_keepdim(1)?
            .argmax_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec2::<u32>()?, &[[1899, 1899]]);

    // Make the tensor non contiguous.
    let tensor = tensor.t()?.contiguous()?.t()?;
    assert_eq!(
        tensor
            .argmax_keepdim(0)?
            .argmax_keepdim(1)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(
        tensor
            .argmax_keepdim(1)?
            .argmax_keepdim(0)?
            .to_vec2::<u32>()?,
        &[[0]]
    );
    assert_eq!(tensor.argmax_keepdim(0)?.to_vec2::<u32>()?, &[[1899, 1899]]);

    let t1 = tensor.reshape((190, 5, 4))?;
    let t2 = t1.transpose(0, 2)?.contiguous()?.transpose(0, 2)?;
    for tensor in [t1, t2] {
        assert_eq!(
            tensor
                .argmax_keepdim(0)?
                .argmax_keepdim(2)?
                .argmax_keepdim(1)?
                .to_vec3::<u32>()?,
            &[[[0]]]
        );
        assert_eq!(
            tensor.argmax_keepdim(0)?.to_vec3::<u32>()?,
            &[[
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
                [189, 189, 189, 189],
            ]]
        );
    }
    Ok(())
}

fn narrow(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.narrow(2, 1, 2)?.to_vec3::<f32>()?,
        &[[[1.0, 4.0], [5.0, 9.0]], [[1.0, 7.0], [2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(1, 1, 1)?.to_vec3::<f32>()?,
        &[[[1.0, 5.0, 9.0]], [[8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 0, 1)?.to_vec3::<f32>()?,
        &[[[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 1, 1)?.to_vec3::<f32>()?,
        &[[[2.0, 1.0, 7.0], [8.0, 2.0, 8.0]]],
    );
    // The following has been checked against PyTorch via:
    //   import torch
    //   t = torch.tensor([[[3., 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]])
    //   t.transpose(-1, -2).narrow(1, 1, 2)
    assert_eq!(
        tensor.t()?.narrow(1, 1, 2)?.to_vec3::<f32>()?,
        &[[[1.0, 5.0], [4.0, 9.0]], [[1.0, 2.0], [7.0, 8.0]]],
    );
    Ok(())
}

fn broadcast(device: &Device) -> Result<()> {
    let data = &[3f32, 1., 4.];
    let tensor = Tensor::new(data, device)?;
    assert_eq!(
        tensor.broadcast_left((3, 1))?.to_vec3::<f32>()?,
        &[[[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]]]
    );
    Ok(())
}

fn cat(device: &Device) -> Result<()> {
    // 1D
    let t1 = Tensor::new(&[3f32, 1., 4.], device)?;
    let t2 = Tensor::new(&[1f32, 5., 9., 2.], device)?;
    let t3 = Tensor::new(&[6f32, 5., 3., 5., 8., 9.], device)?;
    assert_eq!(Tensor::cat(&[&t1], 0)?.to_vec1::<f32>()?, [3f32, 1., 4.],);
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec1::<f32>()?,
        [3f32, 1., 4., 1., 5., 9., 2.],
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2, &t3], 0)?.to_vec1::<f32>()?,
        [3f32, 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9.],
    );

    // 2D
    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, device)?;
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    // PyTorch equivalent:
    //     import torch
    //     t1 = torch.tensor([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2]])
    //     t2 = torch.tensor([[5]*5, [2, 7, 1, 8, 2]])
    //     torch.cat([t1.t(), t2.t()], dim=1).t()
    assert_eq!(
        Tensor::cat(&[&t1.t()?, &t2.t()?], 1)?
            .t()?
            .to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 1)?.to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0, 2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    Ok(())
}

fn embeddings(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let hs = t.embedding(&ids)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 1.0], [4.0, 5.0], [2.0, 3.0]]);
    Ok(())
}

fn cmp(device: &Device) -> Result<()> {
    let t1 = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], device)?;
    let t2 = Tensor::new(&[[1f32, 0f32], [3f32, 3f32], [4f32, 7f32]], device)?;
    assert_eq!(t1.eq(&t2)?.to_vec2::<u8>()?, &[[0, 0], [0, 1], [1, 0]]);
    assert_eq!(t1.ne(&t2)?.to_vec2::<u8>()?, &[[1, 1], [1, 0], [0, 1]]);
    assert_eq!(t1.le(&t2)?.to_vec2::<u8>()?, &[[1, 0], [1, 1], [1, 1]]);
    assert_eq!(t1.lt(&t2)?.to_vec2::<u8>()?, &[[1, 0], [1, 0], [0, 1]]);
    assert_eq!(t1.gt(&t2)?.to_vec2::<u8>()?, &[[0, 1], [0, 0], [0, 0]]);
    assert_eq!(t1.ge(&t2)?.to_vec2::<u8>()?, &[[0, 1], [0, 1], [1, 0]]);
    Ok(())
}

fn index_select(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 2u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let hs = t.index_select(&ids, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0.0, 2.0, 1.0],
            [3.0, 5.0, 4.0],
            [6.0, 8.0, 7.0],
            [9.0, 11.0, 10.0]
        ]
    );
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [3.0, 4.0, 5.0]]
    );
    // Prior to https://github.com/huggingface/candle/pull/1022
    // There would be a bug where the last values in the result tensor would be set to 0.
    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 2u32, 1u32], device)?;
    let hs = t.index_select(&ids, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [6.0, 7.0, 8.0],
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0],
            [6.0, 7.0, 8.0],
            [3.0, 4.0, 5.0],
        ]
    );

    // Test when selecting dim > 0 with ids size different from elem count of
    // target dim in source/input.
    let ids = Tensor::new(&[1u32, 0u32, 1u32], device)?;
    let t = Tensor::arange(1f32, 5f32, device)?.reshape((2, 2))?;
    assert_eq!(t.to_vec2::<f32>()?, &[[1.0, 2.0], [3.0, 4.0]]);
    let hs = t.index_select(&ids, 1)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[2.0, 1.0, 2.0], [4.0, 3.0, 4.0]]);

    Ok(())
}

fn index_add(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[0u32, 1u32, 1u32], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let init = Tensor::ones((4, 2), DType::F32, device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[1.0, 4.0], [4.0, 10.0], [7.0, 16.0], [10.0, 22.0]],
    );
    let init = Tensor::zeros((4, 2), DType::F32, device)?;
    let ids = Tensor::new(&[1u32, 0u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[3.0, 0.0], [9.0, 3.0], [15.0, 6.0], [21.0, 9.0]],
    );

    let init = Tensor::zeros((6, 3), DType::F32, device)?;
    let ids = Tensor::new(&[5u32, 0u32, 1u32, 0u32], device)?;
    let hs = init.index_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [12.0, 14.0, 16.0],
            [6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0]
        ]
    );
    Ok(())
}

fn slice_scatter(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let src = Tensor::arange(100f32, 106f32, device)?.reshape((2, 3))?;
    assert_eq!(
        t.slice_scatter0(&src, 0)?.to_vec2::<f32>()?,
        &[
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    assert_eq!(
        t.slice_scatter0(&src, 1)?.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
            [9.0, 10.0, 11.0]
        ]
    );
    assert_eq!(
        t.slice_scatter0(&src, 2)?.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [100.0, 101.0, 102.0],
            [103.0, 104.0, 105.0],
        ]
    );
    Ok(())
}

fn scatter_add(device: &Device) -> Result<()> {
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], device)?;
    let init = Tensor::ones((4, 5), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [1.0, 2.0, 3.0, 1.0, 1.0],
            [6.0, 1.0, 1.0, 4.0, 5.0],
            [1.0, 9.0, 1.0, 14.0, 1.0],
            [11.0, 1.0, 10.0, 1.0, 12.0]
        ]
    );

    let init = Tensor::ones((6, 3), DType::F32, device)?;
    let hs = init.scatter_add(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [1.0, 11.0, 6.0],
            [1.0, 2.0, 9.0],
            [10.0, 1.0, 3.0],
            [10.0, 8.0, 1.0],
            [1.0, 5.0, 12.0],
            [1.0, 1.0, 1.0]
        ]
    );
    Ok(())
}

fn gather(device: &Device) -> Result<()> {
    let ids = Tensor::new(&[[0u32], [2u32], [1u32], [0u32]], device)?;
    let t = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let hs = t.gather(&ids, 1)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0], [5.0], [7.0], [9.0]]);
    let ids = Tensor::new(
        &[[0u32, 0u32], [2u32, 0u32], [1u32, 1u32], [0u32, 2u32]],
        device,
    )?;
    let hs = t.gather(&ids, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[[0.0, 0.0], [5.0, 3.0], [7.0, 7.0], [9.0, 11.0]]
    );
    let ids = Tensor::new(&[[0u32, 2u32, 0u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 7.0, 2.0]]);
    let ids = Tensor::new(&[[0u32, 2u32, 0u32], [0u32, 1u32, 1u32]], device)?;
    let hs = t.gather(&ids, 0)?;
    assert_eq!(hs.to_vec2::<f32>()?, &[[0.0, 7.0, 2.0], [0.0, 4.0, 5.0]]);
    Ok(())
}

fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3::<f32>()?, &expected);

    // Also perform the matmul on contiguous transposed versions.
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(!a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), &[6, 1, 2]);

    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(!b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), &[6, 1, 3]);

    assert_eq!(a_tt.matmul(&b)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a_tt.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    Ok(())
}

fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), &[3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            // With cuda, we see errors of up to ~1e-12.
            assert!(sum_diff2.to_vec0::<f32>()? < 1e-6)
        }
    }
    Ok(())
}

fn broadcasting(device: &Device) -> Result<()> {
    let t1 = Tensor::arange(0f32, 24f32, device)?.reshape((4, 2, 3))?;
    let t2 = Tensor::new(&[100f32, 200f32], device)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[100.0, 101.0, 102.0], [203.0, 204.0, 205.0]],
            [[106.0, 107.0, 108.0], [209.0, 210.0, 211.0]],
            [[112.0, 113.0, 114.0], [215.0, 216.0, 217.0]],
            [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]
        ]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[100.0, 203.0], [101.0, 204.0], [102.0, 205.0]],
            [[106.0, 209.0], [107.0, 210.0], [108.0, 211.0]],
            [[112.0, 215.0], [113.0, 216.0], [114.0, 217.0]],
            [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]
        ]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-100.0, -99.0, -98.0], [-197.0, -196.0, -195.0]],
            [[-94.0, -93.0, -92.0], [-191.0, -190.0, -189.0]],
            [[-88.0, -87.0, -86.0], [-185.0, -184.0, -183.0]],
            [[-82.0, -81.0, -80.0], [-179.0, -178.0, -177.0]]
        ]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-100.0, -197.0], [-99.0, -196.0], [-98.0, -195.0]],
            [[-94.0, -191.0], [-93.0, -190.0], [-92.0, -189.0]],
            [[-88.0, -185.0], [-87.0, -184.0], [-86.0, -183.0]],
            [[-82.0, -179.0], [-81.0, -178.0], [-80.0, -177.0]]
        ]
    );
    // Test a narrowed version as this uses a layout start_offset.
    let t1 = t1.i(2..)?;
    let s = t1.broadcast_add(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[112.0, 113.0, 114.0], [215.0, 216.0, 217.0]],
            [[118.0, 119.0, 120.0], [221.0, 222.0, 223.0]]
        ]
    );
    let s = t1.t()?.broadcast_add(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[112.0, 215.0], [113.0, 216.0], [114.0, 217.0]],
            [[118.0, 221.0], [119.0, 222.0], [120.0, 223.0]]
        ]
    );
    let s = t1.broadcast_sub(&t2.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-88.0, -87.0, -86.0], [-185.0, -184.0, -183.0]],
            [[-82.0, -81.0, -80.0], [-179.0, -178.0, -177.0]]
        ]
    );
    let s = t1.t()?.broadcast_sub(&t2)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[-88.0, -185.0], [-87.0, -184.0], [-86.0, -183.0]],
            [[-82.0, -179.0], [-81.0, -178.0], [-80.0, -177.0]]
        ]
    );
    let t3 = Tensor::new(1f32, device)?.broadcast_div(&t2)?;
    let s = t1.broadcast_mul(&t2.reshape((2, 1))?)?;
    let s_div = t1.broadcast_div(&t3.reshape((2, 1))?)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[1200.0, 1300.0, 1400.0], [3000.0, 3200.0, 3400.0]],
            [[1800.0, 1900.0, 2000.0], [4200.0, 4400.0, 4600.0]]
        ]
    );
    assert_eq!(s.to_vec3::<f32>()?, s_div.to_vec3::<f32>()?,);
    let s = t1.t()?.broadcast_mul(&t2)?;
    let s_div = t1.t()?.broadcast_div(&t3)?;
    assert_eq!(
        s.to_vec3::<f32>()?,
        &[
            [[1200.0, 3000.0], [1300.0, 3200.0], [1400.0, 3400.0]],
            [[1800.0, 4200.0], [1900.0, 4400.0], [2000.0, 4600.0]]
        ]
    );
    assert_eq!(s.to_vec3::<f32>()?, s_div.to_vec3::<f32>()?,);
    Ok(())
}

fn randn(device: &Device) -> Result<()> {
    let tensor = Tensor::randn(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    let tensor = Tensor::rand(0f32, 1f32, (5, 3), device)?;
    assert_eq!(tensor.dims(), [5, 3]);
    Ok(())
}

test_device!(zeros, zeros_cpu, zeros_gpu, zeros_metal);
test_device!(ones, ones_cpu, ones_gpu, ones_metal);
test_device!(full, full_cpu, full_gpu, full_metal);
test_device!(arange, arange_cpu, arange_gpu, arange_metal);
test_device!(add_mul, add_mul_cpu, add_mul_gpu, add_mul_metal);
test_device!(tensor_2d, tensor_2d_cpu, tensor_2d_gpu, tensor_2d_metal);
test_device!(narrow, narrow_cpu, narrow_gpu, narrow_metal);
test_device!(broadcast, broadcast_cpu, broadcast_gpu, broadcast_metal);
test_device!(cat, cat_cpu, cat_gpu, cat_metal);
test_device!(sum, sum_cpu, sum_gpu, sum_metal);
test_device!(min, min_cpu, min_gpu, min_metal);
test_device!(max, max_cpu, max_gpu, max_metal);
test_device!(argmax, argmax_cpu, argmax_gpu, argmax_metal);
test_device!(argmin, argmin_cpu, argmin_gpu, argmin_metal);
test_device!(transpose, transpose_cpu, transpose_gpu, transpose_metal);
test_device!(unary_op, unary_op_cpu, unary_op_gpu, unary_op_metal);
test_device!(binary_op, binary_op_cpu, binary_op_gpu, binary_op_metal);
test_device!(embeddings, embeddings_cpu, embeddings_gpu, embeddings_metal);
test_device!(cmp, cmp_cpu, cmp_gpu, cmp_metal);
test_device!(matmul, matmul_cpu, matmul_gpu, matmul_metal);
test_device!(
    broadcast_matmul,
    broadcast_matmul_cpu,
    broadcast_matmul_gpu,
    broadcast_matmul_metal
);
test_device!(
    broadcasting,
    broadcasting_cpu,
    broadcasting_gpu,
    broadcasting_metal
);
test_device!(
    index_select,
    index_select_cpu,
    index_select_gpu,
    index_select_metal
);
test_device!(index_add, index_add_cpu, index_add_gpu, index_add_metal);
test_device!(gather, gather_cpu, gather_gpu, gather_metal);
test_device!(
    scatter_add,
    scatter_add_cpu,
    scatter_add_gpu,
    scatter_add_metal
);
test_device!(
    slice_scatter,
    slice_scatter_cpu,
    slice_scatter_gpu,
    slice_scatter_metal
);
test_device!(randn, randn_cpu, randn_gpu, randn_metal);
test_device!(clamp, clamp_cpu, clamp_gpu, clamp_metal);
test_device!(var, var_cpu, var_gpu, var_metal);

// There was originally a bug on the CPU implementation for randn
// https://github.com/huggingface/candle/issues/381
#[test]
fn randn_hasneg() -> Result<()> {
    let t = Tensor::randn(0f32, 1f32, 200, &Device::Cpu)?.to_vec1::<f32>()?;
    if t.iter().all(|&v| v >= 0.) {
        candle_core::bail!("all values in tensors are non-negative")
    }
    Ok(())
}

#[test]
fn pad_with_same() -> Result<()> {
    let t = Tensor::arange(1f32, 5f32, &Device::Cpu)?.reshape((2, 2))?;
    let t0 = t.pad_with_same(0, 1, 2)?;
    assert_eq!(
        t0.to_vec2::<f32>()?,
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
    );
    let t1 = t.pad_with_same(1, 1, 2)?;
    assert_eq!(
        t1.to_vec2::<f32>()?,
        [[1.0, 1.0, 2.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 4.0]]
    );
    Ok(())
}

#[test]
fn i64_abs() -> Result<()> {
    let t = Tensor::new(&[-42i64, 1337], &Device::Cpu)?;
    let t = t.abs()?;
    assert_eq!(t.to_vec1::<i64>()?, [42, 1337]);
    Ok(())
}

#[test]
fn tril_triu_eye() -> Result<()> {
    let t = Tensor::tril2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
    );
    let t = Tensor::triu2(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    let t = Tensor::eye(4, DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    Ok(())
}

#[test]
fn cumsum() -> Result<()> {
    let t = &[3f32, 1., 4., 1., 5.];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(t.cumsum(0)?.to_vec1::<f32>()?, [3., 4., 8., 9., 14.]);
    let t = t.unsqueeze(1)?;
    assert_eq!(
        t.cumsum(0)?.to_vec2::<f32>()?,
        [[3.0], [4.0], [8.0], [9.0], [14.0]]
    );
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [[3.0], [1.0], [4.0], [1.0], [5.0]]
    );
    let t = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let t = Tensor::new(t, &Device::Cpu)?;
    assert_eq!(
        t.cumsum(1)?.to_vec2::<f32>()?,
        [[3.0, 4.0, 8.0, 9.0, 14.0], [2.0, 3.0, 10.0, 18.0, 20.0]],
    );
    assert_eq!(
        t.cumsum(0)?.to_vec2::<f32>()?,
        [[3.0, 1.0, 4.0, 1.0, 5.0], [5.0, 2.0, 11.0, 9.0, 7.0]]
    );
    Ok(())
}

/// A helper function for floating point comparison. Both a and b must be 1D Tensor and contains the same amount of data.
/// Assertion passes if the difference of all pairs of a and b is smaller than epsilon.
fn assert_close(a: &Tensor, b: &Tensor, epsilon: f64) -> Result<()> {
    let a_vec: Vec<f64> = a.to_vec1()?;
    let b_vec: Vec<f64> = b.to_vec1()?;

    assert_eq!(a_vec.len(), b_vec.len());
    for (a, b) in a_vec.iter().zip(b_vec.iter()) {
        assert!((a - b).abs() < epsilon);
    }
    Ok(())
}

#[test]
fn log_sum_exp() -> Result<()> {
    let input = Tensor::new(&[[1f64, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let output = input.log_sum_exp(D::Minus1)?;
    // The expectations obtained from pytorch.
    let expected = Tensor::new(&[3.4076, 6.4076], &Device::Cpu)?;
    assert_close(&output, &expected, 0.00001)?;
    Ok(())
}

#[test]
fn pow() -> Result<()> {
    let lhs = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let rhs = (&lhs - 2.)?;
    let res = lhs.pow(&rhs)?;
    assert_eq!(
        test_utils::to_vec2_round(&res, 4)?,
        [[1.0, 1.0, 3.0], [16.0, 125.0, 1296.0001]]
    );
    Ok(())
}
