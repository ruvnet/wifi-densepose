use candle_core::{DType, Result, Tensor};

#[test]
fn npy() -> Result<()> {
    let npy = Tensor::read_npy("tests/test.npy")?;
    assert_eq!(
        npy.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}

#[test]
fn npz() -> Result<()> {
    let npz = Tensor::read_npz("tests/test.npz")?;
    assert_eq!(npz.len(), 2);
    assert_eq!(npz[0].0, "x");
    assert_eq!(npz[1].0, "x_plus_one");
    assert_eq!(
        npz[1].1.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}
