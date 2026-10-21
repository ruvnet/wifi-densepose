#![cfg(feature = "candle-backend")]
// ZIP9 compatibility must preserve actual tensor names and values, not just compile.
use candle_core::{Device, Tensor};
#[test]
fn npz_roundtrip_retains_unicode_and_nested_tensor_names() -> candle_core::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("tensors.npz");
    let tensor = Tensor::new(&[1f32, -2.5, 3.25], &Device::Cpu)?;
    Tensor::write_npz(&[("nested/room-α", &tensor)], &path)?;
    let restored = Tensor::read_npz(&path)?;
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].0, "nested/room-α");
    assert_eq!(restored[0].1.to_vec1::<f32>()?, vec![1., -2.5, 3.25]);
    Ok(())
}
