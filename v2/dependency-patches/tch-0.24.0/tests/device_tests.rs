use tch::{Device, Tensor};

#[test]
fn tensor_device() {
    let t = Tensor::from_slice(&[3, 1, 4]);
    assert_eq!(t.device(), Device::Cpu)
}
