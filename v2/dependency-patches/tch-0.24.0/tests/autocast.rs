#[cfg(test)]
#[cfg(feature = "cuda-tests")]
mod tests {
    use tch::{autocast, Device, Kind, Tensor};

    #[test]
    fn autocast_narrows_type() {
        let device = Device::Cuda(0);

        let linear = Tensor::rand([10, 10], (Kind::Float, device));
        let input = Tensor::rand([10], (Kind::Float, device));

        autocast(true, || {
            let output1 = autocast(false, || linear.matmul(&input));
            assert_eq!(output1.kind(), Kind::Float);
            let output2 = linear.matmul(&output1);
            assert_eq!(output2.kind(), Kind::Half);
            let output3 = autocast(false, || linear.matmul(&output1));
            assert_eq!(output3.kind(), Kind::Float);
        });
    }
}
