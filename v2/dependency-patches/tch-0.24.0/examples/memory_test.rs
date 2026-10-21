// This example is useful to check for memory leaks.
// A large number of tensors are created either on the cpu or on the gpu and one
// can monitor the main memory usage or the gpu memory at the same time.
use tch::Tensor;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let device = match a.iter().map(|x| x.as_str()).collect::<Vec<_>>().as_slice() {
        [_] => tch::Device::Cpu,
        [_, "cpu"] => tch::Device::Cpu,
        [_, "gpu"] => tch::Device::Cuda(0),
        _ => panic!("usage: main cpu|gpu"),
    };
    let slice = vec![0; 1_000_000];
    for i in 1..1_000_000 {
        let t = Tensor::from_slice(&slice).to_device(device);
        println!("{} {:?}", i, t.size())
    }
}
