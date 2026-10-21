use tch::Tensor;

pub fn from<'a, T>(t: &'a Tensor) -> T
where
    <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
    T: TryFrom<&'a Tensor>,
{
    T::try_from(t).unwrap()
}

#[allow(dead_code)]
pub fn f64_from(t: &Tensor) -> f64 {
    from::<f64>(t)
}

#[allow(dead_code)]
pub fn vec_f64_from(t: &Tensor) -> Vec<f64> {
    from::<Vec<f64>>(&t.reshape(-1))
}

#[allow(dead_code)]
pub fn vec_f32_from(t: &Tensor) -> Vec<f32> {
    from::<Vec<f32>>(&t.reshape(-1))
}

#[allow(dead_code)]
pub fn vec_f16_from(t: &Tensor) -> Vec<half::f16> {
    from::<Vec<half::f16>>(&t.reshape(-1))
}

#[allow(dead_code)]
pub fn vec_i64_from(t: &Tensor) -> Vec<i64> {
    from::<Vec<i64>>(&t.reshape(-1))
}

#[allow(dead_code)]
pub fn vec_i32_from(t: &Tensor) -> Vec<i32> {
    from::<Vec<i32>>(&t.reshape(-1))
}

#[allow(dead_code)]
pub fn vec_bool_from(t: &Tensor) -> Vec<bool> {
    from::<Vec<bool>>(&t.reshape(-1))
}
