use std::convert::{TryFrom, TryInto};
use tch::{IValue, Kind, Tensor};

mod test_utils;
use test_utils::*;

#[test]
fn jit() {
    let x = Tensor::from_slice(&[3, 1, 4, 1, 5]).to_kind(Kind::Float);
    let y = Tensor::from_slice(&[7]).to_kind(Kind::Float);
    // The JIT module is created in create_jit_models.py
    let mod_ = tch::CModule::load("tests/foo.pt").unwrap();
    let result = mod_.forward_ts(&[&x, &y]).unwrap();
    let expected = x * 2.0 + y + 42.0;
    assert_eq!(vec_f64_from(&result), vec_f64_from(&expected));
}

#[test]
fn jit_data() {
    let x = Tensor::from_slice(&[3, 1, 4, 1, 5]).to_kind(Kind::Float);
    let y = Tensor::from_slice(&[7]).to_kind(Kind::Float);
    let mut file = std::fs::File::open("tests/foo.pt").unwrap();
    let mod_ = tch::CModule::load_data(&mut file).unwrap();
    let result = mod_.forward_ts(&[&x, &y]).unwrap();
    let expected = x * 2.0 + y + 42.0;
    assert_eq!(vec_f64_from(&result), vec_f64_from(&expected));
}

#[test]
fn jit1() {
    let mod_ = tch::CModule::load("tests/foo1.pt").unwrap();
    let result = mod_.forward_ts(&[Tensor::from(42), Tensor::from(1337)]).unwrap();
    assert_eq!(from::<i64>(&result), 1421);
    let result = mod_.method_ts("forward", &[Tensor::from(42), Tensor::from(1337)]).unwrap();
    assert_eq!(from::<i64>(&result), 1421);
}

#[test]
fn jit2() {
    let mod_ = tch::CModule::load("tests/foo2.pt").unwrap();
    let result = mod_
        .forward_is(&[IValue::from(Tensor::from(42)), IValue::from(Tensor::from(1337))])
        .unwrap();
    let expected1 = Tensor::from(1421);
    let expected2 = Tensor::from(-1295);
    assert_eq!(result, IValue::from((expected1, expected2)));
    // Destructure the tuple, using an option.
    let (v1, v2) = <(Tensor, Option<Tensor>)>::try_from(result).unwrap();
    assert_eq!(from::<i64>(&v1), 1421);
    assert_eq!(from::<i64>(&v2.unwrap()), -1295);
    let result = mod_
        .method_is("forward", &[IValue::from(Tensor::from(42)), IValue::from(Tensor::from(1337))])
        .unwrap();
    let expected1 = Tensor::from(1421);
    let expected2 = Tensor::from(-1295);
    assert_eq!(result, IValue::from((expected1, expected2)));
    let (v1, v2) = <(Tensor, Tensor)>::try_from(result).unwrap();
    assert_eq!(from::<i64>(&v1), 1421);
    assert_eq!(from::<i64>(&v2), -1295);
}

#[test]
fn jit3() {
    let mod_ = tch::CModule::load("tests/foo3.pt").unwrap();
    let xs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = mod_.forward_ts(&[xs]).unwrap();
    assert_eq!(from::<f64>(&result), 120.0);
}

#[test]
fn jit4() {
    let mod_ = tch::CModule::load("tests/foo4.pt").unwrap();
    let result = mod_.forward_is(&[IValue::from((2.0, 3.0, 4))]).unwrap();
    assert_eq!(result, 14.0.into());
    let v = f64::try_from(result).unwrap();
    assert_eq!(v, 14.0);
    let named_parameters = mod_.named_parameters().unwrap();
    assert_eq!(named_parameters, vec![]);
}

#[test]
fn profiling_mode() {
    assert!(tch::jit::get_profiling_mode());
    tch::jit::set_profiling_mode(false);
    assert!(!tch::jit::get_profiling_mode());
    tch::jit::set_profiling_mode(true);
    assert!(tch::jit::get_profiling_mode());
}

#[test]
fn tensor_expr_fuser() {
    tch::jit::set_tensor_expr_fuser_enabled(true);
    assert!(tch::jit::get_tensor_expr_fuser_enabled());
    tch::jit::set_tensor_expr_fuser_enabled(false);
    assert!(!tch::jit::get_tensor_expr_fuser_enabled());
}

#[test]
fn jit5() {
    let mod_ = tch::CModule::load("tests/foo5.pt").unwrap();
    let result = mod_
        .forward_is(&[IValue::StringList(vec![
            "foo".to_string(),
            "bar".to_string(),
            "foobar".to_string(),
        ])])
        .unwrap();
    assert_eq!(
        result,
        IValue::from(vec![IValue::from("fo"), IValue::from("ba"), IValue::from("fooba")])
    );
    // Destructuring of ivalue.
    let (v1, v2, v3) = <(String, String, String)>::try_from(result).unwrap();
    assert_eq!(v1, "fo");
    assert_eq!(v2, "ba");
    assert_eq!(v3, "fooba");
}

#[test]
fn jit6() {
    let mod_ = tch::CModule::load("tests/foo6.pt").unwrap();
    let xs = Tensor::from_slice(&[3.0, 4.0, 5.0]);
    let result = mod_.forward_is(&[IValue::Tensor(xs)]).unwrap();

    let obj = tch::jit::Object::try_from(result).unwrap();
    let result = obj.method_is::<IValue>("y", &[]).unwrap();
    assert_eq!(result, IValue::Tensor(Tensor::from_slice(&[6.0, 8.0, 10.0])));
}

#[test]
fn create_traced() {
    let mut closure = |inputs: &[Tensor]| {
        let v1 = inputs[0].shallow_clone();
        let v2 = inputs[1].shallow_clone();
        vec![v1 + v2]
    };
    let modl = tch::CModule::create_by_tracing(
        "MyModule",
        "MyFn",
        &[Tensor::from(0.0), Tensor::from(1.0)],
        &mut closure,
    )
    .unwrap();
    let filename = std::env::temp_dir().join(format!("tch-modl-{}", std::process::id()));
    modl.save(&filename).unwrap();
    let modl = tch::CModule::load(&filename).unwrap();
    let xs = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let ys = Tensor::from_slice(&[41.0, 1335.0, std::f64::consts::PI - 3., 4.0, 5.0]);
    let result = modl.method_ts("MyFn", &[xs, ys]).unwrap();
    assert_eq!(
        Vec::<f64>::try_from(&result).unwrap(),
        [42.0, 1337.0, std::f64::consts::PI, 8.0, 10.0]
    )
}

// https://github.com/LaurentMazare/tch-rs/issues/475
#[test]
fn jit_double_free() {
    let mod_ = tch::CModule::load("tests/foo7.pt").unwrap();
    let input = mod_.method_is(
        "make_input_object",
        &[
            &Tensor::from_slice(&[1_f32, 2_f32, 3_f32]).into(),
            &Tensor::from_slice(&[4_f32, 5_f32, 6_f32]).into(),
        ],
    );
    let result = mod_.method_is("add_them", &[&input.unwrap()]);
    let result = match result.unwrap() {
        IValue::Tensor(tensor) => tensor,
        result => panic!("expected a tensor got {result:?}"),
    };
    assert_eq!(Vec::<f64>::try_from(&result).unwrap(), [5.0, 7.0, 9.0])
}

// https://github.com/LaurentMazare/tch-rs/issues/597
#[test]
fn specialized_dict() {
    let mod_ = tch::CModule::load("tests/foo8.pt").unwrap();
    let input = IValue::GenericDict(vec![
        (IValue::String("bar".to_owned()), IValue::Tensor(Tensor::from_slice(&[1_f32, 7_f32]))),
        (
            IValue::String("foo".to_owned()),
            IValue::Tensor(Tensor::from_slice(&[1_f32, 2_f32, 3_f32])),
        ),
    ]);
    let result = mod_.method_is("generate", &[input]).unwrap();
    let result: (Tensor, Tensor) = result.try_into().unwrap();
    assert_eq!(Vec::<f64>::try_from(&result.0).unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(Vec::<f64>::try_from(&result.1).unwrap(), [1.0, 7.0])
}
