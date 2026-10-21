use std::fs;
use tch::nn::OptimizerConfig;
use tch::{nn, nn::linear, nn::Init, nn::VarStore, Device, Kind, TchError, Tensor};

mod test_utils;
use test_utils::*;

#[test]
fn path_components() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();
    let path = root.sub("a");
    let path = path.sub("test");
    assert_eq!(path.components().collect::<Vec<_>>(), vec!["a", "test"]);
}

#[test]
fn var_store_entry() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let t1 = root.entry("key").or_zeros(&[3, 1, 4]);
    let t2 = root.entry("key").or_zeros(&[1, 5, 9]);

    assert_eq!(t1.size(), &[3, 1, 4]);
    assert_eq!(t2.size(), &[3, 1, 4]);
}

#[test]
fn save_and_load_var_store() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-load-complete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let mut vs1 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let mut vs2 = VarStore::new(Device::Cpu);
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 2.0);
    fs::remove_file(&filename).unwrap();

    // Save and reload in half-precision
    vs1.half();
    let mut vs2 = VarStore::new(Device::Cpu);
    let (u2, v2) = add(&vs2.root());
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(u2.kind(), Kind::Half);
    assert_eq!(v2.kind(), Kind::Half);
    assert_eq!(f64_from(&u2.mean(Kind::Half)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Half)), 2.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn save_to_stream_and_load_var_store() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-load-stream-complete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let mut vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save_to_stream(std::fs::File::create(&filename).unwrap()).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 2.0);
    fs::remove_file(&filename).unwrap();

    // Save and reload in half-precision
    vs1.half();
    let mut vs2 = VarStore::new(Device::Cpu);
    let (u2, v2) = add(&vs2.root());
    vs1.save_to_stream(std::fs::File::create(&filename).unwrap()).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(u2.kind(), Kind::Half);
    assert_eq!(v2.kind(), Kind::Half);
    assert_eq!(f64_from(&u2.mean(Kind::Half)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Half)), 2.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn save_and_load_from_stream_var_store() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-load-stream-complete2-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let mut vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load_from_stream(std::fs::File::open(&filename).unwrap()).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 2.0);
    fs::remove_file(&filename).unwrap();

    // Save and reload in half-precision
    vs1.half();
    let mut vs2 = VarStore::new(Device::Cpu);
    let (u2, v2) = add(&vs2.root());
    vs1.save(&filename).unwrap();
    vs2.load_from_stream(std::fs::File::open(&filename).unwrap()).unwrap();
    assert_eq!(u2.kind(), Kind::Half);
    assert_eq!(v2.kind(), Kind::Half);
    assert_eq!(f64_from(&u2.mean(Kind::Half)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Half)), 2.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn save_and_load_partial_var_store() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-partial-load-complete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let mut vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    let missing_variables = vs2.load_partial(&filename).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 2.0);
    assert!(missing_variables.is_empty());
    fs::remove_file(&filename).unwrap();

    // Save and reload in half-precision
    vs1.half();
    let mut vs2 = VarStore::new(Device::Cpu);
    let (u2, v2) = add(&vs2.root());
    vs1.save(&filename).unwrap();
    let missing_variables = vs2.load_partial(&filename).unwrap();
    assert_eq!(u2.kind(), Kind::Half);
    assert_eq!(v2.kind(), Kind::Half);
    assert_eq!(f64_from(&u2.mean(Kind::Half)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Half)), 2.0);
    assert!(missing_variables.is_empty());
    fs::remove_file(filename).unwrap();
}

#[test]
#[should_panic]
fn save_and_load_var_store_incomplete_file() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-load-incomplete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        u
    };
    let add_partial = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let mut u1 = add(&vs1.root());
    let (u2, v2) = add_partial(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn save_and_load_partial_var_store_incomplete_file() {
    let filename =
        std::env::temp_dir().join(format!("tch-vs-partial-load-incomplete-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        u
    };
    let add_partial = |vs: &tch::nn::Path| {
        let v = vs.sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let mut vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let mut u1 = add(&vs1.root());
    let (u2, v2) = add_partial(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    let missing_variables = vs2.load_partial(&filename).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    assert_eq!(missing_variables, vec!(String::from("a.b.t2")));
    fs::remove_file(&filename).unwrap();

    // Save and reload in half-precision
    vs1.half();
    let mut vs2 = VarStore::new(Device::Cpu);
    let (u2, v2) = add_partial(&vs2.root());
    vs1.save(&filename).unwrap();
    let missing_variables = vs2.load_partial(&filename).unwrap();
    assert_eq!(u2.kind(), Kind::Half);
    assert_eq!(v2.kind(), Kind::Float);
    assert_eq!(f64_from(&u2.mean(Kind::Half)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    assert_eq!(missing_variables, vec!(String::from("a.b.t2")));
    fs::remove_file(&filename).unwrap();
}

#[test]
fn init_test() {
    tch::manual_seed(42);
    let vs = VarStore::new(Device::Cpu);
    let zeros = vs.root().zeros("t1", &[3]);
    assert_eq!(vec_f64_from(&zeros), [0., 0., 0.]);
    let zeros = vs.root().var("t2", &[3], Init::Const(0.));
    assert_eq!(vec_f64_from(&zeros), [0., 0., 0.]);
    let ones = vs.root().var("t3", &[3], Init::Const(1.));
    assert_eq!(vec_f64_from(&ones), [1., 1., 1.]);
    let ones = vs.root().var("t4", &[3], Init::Const(0.5));
    assert_eq!(vec_f64_from(&ones), [0.5, 0.5, 0.5]);
    let forty_two = vs.root().var("t4", &[2], Init::Const(42.));
    assert_eq!(vec_f64_from(&forty_two), [42., 42.]);
    let uniform = vs.root().var("t5", &[100], Init::Uniform { lo: 1.0, up: 2.0 });
    let uniform_min = f64_from(&uniform.min());
    let uniform_max = f64_from(&uniform.max());
    assert!(uniform_min >= 1., "{}", "min {uniform_min}");
    assert!(uniform_max <= 2., "{}", "max {uniform_max}");
    let uniform_std = f64_from(&uniform.std(true));
    assert!(uniform_std > 0.15 && uniform_std < 0.35, "{}", "std {uniform_std}");
    let normal = vs.root().var("normal", &[100], Init::Randn { mean: 0., stdev: 0.02 });
    let normal_std = f64_from(&normal.std(true));
    assert!(normal_std <= 0.03, "{}", "std {normal_std}");
    let mut vs2 = VarStore::new(Device::Cpu);
    let ones = vs2.root().ones("t1", &[3]);
    assert_eq!(vec_f64_from(&ones), [1., 1., 1.]);
    vs2.copy(&vs).unwrap();
    assert_eq!(vec_f64_from(&ones), [0., 0., 0.]);
    let ortho = vs.root().var("orthogonal", &[100, 100], Init::Orthogonal { gain: 2.0 });
    let ortho_norm = f64_from(&ortho.linalg_norm_ord_str("fro", None::<i64>, true, Kind::Float));
    assert!(
        f64::abs(ortho_norm - 20.) < 1e-5,
        "{}",
        "ortho_norm initialization failed {ortho_norm}"
    );
    let ortho_shape_fail =
        tch::nn::f_init(Init::Orthogonal { gain: 1.0 }, &[10], Device::Cpu, tch::Kind::Float);
    assert!(ortho_shape_fail.is_err());
    let kaiming_u = vs.root().var("kaiming_u", &[20, 100], nn::init::DEFAULT_KAIMING_UNIFORM);
    assert!(f64::abs(f64_from(&kaiming_u.mean(Kind::Float))) < 5e-3);
    // The expected stdev is sqrt(2 / 100)
    assert!(f64::abs(f64_from(&kaiming_u.std(true)) - (0.02f64).sqrt()) < 2e-3);
    let kaiming_n = vs.root().var("kaiming_n", &[20, 100], nn::init::DEFAULT_KAIMING_NORMAL);
    assert!(f64::abs(f64_from(&kaiming_n.mean(Kind::Float))) < 5e-3);
    assert!(f64::abs(f64_from(&kaiming_n.std(true)) - (0.02f64).sqrt()) < 3e-3);
}

fn check_param_group(mut opt: tch::nn::Optimizer, var_foo: Tensor, var_bar: Tensor) {
    opt.set_lr(0.1);
    opt.set_lr_group(0, 0.);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(0.42f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.00");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.42");
    opt.set_lr_group(0, 0.1);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(0f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "-0.21");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.21");
    opt.set_lr_group(7, 0.);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(0.22f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.01");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.21");
    // The following sets the learning rate for both groups.
    opt.set_lr(0.);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(0.42f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.01");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.21");
    opt.set_lr(0.1);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(0.42f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.11");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.31");
}

#[test]
fn param_group() {
    tch::manual_seed(42);
    let vs = VarStore::new(Device::Cpu);
    let opt = tch::nn::Sgd::default().build(&vs, 1.0).unwrap();
    let root = vs.root();
    let var_foo = root.set_group(0).zeros("var_foo", &[]);
    let var_bar = root.set_group(7).zeros("var_bar", &[]);
    check_param_group(opt, var_foo, var_bar);
}

#[test]
fn save_and_load_with_group() {
    let filename = std::env::temp_dir().join(format!("tch-vs-load-grp-{}", std::process::id()));
    let add = |vs: &tch::nn::Path| {
        let v = vs.set_group(1).sub("a").sub("b").ones("t2", &[3]);
        let u = vs.zeros("t1", &[4]);
        let _w = vs.sub("a").sub("b").sub("ccc").ones("t123", &[3]);
        let _w = vs.sub("a").set_group(4).sub("b").sub("ccc").ones("t123", &[3]);
        (u, v)
    };
    let vs1 = VarStore::new(Device::Cpu);
    let mut vs2 = VarStore::new(Device::Cpu);
    let (mut u1, mut v1) = add(&vs1.root());
    let (u2, v2) = add(&vs2.root());
    tch::no_grad(|| {
        u1 += 42.0;
        v1 *= 2.0;
    });
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v1.mean(Kind::Float)), 2.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 0.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 1.0);
    vs1.save(&filename).unwrap();
    vs2.load(&filename).unwrap();
    assert_eq!(f64_from(&u1.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&u2.mean(Kind::Float)), 42.0);
    assert_eq!(f64_from(&v2.mean(Kind::Float)), 2.0);
    fs::remove_file(filename).unwrap();
}

#[test]
fn param_group_weight_decay() {
    tch::manual_seed(42);
    let vs = VarStore::new(Device::Cpu);
    let mut opt = tch::nn::Sgd::default().build(&vs, 0.0).unwrap();
    opt.set_lr(0.1);
    let root = vs.root();
    let var_foo = root.set_group(0).zeros("var_foo", &[]);
    let var_bar = root.set_group(7).zeros("var_bar", &[]);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(1f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.50");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.50");
    opt.set_weight_decay(0.1);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(1f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.49");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.49");
    opt.set_weight_decay_group(7, 0.);
    for _idx in 1..100 {
        let loss = (&var_foo + &var_bar).mse_loss(&Tensor::from(1f32), tch::Reduction::Mean);
        opt.backward_step(&loss);
    }
    assert_eq!(format!("{:.2}", f64_from(&var_foo)), "0.30");
    assert_eq!(format!("{:.2}", f64_from(&var_bar)), "0.69");
}

#[test]
fn half_precision_conversion_entire_varstore() {
    let mut vs = VarStore::new(Device::Cpu);

    let _ = vs.root().var("zeros", &[1], Init::Const(0.));
    let _ = vs.root().var("ones", &[1], Init::Const(1.));
    let _ = vs.root().var("forty_two", &[1], Init::Const(42.));

    assert_eq!(vs.root().get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().get("ones").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().get("forty_two").unwrap().kind(), Kind::Float);

    vs.half();

    assert_eq!(vs.root().get("zeros").unwrap().kind(), Kind::Half);
    assert_eq!(vs.root().get("ones").unwrap().kind(), Kind::Half);
    assert_eq!(vs.root().get("forty_two").unwrap().kind(), Kind::Half);

    vs.bfloat16();

    assert_eq!(vs.root().get("zeros").unwrap().kind(), Kind::BFloat16);
    assert_eq!(vs.root().get("ones").unwrap().kind(), Kind::BFloat16);
    assert_eq!(vs.root().get("forty_two").unwrap().kind(), Kind::BFloat16);

    vs.double();

    assert_eq!(vs.root().get("zeros").unwrap().kind(), Kind::Double);
    assert_eq!(vs.root().get("ones").unwrap().kind(), Kind::Double);
    assert_eq!(vs.root().get("forty_two").unwrap().kind(), Kind::Double);

    vs.float();

    assert_eq!(vs.root().get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().get("ones").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().get("forty_two").unwrap().kind(), Kind::Float);
    assert_eq!(format!("{:.2}", f64_from(&vs.root().get("zeros").unwrap())), "0.00");
    assert_eq!(format!("{:.2}", f64_from(&vs.root().get("ones").unwrap())), "1.00");
    assert_eq!(format!("{:.2}", f64_from(&vs.root().get("forty_two").unwrap())), "42.00");
}

#[test]
fn path_half_precision_conversion() {
    let vs = VarStore::new(Device::Cpu);

    // Define a VarStore with 3 variables. 2 of them are in a sub-path named "convert" and
    // will be cast to half-precision. The other variables in the VarStore will be unaffected
    let _ = vs.root().sub("ignore").var("zeros", &[1], Init::Const(0.));
    let _ = vs.root().sub("convert").sub("group_1").var("ones", &[1], Init::Const(1.));
    let linear_layer =
        linear(vs.root().sub("convert").sub("group_2").sub("linear"), 10, 42, Default::default());

    assert_eq!(vs.root().sub("ignore").get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_1").get("ones").unwrap().kind(), Kind::Float);
    assert_eq!(linear_layer.ws.kind(), Kind::Float);
    assert_eq!(linear_layer.bs.as_ref().unwrap().kind(), Kind::Float);

    vs.root().sub("convert").half();

    assert_eq!(vs.root().sub("ignore").get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_1").get("ones").unwrap().kind(), Kind::Half);
    assert_eq!(linear_layer.ws.kind(), Kind::Half);
    assert_eq!(linear_layer.bs.as_ref().unwrap().kind(), Kind::Half);

    vs.root().sub("convert").float();

    assert_eq!(vs.root().sub("ignore").get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_1").get("ones").unwrap().kind(), Kind::Float);
    assert_eq!(linear_layer.ws.kind(), Kind::Float);
    assert_eq!(linear_layer.bs.as_ref().unwrap().kind(), Kind::Float);
}

#[test]
fn path_free_type_conversion() {
    let mut vs = VarStore::new(Device::Cpu);

    // Define a VarStore with 3 variables. 2 of them are in a sub-path named "convert" and
    // will be cast to half-precision. The other variables in the VarStore will be unaffected
    let _ = vs.root().sub("ignore").var("zeros", &[1], Init::Const(0.));
    let _ = vs.root().sub("convert").sub("group_1").var("ones", &[1], Init::Const(1.));
    let _ = vs.root().sub("convert").sub("group_2").var("zeros", &[1], Init::Const(0.));

    assert_eq!(vs.root().sub("ignore").get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_1").get("ones").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_2").get("zeros").unwrap().kind(), Kind::Float);

    // Disable gradient tracking as this would raise an error when converting to a
    // non-float type like bool.
    vs.freeze();
    vs.root().sub("convert").set_kind(Kind::Bool);

    assert_eq!(vs.root().sub("ignore").get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_1").get("ones").unwrap().kind(), Kind::Bool);
    assert_eq!(vs.root().sub("convert").sub("group_2").get("zeros").unwrap().kind(), Kind::Bool);

    vs.root().sub("convert").set_kind(Kind::Float);
    vs.unfreeze();

    assert_eq!(vs.root().sub("ignore").get("zeros").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_1").get("ones").unwrap().kind(), Kind::Float);
    assert_eq!(vs.root().sub("convert").sub("group_2").get("zeros").unwrap().kind(), Kind::Float);

    assert_eq!(format!("{:.2}", f64_from(&vs.root().sub("ignore").get("zeros").unwrap())), "0.00");
    assert_eq!(
        format!("{:.2}", f64_from(&vs.root().sub("convert").sub("group_1").get("ones").unwrap())),
        "1.00"
    );
    assert_eq!(
        format!("{:.2}", f64_from(&vs.root().sub("convert").sub("group_2").get("zeros").unwrap())),
        "0.00"
    );
}

#[test]
fn device_migration() {
    if tch::Cuda::is_available() {
        let mut vs = VarStore::new(Device::Cpu);

        let _ = vs.root().var("zeros", &[3], Init::Const(0.));
        let _ = vs.root().var("ones", &[3], Init::Const(1.));
        let linear_layer = linear(vs.root().sub("linear"), 10, 42, Default::default());

        vs.set_device(Device::Cuda(0));

        assert_eq!(vs.root().get("zeros").unwrap().device(), Device::Cuda(0));
        assert_eq!(vs.root().get("ones").unwrap().device(), Device::Cuda(0));
        assert_eq!(linear_layer.ws.device(), Device::Cuda(0));
        assert_eq!(linear_layer.bs.as_ref().unwrap().device(), Device::Cuda(0));
        assert_eq!(vs.device(), Device::Cuda(0));

        vs.set_device(Device::Cpu);

        assert_eq!(vs.root().get("zeros").unwrap().device(), Device::Cpu);
        assert_eq!(vs.root().get("ones").unwrap().device(), Device::Cpu);
        assert_eq!(linear_layer.ws.device(), Device::Cpu);
        assert_eq!(linear_layer.bs.as_ref().unwrap().device(), Device::Cpu);
        assert_eq!(vs.device(), Device::Cpu);
    }
}

#[test]
fn merge_var_stores_no_prefixes() {
    let vs_1 = VarStore::new(Device::Cpu);
    let _ = vs_1.root().entry("key_1").or_zeros(&[3, 1, 4]);
    let _ = vs_1.root().entry("key_2").or_zeros(&[1, 5, 9]);

    let vs_2 = VarStore::new(Device::Cpu);
    let _ = vs_2.root().entry("key_3").or_zeros(&[2, 4, 4]);
    let _ = vs_2.root().entry("key_4").or_zeros(&[5, 2, 3]);

    let merged_vs = VarStore::merge(vec![(vs_1, None), (vs_2, None)]).unwrap();
    assert_eq!(merged_vs.variables().len(), 4);
    assert_eq!(merged_vs.trainable_variables().len(), 4);
    assert!(merged_vs.variables().contains_key("key_1"));
    assert!(merged_vs.variables().contains_key("key_2"));
    assert!(merged_vs.variables().contains_key("key_3"));
    assert!(merged_vs.variables().contains_key("key_4"));
}

#[test]
fn merge_var_stores_conflicting_keys() {
    let vs_1 = VarStore::new(Device::Cpu);
    let _ = vs_1.root().entry("duplicate_key_1").or_zeros(&[3, 1, 4]);
    let _ = vs_1.root().entry("key_2").or_zeros(&[1, 5, 9]);

    let vs_2 = VarStore::new(Device::Cpu);
    let _ = vs_2.root().entry("duplicate_key_1").or_zeros(&[2, 4, 4]);
    let _ = vs_2.root().entry("key_3").or_zeros(&[5, 2, 3]);

    let merged_vs = VarStore::merge(vec![(vs_1, None), (vs_2, None)]);
    assert!(matches!(merged_vs, Err(TchError::Torch(t)) if t==
            "Duplicate variable name found: duplicate_key_1. Provide a unique prefix to allow merge operation"));
}

#[test]
fn merge_var_stores_with_prefixes() {
    let vs_1 = VarStore::new(Device::Cpu);
    let _ = vs_1.root().entry("key_1").or_zeros(&[3, 1, 4]);
    let _ = vs_1.root().entry("key_2").or_zeros(&[1, 5, 9]);

    let vs_2 = VarStore::new(Device::Cpu);
    let _ = vs_2.root().entry("key_3").or_zeros(&[2, 4, 4]);
    let _ = vs_2.root().entry("key_4").or_zeros(&[5, 2, 3]);

    let merged_vs = VarStore::merge(vec![(vs_1, Some("vs_1.")), (vs_2, Some("vs_2."))]).unwrap();
    assert_eq!(merged_vs.variables().len(), 4);
    assert_eq!(merged_vs.trainable_variables().len(), 4);
    assert!(merged_vs.variables().contains_key("vs_1.key_1"));
    assert!(merged_vs.variables().contains_key("vs_1.key_2"));
    assert!(merged_vs.variables().contains_key("vs_2.key_3"));
    assert!(merged_vs.variables().contains_key("vs_2.key_4"));
}
