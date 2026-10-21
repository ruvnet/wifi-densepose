mod a2c;
mod ddpg;
mod gym_env;
mod policy_gradient;
mod ppo;
mod vec_gym_env;

fn main() -> cpython::PyResult<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.iter().map(|x| x.as_str()).collect::<Vec<_>>().as_slice() {
        [_, "a2c"] => a2c::train()?,
        [_, "a2c-sample", weight_file] => a2c::sample(weight_file)?,
        [_, "pg"] => policy_gradient::run()?,
        [_, "ppo"] => ppo::train()?,
        [_, "ppo-sample", weight_file] => ppo::sample(weight_file)?,
        [_, "ddpg"] => ddpg::run()?,
        _ => println!("usage: main pg|a2c|a2c-sample|ppo|ppo-sample|ddpg"),
    }
    Ok(())
}
