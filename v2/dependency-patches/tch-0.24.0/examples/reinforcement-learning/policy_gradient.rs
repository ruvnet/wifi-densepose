// Policy gradient example.
// This uses OpenAI Gym environment through rust-cpython.
//
// This is adapted from OpenAI Spinning Up series:
// https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
// A TensorFlow Python implementation can be found here:
// https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py

use super::gym_env::{GymEnv, Step};
use tch::{nn, nn::OptimizerConfig, Kind::Float, Tensor};

fn model(p: &nn::Path, input_shape: &[i64], nact: i64) -> impl nn::Module {
    let nin = input_shape.iter().product::<i64>();
    nn::seq()
        .add(nn::linear(p / "lin1", nin, 32, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(p / "lin2", 32, nact, Default::default()))
}

fn accumulate_rewards(steps: &[Step<i64>]) -> Vec<f64> {
    let mut rewards: Vec<f64> = steps.iter().map(|s| s.reward).collect();
    let mut acc_reward = 0f64;
    for (i, reward) in rewards.iter_mut().enumerate().rev() {
        if steps[i].is_done {
            acc_reward = 0.0;
        }
        acc_reward += *reward;
        *reward = acc_reward;
    }
    rewards
}

/// Trains an agent using the policy gradient algorithm.
pub fn run() -> cpython::PyResult<()> {
    let env = GymEnv::new("CartPole-v0")?;
    println!("action space: {:?}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let model = model(&vs.root(), env.observation_space(), env.action_space());
    let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    for epoch_idx in 0..50 {
        let mut obs = env.reset()?;
        let mut steps: Vec<Step<i64>> = vec![];
        // Perform some rollouts with the current model.
        loop {
            let action = tch::no_grad(|| {
                obs.unsqueeze(0).apply(&model).softmax(1, Float).multinomial(1, true)
            });
            let action = i64::try_from(action).unwrap();
            let step = env.step(action)?;
            steps.push(step.copy_with_obs(&obs));
            obs = if step.is_done { env.reset()? } else { step.obs };
            if step.is_done && steps.len() > 5000 {
                break;
            }
        }
        let sum_r: f64 = steps.iter().map(|s| s.reward).sum();
        let episodes: i64 = steps.iter().map(|s| s.is_done as i64).sum();
        println!(
            "epoch: {:<3} episodes: {:<5} avg reward per episode: {:.2}",
            epoch_idx,
            episodes,
            sum_r / episodes as f64
        );

        // Train the model via policy gradient on the rollout data.
        let batch_size = steps.len() as i64;
        let actions: Vec<i64> = steps.iter().map(|s| s.action).collect();
        let actions = Tensor::from_slice(&actions).unsqueeze(1);
        let rewards = accumulate_rewards(&steps);
        let rewards = Tensor::from_slice(&rewards).to_kind(Float);
        let action_mask =
            Tensor::zeros([batch_size, 2], tch::kind::FLOAT_CPU).scatter_value(1, &actions, 1.0);
        let obs: Vec<Tensor> = steps.into_iter().map(|s| s.obs).collect();
        let logits = Tensor::stack(&obs, 0).apply(&model);
        let log_probs =
            (action_mask * logits.log_softmax(1, Float)).sum_dim_intlist(1, false, Float);
        let loss = -(rewards * log_probs).mean(Float);
        opt.backward_step(&loss)
    }
    Ok(())
}
