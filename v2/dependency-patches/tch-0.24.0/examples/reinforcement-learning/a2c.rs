/* Advantage Actor Critic (A2C) model.
   A2C is a synchronous variant of Asynchronous the Advantage Actor Critic (A3C)
   model introduced by DeepMind in https://arxiv.org/abs/1602.01783

   See https://blog.openai.com/baselines-acktr-a2c/ for a reference
   python implementation.
*/
use super::vec_gym_env::VecGymEnv;
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::{nn, nn::OptimizerConfig, Kind::Float, Tensor};

const ENV_NAME: &str = "SpaceInvadersNoFrameskip-v4";
const NPROCS: i64 = 16;
const NSTEPS: i64 = 5;
const NSTACK: i64 = 4;
const UPDATES: i64 = 1000000;

type Model = Box<dyn Fn(&Tensor) -> (Tensor, Tensor)>;

fn model(p: &nn::Path, nact: i64) -> Model {
    let stride = |s| nn::ConvConfig { stride: s, ..Default::default() };
    let seq = nn::seq()
        .add(nn::conv2d(p / "c1", NSTACK, 32, 8, stride(4)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
        .add_fn(|xs| xs.relu().flat_view())
        .add(nn::linear(p / "l1", 3136, 512, Default::default()))
        .add_fn(|xs| xs.relu());
    let critic = nn::linear(p / "cl", 512, 1, Default::default());
    let actor = nn::linear(p / "al", 512, nact, Default::default());
    let device = p.device();
    Box::new(move |xs: &Tensor| {
        let xs = xs.to_device(device).apply(&seq);
        (xs.apply(&critic), xs.apply(&actor))
    })
}

#[derive(Debug)]
struct FrameStack {
    data: Tensor,
    nprocs: i64,
    nstack: i64,
}

impl FrameStack {
    fn new(nprocs: i64, nstack: i64) -> FrameStack {
        FrameStack { data: Tensor::zeros([nprocs, nstack, 84, 84], FLOAT_CPU), nprocs, nstack }
    }

    fn update<'a>(&'a mut self, img: &Tensor, masks: Option<&Tensor>) -> &'a Tensor {
        if let Some(masks) = masks {
            self.data *= masks.view([self.nprocs, 1, 1, 1])
        };
        let slice = |i| self.data.narrow(1, i, 1);
        for i in 1..self.nstack {
            slice(i - 1).copy_(&slice(i))
        }
        slice(self.nstack - 1).copy_(img);
        &self.data
    }
}

/// Trains an agent using A2C.
pub fn train() -> cpython::PyResult<()> {
    let env = VecGymEnv::new(ENV_NAME, None, NPROCS)?;
    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = model(&vs.root(), env.action_space());
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    let mut sum_rewards = Tensor::zeros([NPROCS], FLOAT_CPU);
    let mut total_rewards = 0f64;
    let mut total_episodes = 0f64;

    let mut frame_stack = FrameStack::new(NPROCS, NSTACK);
    let _ = frame_stack.update(&env.reset()?, None);
    let s_states = Tensor::zeros([NSTEPS + 1, NPROCS, NSTACK, 84, 84], FLOAT_CPU);
    for update_index in 0..UPDATES {
        s_states.get(0).copy_(&s_states.get(-1));
        let s_values = Tensor::zeros([NSTEPS, NPROCS], FLOAT_CPU);
        let s_rewards = Tensor::zeros([NSTEPS, NPROCS], FLOAT_CPU);
        let s_actions = Tensor::zeros([NSTEPS, NPROCS], INT64_CPU);
        let s_masks = Tensor::zeros([NSTEPS, NPROCS], FLOAT_CPU);
        for s in 0..NSTEPS {
            let (critic, actor) = tch::no_grad(|| model(&s_states.get(s)));
            let probs = actor.softmax(-1, Float);
            let actions = probs.multinomial(1, true).squeeze_dim(-1);
            let step = env.step(Vec::<i64>::try_from(&actions).unwrap())?;

            sum_rewards += &step.reward;
            total_rewards += f64::try_from((&sum_rewards * &step.is_done).sum(Float)).unwrap();
            total_episodes += f64::try_from(step.is_done.sum(Float)).unwrap();

            let masks = Tensor::from(1f32) - step.is_done;
            sum_rewards *= &masks;
            let obs = frame_stack.update(&step.obs, Some(&masks));
            s_actions.get(s).copy_(&actions);
            s_values.get(s).copy_(&critic.squeeze_dim(-1));
            s_states.get(s + 1).copy_(obs);
            s_rewards.get(s).copy_(&step.reward);
            s_masks.get(s).copy_(&masks);
        }
        let s_returns = {
            let r = Tensor::zeros([NSTEPS + 1, NPROCS], FLOAT_CPU);
            let critic = tch::no_grad(|| model(&s_states.get(-1)).0);
            r.get(-1).copy_(&critic.view([NPROCS]));
            for s in (0..NSTEPS).rev() {
                let r_s = s_rewards.get(s) + r.get(s + 1) * s_masks.get(s) * 0.99;
                r.get(s).copy_(&r_s);
            }
            r
        };
        let (critic, actor) =
            model(&s_states.narrow(0, 0, NSTEPS).view([NSTEPS * NPROCS, NSTACK, 84, 84]));
        let critic = critic.view([NSTEPS, NPROCS]);
        let actor = actor.view([NSTEPS, NPROCS, -1]);
        let log_probs = actor.log_softmax(-1, Float);
        let probs = actor.softmax(-1, Float);
        let action_log_probs = {
            let index = s_actions.unsqueeze(-1).to_device(device);
            log_probs.gather(2, &index, false).squeeze_dim(-1)
        };
        let dist_entropy = (-log_probs * probs).sum_dim_intlist(-1, false, Float).mean(Float);
        let advantages = s_returns.narrow(0, 0, NSTEPS).to_device(device) - critic;
        let value_loss = (&advantages * &advantages).mean(Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        opt.backward_step_clip(&loss, 0.5);
        if update_index > 0 && update_index % 500 == 0 {
            println!("{} {:.0} {}", update_index, total_episodes, total_rewards / total_episodes);
            total_rewards = 0.;
            total_episodes = 0.;
        }
        if update_index > 0 && update_index % 10000 == 0 {
            if let Err(err) = vs.save(format!("a2c{update_index}.ot")) {
                println!("error while saving {err}")
            }
        }
    }
    Ok(())
}

/// Samples some trajectory for a trained agent, the observation frames are stored in /dev/shm.
pub fn sample<T: AsRef<std::path::Path>>(weight_file: T) -> cpython::PyResult<()> {
    let env = VecGymEnv::new(ENV_NAME, Some("/dev/shm"), 1)?;
    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let mut vs = nn::VarStore::new(tch::Device::Cpu);
    let model = model(&vs.root(), env.action_space());
    vs.load(weight_file).unwrap();

    let mut frame_stack = FrameStack::new(1, NSTACK);
    let mut obs = frame_stack.update(&env.reset()?, None);

    for _index in 0..5000 {
        let (_critic, actor) = tch::no_grad(|| model(obs));
        let probs = actor.softmax(-1, Float);
        let actions = probs.multinomial(1, true).squeeze_dim(-1);
        let step = env.step(Vec::<i64>::try_from(&actions).unwrap())?;

        let masks = Tensor::from(1f32) - step.is_done;
        obs = frame_stack.update(&step.obs, Some(&masks));
    }
    Ok(())
}
