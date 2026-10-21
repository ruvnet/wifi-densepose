// Vectorized version of the gym environment.
use cpython::{buffer::PyBuffer, NoArgs, ObjectProtocol, PyObject, PyResult, Python};
use tch::Tensor;

#[derive(Debug)]
pub struct Step {
    pub obs: Tensor,
    pub reward: Tensor,
    pub is_done: Tensor,
}

pub struct VecGymEnv {
    env: PyObject,
    action_space: i64,
    observation_space: Vec<i64>,
}

impl VecGymEnv {
    pub fn new(name: &str, img_dir: Option<&str>, nprocesses: i64) -> PyResult<VecGymEnv> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let sys = py.import("sys")?;
        let path = sys.get(py, "path")?;
        let _ = path.call_method(py, "append", ("examples/reinforcement-learning",), None)?;
        let gym = py.import("atari_wrappers")?;
        let env = gym.call(py, "make", (name, img_dir, nprocesses), None)?;
        let action_space = env.getattr(py, "action_space")?;
        let action_space = action_space.getattr(py, "n")?.extract(py)?;
        let observation_space = env.getattr(py, "observation_space")?;
        let observation_space: Vec<i64> = observation_space.getattr(py, "shape")?.extract(py)?;
        let observation_space =
            [vec![nprocesses].as_slice(), observation_space.as_slice()].concat();
        Ok(VecGymEnv { env, action_space, observation_space })
    }

    pub fn reset(&self) -> PyResult<Tensor> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let obs = self.env.call_method(py, "reset", NoArgs, None)?;
        let obs = obs.call_method(py, "flatten", NoArgs, None)?;
        let obs = Tensor::from_slice(&obs.extract::<Vec<f32>>(py)?);
        Ok(obs.view_(&self.observation_space))
    }

    pub fn step(&self, action: Vec<i64>) -> PyResult<Step> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let step = self.env.call_method(py, "step", (action,), None)?;
        let obs = step.get_item(py, 0)?.call_method(py, "flatten", NoArgs, None)?;
        let obs_buffer = PyBuffer::get(py, &obs)?;
        let obs_vec: Vec<u8> = obs_buffer.to_vec(py)?;
        let obs =
            Tensor::from_slice(&obs_vec).view_(&self.observation_space).to_kind(tch::Kind::Float);
        let reward = Tensor::from_slice(&step.get_item(py, 1)?.extract::<Vec<f32>>(py)?);
        let is_done = Tensor::from_slice(&step.get_item(py, 2)?.extract::<Vec<f32>>(py)?);
        Ok(Step { obs, reward, is_done })
    }

    pub fn action_space(&self) -> i64 {
        self.action_space
    }

    pub fn observation_space(&self) -> &[i64] {
        &self.observation_space
    }
}
