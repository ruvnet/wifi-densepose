pub mod actor;
pub mod learning;
pub mod observation;
pub mod reward;
pub mod role_attention;
pub mod trainer;
pub mod training_loop;

pub use actor::{MappoActor, ActorConfig, ActorAction};
pub use learning::{LearningPattern, CuriosityModule, MetaAdapter, shaped_reward};
pub use observation::LocalObservation;
pub use reward::{RewardCalculator, RewardContext};
pub use role_attention::{NodeRole, RoleAttention, triangulation_geometry_penalty};
pub use trainer::{TrainingConfig, TrainingMode, DomainRandomizationConfig};
pub use training_loop::{ReplayBuffer, Transition, PpoConfig, UpdateStats, ppo_update};

#[cfg(feature = "train")]
pub mod candle_ppo;
#[cfg(feature = "train")]
pub use candle_ppo::{CandleActorCritic, CandlePpoConfig, CandleTrainer, select_device};
