//! Renderer independent scene model.

use serde::{Deserialize, Serialize};

/// Hard limits keep untrusted scene documents bounded in memory and time.
pub const MAX_ACTORS: usize = 32;
pub const MAX_PHASES: usize = 10_000;
pub const MAX_TARGETS_PER_ACTOR: usize = 64;
pub const MAX_NAME_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct EulerDeg {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl EulerDeg {
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneSource {
    Authored,
    ObservedWifiCsi,
    Imported,
}

impl Default for SceneSource {
    fn default() -> Self {
        Self::Authored
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Actor {
    pub id: String,
    pub rig: String,
    pub start_pose: String,
    pub track_id: Option<u64>,
    pub confidence: f32,
    pub position: Vec3,
}

impl Actor {
    pub fn humanoid(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            rig: "humanoid".into(),
            start_pose: "standing".into(),
            track_id: None,
            confidence: 1.0,
            position: Vec3::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointTarget {
    pub joint: String,
    pub action: String,
    pub degrees: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorTargets {
    pub actor_id: String,
    pub joints: Vec<JointTarget>,
    pub ground_lock: Vec<String>,
    pub travel: Option<Vec3>,
    pub confidence: f32,
}

impl ActorTargets {
    pub fn new(actor_id: impl Into<String>) -> Self {
        Self {
            actor_id: actor_id.into(),
            joints: Vec::new(),
            ground_lock: Vec::new(),
            travel: None,
            confidence: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectorRef {
    pub actor_id: String,
    pub effector: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    pub from: EffectorRef,
    pub to: EffectorRef,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Timing {
    Flow,
    Settle,
    Drive,
    Snap,
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

impl Timing {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flow => "flow",
            Self::Settle => "settle",
            Self::Drive => "drive",
            Self::Snap => "snap",
            Self::Linear => "linear",
            Self::EaseIn => "ease-in",
            Self::EaseOut => "ease-out",
            Self::EaseInOut => "ease-in-out",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub start_ms: u64,
    pub duration_ms: u64,
    pub timing: Timing,
    pub actors: Vec<ActorTargets>,
    pub contacts: Vec<Contact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub version: String,
    pub name: String,
    pub source: SceneSource,
    pub actors: Vec<Actor>,
    pub phases: Vec<Phase>,
    pub repeat: u32,
}

impl Scene {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: crate::PROTOCOL_VERSION.into(),
            name: name.into(),
            source: SceneSource::Authored,
            actors: Vec::new(),
            phases: Vec::new(),
            repeat: 1,
        }
    }
}

/// One actor at one observed instant. Raw observations remain separate from
/// semantic phases so 20 Hz input does not become unreadable scene text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActorObservation {
    pub actor_id: String,
    pub track_id: u64,
    pub confidence: f32,
    pub position: Vec3,
    pub joints: Vec<JointTarget>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFrame {
    pub timestamp_ms: u64,
    pub actors: Vec<ActorObservation>,
}
