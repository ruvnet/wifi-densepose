//! Confidence preserving reduction of high-rate observations into phases.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};
use crate::model::*;

#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub minimum_phase_ms: u64,
    pub maximum_phase_ms: u64,
    pub stable_velocity_deg_s: f32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            minimum_phase_ms: 200,
            maximum_phase_ms: 3_000,
            stable_velocity_deg_s: 8.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhaseSegmenter {
    config: SegmenterConfig,
}

impl PhaseSegmenter {
    pub fn new(config: SegmenterConfig) -> Result<Self> {
        if config.minimum_phase_ms == 0
            || config.maximum_phase_ms < config.minimum_phase_ms
            || !config.stable_velocity_deg_s.is_finite()
            || config.stable_velocity_deg_s < 0.0
        {
            return Err(Error::Observation("invalid segmenter configuration".into()));
        }
        Ok(Self { config })
    }

    pub fn segment(&self, name: impl Into<String>, frames: &[SceneFrame]) -> Result<Scene> {
        if frames.is_empty() {
            return Err(Error::Observation("at least one frame is required".into()));
        }
        for pair in frames.windows(2) {
            if pair[1].timestamp_ms <= pair[0].timestamp_ms {
                return Err(Error::Observation(
                    "timestamps must be strictly increasing".into(),
                ));
            }
        }
        let mut scene = Scene::new(name);
        scene.source = SceneSource::ObservedWifiCsi;
        scene.actors = actors_from_frames(frames);

        if frames.len() == 1 {
            return Ok(scene);
        }
        let mut start = 0_usize;
        let mut was_stable = true;
        for i in 1..frames.len() {
            let dt = frames[i].timestamp_ms - frames[i - 1].timestamp_ms;
            let velocity = angular_velocity(&frames[i - 1], &frames[i], dt);
            let stable = velocity <= self.config.stable_velocity_deg_s;
            let elapsed = frames[i].timestamp_ms - frames[start].timestamp_ms;
            let membership_changed = actor_ids(&frames[i - 1]) != actor_ids(&frames[i]);
            let boundary = membership_changed
                || elapsed >= self.config.maximum_phase_ms
                || (stable && !was_stable && elapsed >= self.config.minimum_phase_ms);
            if boundary {
                scene
                    .phases
                    .push(make_phase(scene.phases.len(), &frames[start], &frames[i]));
                start = i;
            }
            was_stable = stable;
        }
        let last = frames.len() - 1;
        if start < last {
            scene.phases.push(make_phase(
                scene.phases.len(),
                &frames[start],
                &frames[last],
            ));
        }
        Ok(scene)
    }
}

fn actors_from_frames(frames: &[SceneFrame]) -> Vec<Actor> {
    let mut found = BTreeMap::<u64, Actor>::new();
    for frame in frames {
        for observed in &frame.actors {
            found
                .entry(observed.track_id)
                .and_modify(|actor| {
                    actor.confidence = actor.confidence.max(observed.confidence);
                    actor.position = observed.position;
                })
                .or_insert_with(|| {
                    let mut actor = Actor::humanoid(observed.actor_id.clone());
                    actor.track_id = Some(observed.track_id);
                    actor.confidence = observed.confidence;
                    actor.position = observed.position;
                    actor
                });
        }
    }
    found.into_values().collect()
}

fn make_phase(index: usize, start: &SceneFrame, end: &SceneFrame) -> Phase {
    let actors = end
        .actors
        .iter()
        .map(|observed| ActorTargets {
            actor_id: observed.actor_id.clone(),
            joints: observed.joints.clone(),
            ground_lock: Vec::new(),
            travel: Some(observed.position),
            confidence: observed.confidence,
        })
        .collect();
    Phase {
        name: format!("Observed {}", index + 1),
        start_ms: start.timestamp_ms,
        duration_ms: (end.timestamp_ms - start.timestamp_ms).max(1),
        timing: Timing::Linear,
        actors,
        contacts: Vec::new(),
    }
}

fn actor_ids(frame: &SceneFrame) -> BTreeSet<u64> {
    frame.actors.iter().map(|a| a.track_id).collect()
}

fn angular_velocity(previous: &SceneFrame, current: &SceneFrame, dt_ms: u64) -> f32 {
    let mut sum = 0.0;
    let mut count = 0_u32;
    for a in &current.actors {
        let Some(before) = previous.actors.iter().find(|p| p.track_id == a.track_id) else {
            return f32::INFINITY;
        };
        for joint in &a.joints {
            if let Some(old) = before
                .joints
                .iter()
                .find(|j| j.joint == joint.joint && j.action == joint.action)
            {
                sum += (joint.degrees - old.degrees).abs() * joint.confidence.min(old.confidence);
                count += 1;
            }
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum / count as f32) / (dt_ms as f32 / 1000.0).max(0.001)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ms: u64, angle: f32) -> SceneFrame {
        SceneFrame {
            timestamp_ms: ms,
            actors: vec![ActorObservation {
                actor_id: "p1".into(),
                track_id: 1,
                confidence: 0.9,
                position: Vec3::default(),
                joints: vec![JointTarget {
                    joint: "knee_left".into(),
                    action: "flex".into(),
                    degrees: angle,
                    confidence: 0.9,
                }],
            }],
        }
    }

    #[test]
    fn creates_compact_observed_scene() {
        let frames = vec![
            frame(0, 0.0),
            frame(100, 20.0),
            frame(200, 40.0),
            frame(300, 40.1),
            frame(500, 40.1),
        ];
        let scene = PhaseSegmenter::new(SegmenterConfig::default())
            .unwrap()
            .segment("squat", &frames)
            .unwrap();
        assert_eq!(scene.actors.len(), 1);
        assert!(!scene.phases.is_empty());
        assert!(scene.phases.len() < frames.len());
        assert_eq!(scene.source, SceneSource::ObservedWifiCsi);
    }

    #[test]
    fn rejects_reversed_time() {
        let frames = vec![frame(100, 0.0), frame(50, 1.0)];
        assert!(PhaseSegmenter::new(SegmenterConfig::default())
            .unwrap()
            .segment("bad", &frames)
            .is_err());
    }
}
