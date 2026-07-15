//! Deterministic scene validation with provenance-aware range policy.

use std::collections::HashSet;

use crate::model::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub max_actors: usize,
    pub max_phases: usize,
    /// Authored values outside general ROM are errors. Observed values are
    /// warnings because RF estimates are not medical measurements.
    pub authored_rom_is_error: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_actors: MAX_ACTORS,
            max_phases: MAX_PHASES,
            authored_rom_is_error: true,
        }
    }
}

pub fn validate_scene(scene: &Scene, config: &ValidationConfig) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if scene.name.is_empty() || scene.name.len() > MAX_NAME_BYTES {
        error(
            &mut issues,
            "scene.name",
            "name must contain 1 to 128 bytes",
        );
    }
    if scene.actors.len() > config.max_actors {
        error(&mut issues, "scene.actors", "actor limit exceeded");
    }
    if scene.phases.len() > config.max_phases {
        error(&mut issues, "scene.phases", "phase limit exceeded");
    }

    let mut ids = HashSet::new();
    for (i, actor) in scene.actors.iter().enumerate() {
        let path = format!("actors[{i}]");
        if !valid_id(&actor.id) {
            error(&mut issues, &path, "invalid actor identifier");
        }
        if !ids.insert(actor.id.as_str()) {
            error(&mut issues, &path, "duplicate actor identifier");
        }
        check_confidence(&mut issues, &format!("{path}.confidence"), actor.confidence);
        if !actor.position.is_finite() {
            error(
                &mut issues,
                &format!("{path}.position"),
                "position must be finite",
            );
        }
    }

    let actor_ids: HashSet<&str> = scene.actors.iter().map(|a| a.id.as_str()).collect();
    for (pi, phase) in scene.phases.iter().enumerate() {
        let path = format!("phases[{pi}]");
        if phase.duration_ms == 0 || phase.duration_ms > 300_000 {
            error(
                &mut issues,
                &format!("{path}.duration_ms"),
                "duration must be 1 to 300000 ms",
            );
        }
        let mut phase_actors = HashSet::new();
        for targets in &phase.actors {
            if !actor_ids.contains(targets.actor_id.as_str()) {
                error(&mut issues, &path, "phase references unknown actor");
            }
            if !phase_actors.insert(targets.actor_id.as_str()) {
                error(&mut issues, &path, "actor appears more than once in phase");
            }
            if targets.joints.len() > MAX_TARGETS_PER_ACTOR {
                error(&mut issues, &path, "joint target limit exceeded");
            }
            check_confidence(&mut issues, &path, targets.confidence);
            for joint in &targets.joints {
                check_confidence(&mut issues, &path, joint.confidence);
                if !joint.degrees.is_finite() {
                    error(&mut issues, &path, "joint angle must be finite");
                    continue;
                }
                if let Some((min, max)) = range(&joint.joint, &joint.action) {
                    if joint.degrees < min || joint.degrees > max {
                        let severity = if scene.source == SceneSource::Authored
                            && config.authored_rom_is_error
                        {
                            ValidationSeverity::Error
                        } else {
                            ValidationSeverity::Warning
                        };
                        issues.push(ValidationIssue {
                            severity,
                            path: path.clone(),
                            message: format!(
                                "{}.{} angle {} outside general range [{}, {}]",
                                joint.joint, joint.action, joint.degrees, min, max
                            ),
                        });
                    }
                }
            }
        }
        for contact in &phase.contacts {
            if !actor_ids.contains(contact.from.actor_id.as_str())
                || !actor_ids.contains(contact.to.actor_id.as_str())
            {
                error(&mut issues, &path, "contact references unknown actor");
            }
            check_confidence(&mut issues, &path, contact.confidence);
        }
    }
    issues
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .enumerate()
            .all(|(i, b)| b.is_ascii_alphanumeric() || b == b'_' || (i > 0 && b == b'-'))
}

fn check_confidence(issues: &mut Vec<ValidationIssue>, path: &str, value: f32) {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        error(issues, path, "confidence must be finite and in [0, 1]");
    }
}

fn error(issues: &mut Vec<ValidationIssue>, path: &str, message: &str) {
    issues.push(ValidationIssue {
        severity: ValidationSeverity::Error,
        path: path.into(),
        message: message.into(),
    });
}

fn range(joint: &str, action: &str) -> Option<(f32, f32)> {
    match (
        joint.trim_end_matches("_left").trim_end_matches("_right"),
        action,
    ) {
        ("shoulder", "flex") => Some((0.0, 180.0)),
        ("shoulder", "extend") => Some((0.0, 60.0)),
        ("elbow", "flex") => Some((0.0, 154.0)),
        ("hip", "flex") => Some((0.0, 135.0)),
        ("hip", "extend") => Some((0.0, 20.0)),
        ("knee", "flex") => Some((0.0, 144.0)),
        ("pelvis", "hinge") => Some((0.0, 120.0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_rom_is_warning_not_medical_error() {
        let mut scene = Scene::new("observed");
        scene.source = SceneSource::ObservedWifiCsi;
        scene.actors.push(Actor::humanoid("p1"));
        let mut targets = ActorTargets::new("p1");
        targets.joints.push(JointTarget {
            joint: "knee_left".into(),
            action: "flex".into(),
            degrees: 170.0,
            confidence: 0.5,
        });
        scene.phases.push(Phase {
            name: "frame".into(),
            start_ms: 0,
            duration_ms: 50,
            timing: Timing::Linear,
            actors: vec![targets],
            contacts: vec![],
        });
        let issues = validate_scene(&scene, &ValidationConfig::default());
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, ValidationSeverity::Warning);
    }
}
