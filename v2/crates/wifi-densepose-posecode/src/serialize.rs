//! Canonical deterministic PoseCode text serialization.

use std::fmt::Write;

use crate::model::*;

pub fn to_posecode(scene: &Scene) -> String {
    let mut out = String::new();
    writeln!(out, "posecode scene \"{}\"", safe_text(&scene.name)).unwrap();
    writeln!(out, "source {}", source_name(scene.source)).unwrap();
    for actor in &scene.actors {
        writeln!(out, "actor {}:", actor.id).unwrap();
        writeln!(out, "  rig {}", actor.rig).unwrap();
        writeln!(out, "  pose start = {}", actor.start_pose).unwrap();
        if let Some(track) = actor.track_id {
            writeln!(out, "  track {track}").unwrap();
        }
        writeln!(out, "  confidence {}", number(actor.confidence)).unwrap();
        writeln!(
            out,
            "  position {} {} {}",
            number(actor.position.x),
            number(actor.position.y),
            number(actor.position.z)
        )
        .unwrap();
    }
    for phase in &scene.phases {
        writeln!(
            out,
            "step \"{}\" {}s {}:",
            safe_text(&phase.name),
            number(phase.duration_ms as f32 / 1000.0),
            phase.timing.as_str()
        )
        .unwrap();
        for actor in &phase.actors {
            for joint in &actor.joints {
                writeln!(
                    out,
                    "  {}.{}: {} {} {}",
                    actor.actor_id,
                    joint.joint,
                    joint.action,
                    number(joint.degrees),
                    number(joint.confidence)
                )
                .unwrap();
            }
            if !actor.ground_lock.is_empty() {
                writeln!(
                    out,
                    "  {}.ground-lock: {}",
                    actor.actor_id,
                    actor.ground_lock.join(", ")
                )
                .unwrap();
            }
            if let Some(p) = actor.travel {
                writeln!(
                    out,
                    "  {}.travel: {} {} {}",
                    actor.actor_id,
                    number(p.x),
                    number(p.y),
                    number(p.z)
                )
                .unwrap();
            }
        }
        for contact in &phase.contacts {
            writeln!(
                out,
                "  contact {}.{} {}.{} {}",
                contact.from.actor_id,
                contact.from.effector,
                contact.to.actor_id,
                contact.to.effector,
                number(contact.confidence)
            )
            .unwrap();
        }
    }
    writeln!(out, "repeat {}", scene.repeat).unwrap();
    out
}

fn source_name(source: SceneSource) -> &'static str {
    match source {
        SceneSource::Authored => "authored",
        SceneSource::ObservedWifiCsi => "observed_wifi_csi",
        SceneSource::Imported => "imported",
    }
}

fn safe_text(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_control() && *c != '"')
        .collect()
}

fn number(value: f32) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let mut text = format!("{value:.4}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_posecode;

    #[test]
    fn canonical_text_round_trips() {
        let mut scene = Scene::new("two people");
        scene.actors.push(Actor::humanoid("p1"));
        scene.actors.push(Actor::humanoid("p2"));
        let text = to_posecode(&scene);
        assert_eq!(parse_posecode(&text).unwrap(), scene);
        assert_eq!(to_posecode(&parse_posecode(&text).unwrap()), text);
    }
}
