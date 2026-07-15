//! Bounded line parser for the PoseCode 0.2 multi-actor extension.

use crate::error::{Error, Result};
use crate::model::*;
use crate::validate::{validate_scene, ValidationConfig, ValidationSeverity};

const MAX_INPUT_BYTES: usize = 1_048_576;
const MAX_LINE_BYTES: usize = 4096;

pub fn parse_posecode(source: &str) -> Result<Scene> {
    if source.len() > MAX_INPUT_BYTES {
        return Err(Error::parse(0, "document exceeds 1 MiB"));
    }
    let mut scene: Option<Scene> = None;
    let mut current_actor: Option<usize> = None;
    let mut current_phase: Option<usize> = None;
    let mut elapsed_ms = 0_u64;

    for (index, raw) in source.lines().enumerate() {
        let line_no = index + 1;
        if raw.len() > MAX_LINE_BYTES {
            return Err(Error::parse(line_no, "line exceeds 4096 bytes"));
        }
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        if scene.is_none() {
            let rest = line
                .strip_prefix("posecode scene ")
                .ok_or_else(|| Error::parse(line_no, "expected posecode scene header"))?;
            scene = Some(Scene::new(parse_quoted(rest, line_no)?));
            continue;
        }
        let doc = scene.as_mut().expect("initialized above");

        if let Some(value) = line.strip_prefix("source ") {
            doc.source = match value {
                "authored" => SceneSource::Authored,
                "observed_wifi_csi" => SceneSource::ObservedWifiCsi,
                "imported" => SceneSource::Imported,
                _ => return Err(Error::parse(line_no, "unknown source")),
            };
            continue;
        }
        if let Some(value) = line.strip_prefix("actor ") {
            let id = value
                .strip_suffix(':')
                .ok_or_else(|| Error::parse(line_no, "actor declaration must end with ':'"))?;
            doc.actors.push(Actor::humanoid(id.trim()));
            current_actor = Some(doc.actors.len() - 1);
            current_phase = None;
            continue;
        }
        if let Some(value) = line.strip_prefix("step ") {
            let (name, tail) = take_quoted(value, line_no)?;
            let fields: Vec<&str> = tail.trim_end_matches(':').split_whitespace().collect();
            if fields.len() != 2 {
                return Err(Error::parse(line_no, "step requires duration and timing"));
            }
            let duration_ms = parse_duration(fields[0], line_no)?;
            let timing = parse_timing(fields[1], line_no)?;
            doc.phases.push(Phase {
                name,
                start_ms: elapsed_ms,
                duration_ms,
                timing,
                actors: Vec::new(),
                contacts: Vec::new(),
            });
            elapsed_ms = elapsed_ms
                .checked_add(duration_ms)
                .ok_or_else(|| Error::parse(line_no, "timeline overflow"))?;
            current_phase = Some(doc.phases.len() - 1);
            current_actor = None;
            continue;
        }
        if let Some(value) = line.strip_prefix("repeat ") {
            doc.repeat = value
                .parse()
                .map_err(|_| Error::parse(line_no, "repeat must be an integer"))?;
            if doc.repeat == 0 || doc.repeat > 10_000 {
                return Err(Error::parse(line_no, "repeat must be 1 to 10000"));
            }
            continue;
        }

        if let Some(ai) = current_actor {
            parse_actor_field(&mut doc.actors[ai], line, line_no)?;
            continue;
        }
        if let Some(pi) = current_phase {
            parse_phase_field(&mut doc.phases[pi], line, line_no)?;
            continue;
        }
        return Err(Error::parse(line_no, "directive is outside actor or step"));
    }

    let scene = scene.ok_or_else(|| Error::parse(0, "empty document"))?;
    if let Some(issue) = validate_scene(&scene, &ValidationConfig::default())
        .into_iter()
        .find(|i| i.severity == ValidationSeverity::Error)
    {
        return Err(Error::Validation(format!(
            "{}: {}",
            issue.path, issue.message
        )));
    }
    Ok(scene)
}

fn parse_actor_field(actor: &mut Actor, line: &str, line_no: usize) -> Result<()> {
    if let Some(value) = line.strip_prefix("rig ") {
        actor.rig = value.trim().into();
        return Ok(());
    }
    if let Some(value) = line.strip_prefix("pose start = ") {
        actor.start_pose = value.trim().into();
        return Ok(());
    }
    if let Some(value) = line.strip_prefix("track ") {
        actor.track_id = Some(
            value
                .parse()
                .map_err(|_| Error::parse(line_no, "track must be an integer"))?,
        );
        return Ok(());
    }
    if let Some(value) = line.strip_prefix("confidence ") {
        actor.confidence = parse_confidence(value, line_no)?;
        return Ok(());
    }
    if let Some(value) = line.strip_prefix("position ") {
        actor.position = parse_vec3(value, line_no)?;
        return Ok(());
    }
    Err(Error::parse(line_no, "unknown actor directive"))
}

fn parse_phase_field(phase: &mut Phase, line: &str, line_no: usize) -> Result<()> {
    if let Some(value) = line.strip_prefix("contact ") {
        let fields: Vec<&str> = value.split_whitespace().collect();
        if !(fields.len() == 2 || fields.len() == 3) {
            return Err(Error::parse(
                line_no,
                "contact requires two effectors and optional confidence",
            ));
        }
        phase.contacts.push(Contact {
            from: parse_effector(fields[0], line_no)?,
            to: parse_effector(fields[1], line_no)?,
            confidence: if fields.len() == 3 {
                parse_confidence(fields[2], line_no)?
            } else {
                1.0
            },
        });
        return Ok(());
    }
    let (lhs, rhs) = line
        .split_once(':')
        .ok_or_else(|| Error::parse(line_no, "phase directive requires ':'"))?;
    let (actor_id, target) = lhs
        .trim()
        .split_once('.')
        .ok_or_else(|| Error::parse(line_no, "target must be actor qualified"))?;
    let actor = phase_actor_mut(phase, actor_id);
    match target {
        "ground-lock" => {
            actor.ground_lock = rhs
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        "travel" => {
            actor.travel = Some(parse_vec3_or_xz(rhs, line_no)?);
        }
        joint => {
            let fields: Vec<&str> = rhs.split_whitespace().collect();
            if !(fields.len() == 2 || fields.len() == 3) {
                return Err(Error::parse(
                    line_no,
                    "joint target requires action, degrees and optional confidence",
                ));
            }
            actor.joints.push(JointTarget {
                joint: joint.into(),
                action: fields[0].into(),
                degrees: fields[1]
                    .parse()
                    .map_err(|_| Error::parse(line_no, "degrees must be numeric"))?,
                confidence: if fields.len() == 3 {
                    parse_confidence(fields[2], line_no)?
                } else {
                    1.0
                },
            });
        }
    }
    Ok(())
}

fn phase_actor_mut<'a>(phase: &'a mut Phase, id: &str) -> &'a mut ActorTargets {
    if let Some(index) = phase.actors.iter().position(|a| a.actor_id == id) {
        return &mut phase.actors[index];
    }
    phase.actors.push(ActorTargets::new(id));
    phase.actors.last_mut().expect("just pushed")
}

fn parse_effector(value: &str, line: usize) -> Result<EffectorRef> {
    let (actor_id, effector) = value
        .split_once('.')
        .ok_or_else(|| Error::parse(line, "effector must be actor qualified"))?;
    Ok(EffectorRef {
        actor_id: actor_id.into(),
        effector: effector.into(),
    })
}

fn parse_vec3(value: &str, line: usize) -> Result<Vec3> {
    let f: Vec<&str> = value.split_whitespace().collect();
    if f.len() != 3 {
        return Err(Error::parse(line, "position requires x y z"));
    }
    Ok(Vec3::new(
        parse_f32(f[0], line)?,
        parse_f32(f[1], line)?,
        parse_f32(f[2], line)?,
    ))
}

fn parse_vec3_or_xz(value: &str, line: usize) -> Result<Vec3> {
    let f: Vec<&str> = value.split_whitespace().collect();
    match f.len() {
        2 => Ok(Vec3::new(
            parse_f32(f[0], line)?,
            0.0,
            parse_f32(f[1], line)?,
        )),
        3 => parse_vec3(value, line),
        _ => Err(Error::parse(line, "travel requires x z or x y z")),
    }
}

fn parse_f32(value: &str, line: usize) -> Result<f32> {
    let v: f32 = value
        .parse()
        .map_err(|_| Error::parse(line, "expected finite number"))?;
    if !v.is_finite() {
        return Err(Error::parse(line, "expected finite number"));
    }
    Ok(v)
}

fn parse_confidence(value: &str, line: usize) -> Result<f32> {
    let v = parse_f32(value, line)?;
    if !(0.0..=1.0).contains(&v) {
        return Err(Error::parse(line, "confidence must be in [0, 1]"));
    }
    Ok(v)
}

fn parse_duration(value: &str, line: usize) -> Result<u64> {
    let seconds = value
        .strip_suffix('s')
        .ok_or_else(|| Error::parse(line, "duration must end in s"))?;
    let seconds = parse_f32(seconds, line)?;
    if seconds <= 0.0 || seconds > 300.0 {
        return Err(Error::parse(line, "duration must be in (0, 300] seconds"));
    }
    Ok((seconds * 1000.0).round() as u64)
}

fn parse_timing(value: &str, line: usize) -> Result<Timing> {
    match value {
        "flow" => Ok(Timing::Flow),
        "settle" => Ok(Timing::Settle),
        "drive" => Ok(Timing::Drive),
        "snap" => Ok(Timing::Snap),
        "linear" => Ok(Timing::Linear),
        "ease-in" => Ok(Timing::EaseIn),
        "ease-out" => Ok(Timing::EaseOut),
        "ease-in-out" => Ok(Timing::EaseInOut),
        _ => Err(Error::parse(line, "unknown timing")),
    }
}

fn parse_quoted(value: &str, line: usize) -> Result<String> {
    let (quoted, tail) = take_quoted(value, line)?;
    if !tail.trim().is_empty() {
        return Err(Error::parse(line, "unexpected text after quoted name"));
    }
    Ok(quoted)
}

fn take_quoted(value: &str, line: usize) -> Result<(String, &str)> {
    let rest = value
        .strip_prefix('"')
        .ok_or_else(|| Error::parse(line, "expected quoted name"))?;
    let end = rest
        .find('"')
        .ok_or_else(|| Error::parse(line, "unterminated quoted name"))?;
    Ok((rest[..end].to_string(), &rest[end + 1..]))
}

fn strip_comment(line: &str) -> &str {
    let hash = line.find('#');
    let slash = line.find("//");
    match (hash, slash) {
        (Some(a), Some(b)) => &line[..a.min(b)],
        (Some(a), None) | (None, Some(a)) => &line[..a],
        (None, None) => line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENE: &str = r#"posecode scene "Assisted squat"
source observed_wifi_csi
actor patient:
  rig humanoid
  pose start = standing
  track 7
  confidence 0.82
  position 0 0 0
actor therapist:
  rig humanoid
  pose start = standing
  position 1.2 0 0
step "Lower" 1.5s flow:
  patient.knee_left: flex 95 0.8
  patient.ground-lock: feet
  therapist.shoulder_left: flex 30
  contact therapist.hand_left patient.shoulder_right 0.7
repeat 2
"#;

    #[test]
    fn parses_multiple_actors_and_contacts() {
        let scene = parse_posecode(SCENE).unwrap();
        assert_eq!(scene.actors.len(), 2);
        assert_eq!(scene.phases[0].actors.len(), 2);
        assert_eq!(scene.phases[0].contacts.len(), 1);
        assert_eq!(scene.repeat, 2);
    }

    #[test]
    fn rejects_unknown_actor_reference() {
        let bad = "posecode scene \"x\"\nactor p1:\n rig humanoid\nstep \"x\" 1s flow:\n ghost.knee_left: flex 20";
        assert!(parse_posecode(bad).is_err());
    }

    #[test]
    fn rejects_non_finite_number() {
        let bad = "posecode scene \"x\"\nactor p1:\n position NaN 0 0";
        assert!(parse_posecode(bad).is_err());
    }

    #[test]
    fn rejects_duplicate_actors() {
        let bad = "posecode scene \"x\"\nactor p1:\n rig humanoid\nactor p1:\n rig humanoid";
        assert!(parse_posecode(bad).is_err());
    }

    #[test]
    fn rejects_oversized_document_before_parsing() {
        let bad = "x".repeat(MAX_INPUT_BYTES + 1);
        assert!(parse_posecode(&bad).is_err());
    }
}
