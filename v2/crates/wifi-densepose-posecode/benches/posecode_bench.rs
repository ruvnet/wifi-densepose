use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wifi_densepose_posecode::{parse_posecode, to_posecode};

const TWO_ACTORS: &str = r#"posecode scene "crossing"
source observed_wifi_csi
actor person_1:
  rig humanoid
  pose start = standing
  track 1
  confidence 0.8
  position 0 0 0
actor person_2:
  rig humanoid
  pose start = standing
  track 2
  confidence 0.8
  position 1 0 0
step "Move" 0.05s linear:
  person_1.knee_left: flex 30 0.8
  person_1.hip_left: flex 20 0.8
  person_2.knee_right: flex 35 0.8
  person_2.hip_right: flex 25 0.8
repeat 1
"#;

fn benchmarks(c: &mut Criterion) {
    c.bench_function("parse_two_actor_scene", |b| {
        b.iter(|| parse_posecode(black_box(TWO_ACTORS)).unwrap())
    });
    let scene = parse_posecode(TWO_ACTORS).unwrap();
    c.bench_function("serialize_two_actor_scene", |b| {
        b.iter(|| to_posecode(black_box(&scene)))
    });
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
