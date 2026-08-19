#![allow(missing_docs)]

mod common;

use common::observation;
use std::path::PathBuf;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

#[test]
fn checked_in_schemas_parse_and_cover_serialized_contract_fields() {
    let raw = observation(1);
    let result = PhysicsEngine::new(PhysicsConfig::default())
        .unwrap()
        .process(&raw, raw.timestamp_ns);
    assert_schema_required_fields("pose-observation-v2.schema.json", &raw);
    assert_schema_required_fields("pose-refinement-v1.schema.json", &result);
}

fn assert_schema_required_fields(name: &str, value: &impl serde::Serialize) {
    let schema_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/schemas")
        .join(name);
    let schema: serde_json::Value =
        serde_json::from_slice(&std::fs::read(schema_path).unwrap()).unwrap();
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(schema["additionalProperties"], false);
    let serialized = serde_json::to_value(value).unwrap();
    for required in schema["required"].as_array().unwrap() {
        let field = required.as_str().unwrap();
        assert!(
            serialized.get(field).is_some(),
            "schema requires missing field {field}"
        );
    }
}
