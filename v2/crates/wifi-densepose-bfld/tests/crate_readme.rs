//! Validate the crate README. Same `include_str!` pattern iter-30/47/48 used
//! for HA blueprints / examples. crates.io renders this file, so doc drift
//! against the actual public API is operator-visible.

#![cfg(feature = "std")]

const README: &str = include_str!("../README.md");

#[test]
fn readme_documents_three_structural_invariants() {
    for needle in [
        "**I1**",
        "**I2**",
        "**I3**",
        "Raw BFI never exits the node",
        "Identity embedding is in-RAM-only",
        "Cross-site identity correlation",
    ] {
        assert!(README.contains(needle), "README missing invariant text: {needle}");
    }
}

#[test]
fn readme_documents_feature_flag_matrix() {
    for needle in ["`std`", "`serde-json`", "`mqtt`", "`soul-signature`"] {
        assert!(README.contains(needle), "feature flag {needle} missing from README");
    }
}

#[test]
fn readme_documents_both_runnable_examples() {
    assert!(README.contains("cargo run -p wifi-densepose-bfld --example bfld_minimal"));
    assert!(README.contains("cargo run -p wifi-densepose-bfld --example bfld_handle"));
}

#[test]
fn readme_documents_three_test_invocations() {
    assert!(README.contains("cargo test -p wifi-densepose-bfld --no-default-features"));
    assert!(README.contains("cargo test -p wifi-densepose-bfld --features mqtt"));
}

#[test]
fn readme_references_companion_adrs_118_through_123() {
    for adr in ["118", "119", "120", "121", "122", "123"] {
        assert!(README.contains(adr), "README must cite ADR-{adr}");
    }
}

#[test]
fn readme_quickstart_uses_canonical_public_api() {
    // The quickstart snippets must reference the actual operator-facing
    // surface — drift here would mislead first-time users.
    // Normalize line endings so the multi-line needle below is robust to a
    // CRLF checkout (Windows / `core.autocrlf=true`); the README renders
    // identically either way on crates.io.
    let readme = README.replace("\r\n", "\n");
    for needle in [
        "BfldPipeline::new",
        "BfldConfig::new",
        "SignatureHasher::new",
        "SensingInputs",
        "IdentityEmbedding::from_raw",
        "pipeline\n    .process",
        "publish_availability_online",
        "publish_discovery",
        "BfldPipelineHandle::spawn",
        "PipelineInput",
    ] {
        assert!(readme.contains(needle), "quickstart missing canonical API: {needle}");
    }
}

#[test]
fn readme_points_at_research_bundle_and_blueprints() {
    assert!(README.contains("docs/research/BFLD/"));
    assert!(README.contains("cog-ha-matter/blueprints/bfld/"));
    assert!(README.contains("bfld-mqtt-integration.yml"));
}

#[test]
fn readme_documents_env_gated_mosquitto_integration() {
    assert!(README.contains("BFLD_MQTT_BROKER=tcp://localhost:1883"));
    assert!(README.contains("mosquitto_integration"));
}
