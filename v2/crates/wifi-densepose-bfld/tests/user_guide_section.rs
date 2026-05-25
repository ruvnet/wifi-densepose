//! Validate the BFLD section in `docs/user-guide.md` per the project's
//! pre-merge checklist item #6 ("Update if new data sources, CLI flags, or
//! setup steps were added"). Test embeds the user-guide via include_str
//! and asserts the operator-facing surface is documented.

#![cfg(feature = "std")]

const USER_GUIDE: &str = include_str!("../../../../docs/user-guide.md");

#[test]
fn user_guide_documents_bfld_section_in_ha_chapter() {
    assert!(
        USER_GUIDE.contains("### BFLD — privacy-gated WiFi BFI sensing layer (ADR-118)"),
        "user-guide must carry a BFLD subsection under the HA chapter",
    );
}

#[test]
fn user_guide_bfld_section_names_three_structural_invariants() {
    assert!(USER_GUIDE.contains("**I1**"));
    assert!(USER_GUIDE.contains("**I2**"));
    assert!(USER_GUIDE.contains("**I3**"));
    assert!(USER_GUIDE.contains("Raw BFI never exits"));
    assert!(USER_GUIDE.contains("in-RAM-only"));
    assert!(USER_GUIDE.contains("cryptographically impossible"));
}

#[test]
fn user_guide_bfld_section_shows_both_runnable_examples() {
    assert!(USER_GUIDE.contains("cargo run -p wifi-densepose-bfld --example bfld_minimal"));
    assert!(USER_GUIDE.contains("cargo run -p wifi-densepose-bfld --example bfld_handle"));
}

#[test]
fn user_guide_bfld_section_documents_publish_lifecycle() {
    for needle in [
        "publish_availability_online",
        "publish_discovery",
        "BfldPipelineHandle::spawn",
        "handle.send",
    ] {
        assert!(USER_GUIDE.contains(needle), "user-guide missing {needle}");
    }
}

#[test]
fn user_guide_bfld_section_documents_four_privacy_classes() {
    for class in ["`Raw`", "`Derived`", "`Anonymous`", "`Restricted`"] {
        assert!(
            USER_GUIDE.contains(class),
            "user-guide must document the {class} privacy class",
        );
    }
}

#[test]
fn user_guide_bfld_section_lists_three_operator_blueprints() {
    for blueprint in ["presence-lighting", "motion-hvac", "identity-risk-anomaly"] {
        assert!(
            USER_GUIDE.contains(blueprint),
            "user-guide must mention HA blueprint {blueprint}",
        );
    }
}

#[test]
fn user_guide_bfld_section_documents_mqtt_topic_tree() {
    for topic in [
        "ruview/<node_id>/bfld/availability",
        "ruview/<node_id>/bfld/presence/state",
        "ruview/<node_id>/bfld/identity_risk/state",
    ] {
        assert!(USER_GUIDE.contains(topic), "user-guide missing topic {topic}");
    }
}

#[test]
fn user_guide_bfld_section_points_at_companion_artifacts() {
    assert!(
        USER_GUIDE.contains("v2/crates/wifi-densepose-bfld/README.md"),
        "user-guide must link to the crate README",
    );
    assert!(
        USER_GUIDE.contains("research/BFLD/"),
        "user-guide must link to the research dossier",
    );
}
