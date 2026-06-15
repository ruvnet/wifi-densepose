//! Honest-labeling source-presence tests (ADR-160).
//!
//! These tests assert that the claim-surface fixes A1–A4 are physically present
//! in the source (disclaimers added, uncited stats removed, overclaiming names
//! renamed). They are deliberately source-text assertions (`include_str!`),
//! mirroring ADR-159 §A5 / `cog-pose-estimation`'s `manifest_roundtrips` pattern:
//! the win here is making the *labels* true, which is a documentation invariant,
//! not a runtime capability. Each test is designed to FAIL on the pre-fix source.

// ── A1: medical modules carry the mandatory disclaimer + feature gate ─────────

const MED_SEIZURE: &str = include_str!("../src/med_seizure_detect.rs");
const MED_CARDIAC: &str = include_str!("../src/med_cardiac_arrhythmia.rs");
const MED_RESP: &str = include_str!("../src/med_respiratory_distress.rs");
const MED_APNEA: &str = include_str!("../src/med_sleep_apnea.rs");
const MED_GAIT: &str = include_str!("../src/med_gait_analysis.rs");

const MED_MODULES: &[(&str, &str)] = &[
    ("med_seizure_detect", MED_SEIZURE),
    ("med_cardiac_arrhythmia", MED_CARDIAC),
    ("med_respiratory_distress", MED_RESP),
    ("med_sleep_apnea", MED_APNEA),
    ("med_gait_analysis", MED_GAIT),
];


/// Char-boundary-safe prefix of up to `max` bytes (module headers are ASCII-ish
/// but contain box-drawing chars, so a naive byte slice can split a UTF-8 char).
fn char_safe_prefix(s: &str, max: usize) -> &str {
    let mut end = s.len().min(max);
    while end > 0 && !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}

/// A1(a): every med_* module's `//!` header must carry the mandatory disclaimer
/// stating it is experimental, not clinically validated, and not a medical device.
#[test]
fn a1_med_modules_have_clinical_disclaimer() {
    for (name, src) in MED_MODULES {
        // Search the whole module doc-comment region (first ~2KB) for robustness.
        let scan = char_safe_prefix(src, 2048);
        assert!(
            scan.contains("NOT VALIDATED AGAINST CLINICAL DATA"),
            "{name}: missing 'NOT VALIDATED AGAINST CLINICAL DATA' disclaimer"
        );
        assert!(
            scan.contains("NOT A MEDICAL DEVICE"),
            "{name}: missing 'NOT A MEDICAL DEVICE' disclaimer"
        );
        assert!(
            scan.contains("EXPERIMENTAL"),
            "{name}: missing 'EXPERIMENTAL' marker"
        );
        // ADR cross-reference so the disclaimer is traceable.
        assert!(
            scan.contains("ADR-160"),
            "{name}: disclaimer should cite ADR-160"
        );
    }
}

/// A1(c): all five med_* modules must be gated behind the non-default
/// `medical-experimental` cargo feature in lib.rs (cannot be silently shipped).
#[test]
fn a1_med_modules_gated_behind_medical_experimental() {
    const LIB: &str = include_str!("../src/lib.rs");
    for (name, _) in MED_MODULES {
        // Each module declaration must be immediately preceded by the cfg gate.
        let decl = format!("pub mod {name};");
        let idx = LIB
            .find(&decl)
            .unwrap_or_else(|| panic!("{name}: `{decl}` not found in lib.rs"));
        let preceding = &LIB[idx.saturating_sub(80)..idx];
        assert!(
            preceding.contains("#[cfg(feature = \"medical-experimental\")]"),
            "{name}: `{decl}` not gated behind medical-experimental in lib.rs"
        );
    }
    // The feature itself must exist in Cargo.toml.
    const CARGO: &str = include_str!("../Cargo.toml");
    assert!(
        CARGO.contains("medical-experimental = []"),
        "Cargo.toml missing `medical-experimental` feature definition"
    );
    // And it must NOT be in the default feature set.
    let default_line = CARGO
        .lines()
        .find(|l| l.trim_start().starts_with("default = ["))
        .expect("Cargo.toml missing default features");
    assert!(
        !default_line.contains("medical-experimental"),
        "medical-experimental must be NON-default; found in: {default_line}"
    );
}

/// A1(b): the seizure module must no longer assert detection as fact
/// ("Detects tonic-clonic seizures") and must use softened "candidate"/"flags"
/// language instead.
#[test]
fn a1_seizure_verbs_softened() {
    assert!(
        !MED_SEIZURE.contains("Detects tonic-clonic seizures"),
        "med_seizure_detect still asserts 'Detects tonic-clonic seizures' as fact"
    );
    assert!(
        MED_SEIZURE.contains("candidate") && MED_SEIZURE.contains("signature"),
        "med_seizure_detect should describe 'candidate ... signatures' (experimental)"
    );
}

// ── A2: affect modules carry the speculative/unvalidated disclaimer ───────────

const EXO_HAPPINESS: &str = include_str!("../src/exo_happiness_score.rs");
const EXO_EMOTION: &str = include_str!("../src/exo_emotion_detect.rs");

/// A2: both affect modules must declare outputs are NOT measurements of emotion
/// and cite ADR-160.
#[test]
fn a2_affect_modules_have_unvalidated_disclaimer() {
    for (name, src) in [("exo_happiness_score", EXO_HAPPINESS), ("exo_emotion_detect", EXO_EMOTION)] {
        let scan = char_safe_prefix(src, 2048);
        assert!(
            scan.contains("NOT measurements of emotion") || scan.contains("NOT a")
                && scan.contains("affect"),
            "{name}: missing 'NOT measurements of emotion' style disclaimer"
        );
        assert!(
            scan.to_lowercase().contains("speculative")
                || scan.to_lowercase().contains("unvalidated"),
            "{name}: missing speculative/unvalidated qualifier"
        );
        assert!(scan.contains("ADR-160"), "{name}: disclaimer should cite ADR-160");
    }
}

/// A2: the uncited "Happy people walk ~12% faster" statistic must be deleted.
#[test]
fn a2_uncited_12_percent_stat_removed() {
    assert!(
        !EXO_HAPPINESS.contains("12% faster"),
        "exo_happiness_score still contains the uncited '12% faster' claim"
    );
    assert!(
        !EXO_HAPPINESS.contains("~12% above"),
        "exo_happiness_score still contains the uncited '~12% above' claim"
    );
    assert!(
        !EXO_HAPPINESS.contains("Happy people walk"),
        "exo_happiness_score still contains the uncited 'Happy people walk' claim"
    );
}

/// A2: HAPPINESS_SCORE must be documented as a gait-energy proxy, not an affect
/// measurement.
#[test]
fn a2_happiness_reframed_as_proxy() {
    assert!(
        EXO_HAPPINESS.contains("gait-energy proxy"),
        "exo_happiness_score should document HAPPINESS_SCORE as a 'gait-energy proxy'"
    );
}

// ── A3: weapon-detect renamed to honest physical quantities ───────────────────

const SEC_WEAPON: &str = include_str!("../src/sec_weapon_detect.rs");
const LIB_RS: &str = include_str!("../src/lib.rs");

/// A3: the weapon-grade overclaim must be gone from the event/const names.
#[test]
fn a3_weapon_names_renamed_to_reflectivity() {
    // The module must no longer *define* a WEAPON_ALERT event or WEAPON_RATIO_THRESH
    // const. (A doc-comment may still reference the old name historically, e.g.
    // "formerly `EVENT_WEAPON_ALERT`" — we assert on the definitions, not mentions.)
    assert!(
        !SEC_WEAPON.contains("pub const EVENT_WEAPON_ALERT"),
        "sec_weapon_detect still defines/exports EVENT_WEAPON_ALERT"
    );
    assert!(
        !SEC_WEAPON.contains("const WEAPON_RATIO_THRESH"),
        "sec_weapon_detect still defines WEAPON_RATIO_THRESH"
    );
    // Honest replacements must be present.
    assert!(
        SEC_WEAPON.contains("EVENT_HIGH_METAL_REFLECTIVITY"),
        "sec_weapon_detect missing renamed EVENT_HIGH_METAL_REFLECTIVITY"
    );
    assert!(
        SEC_WEAPON.contains("HIGH_REFLECTIVITY_THRESH"),
        "sec_weapon_detect missing renamed HIGH_REFLECTIVITY_THRESH"
    );
}

/// A3: the lib.rs event registry must no longer export a `WEAPON_ALERT` name.
#[test]
fn a3_registry_no_longer_exports_weapon_alert() {
    assert!(
        !LIB_RS.contains("pub const WEAPON_ALERT"),
        "event_types registry still exports WEAPON_ALERT"
    );
    assert!(
        LIB_RS.contains("pub const HIGH_METAL_REFLECTIVITY"),
        "event_types registry missing HIGH_METAL_REFLECTIVITY (id 221)"
    );
}

// ── A4: quasi-medical / sign-language exotic modules carry the disclaimer ──────

const EXO_DREAM: &str = include_str!("../src/exo_dream_stage.rs");
const EXO_SIGN: &str = include_str!("../src/exo_gesture_language.rs");

/// A4: dream-stage and gesture-language modules promote the Exotic/Research tag
/// into a header disclaimer and state they are not validated.
#[test]
fn a4_exotic_modules_have_experimental_disclaimer() {
    for (name, src) in [("exo_dream_stage", EXO_DREAM), ("exo_gesture_language", EXO_SIGN)] {
        let scan = char_safe_prefix(src, 2048);
        assert!(
            scan.contains("EXPERIMENTAL") && scan.contains("NOT VALIDATED"),
            "{name}: missing EXPERIMENTAL / NOT VALIDATED disclaimer"
        );
        assert!(scan.contains("ADR-160"), "{name}: disclaimer should cite ADR-160");
        assert!(
            scan.contains("Research"),
            "{name}: should surface the Exotic/Research registry tag in the header"
        );
    }
}

// ── A5: the static_mut soundness fix is present (no per-call static mut bufs) ──

/// A5: claim-bearing modules must no longer use a `static mut` event scratch
/// buffer (latent aliasing UB). They now own a per-instance `events` field.
#[test]
fn a5_claim_bearing_modules_have_no_static_mut_event_buffer() {
    let modules: &[(&str, &str)] = &[
        ("med_seizure_detect", MED_SEIZURE),
        ("med_cardiac_arrhythmia", MED_CARDIAC),
        ("med_respiratory_distress", MED_RESP),
        ("med_sleep_apnea", MED_APNEA),
        ("med_gait_analysis", MED_GAIT),
        ("exo_happiness_score", EXO_HAPPINESS),
        ("exo_emotion_detect", EXO_EMOTION),
        ("sec_weapon_detect", SEC_WEAPON),
        ("exo_dream_stage", EXO_DREAM),
        ("exo_gesture_language", EXO_SIGN),
    ];
    for (name, src) in modules {
        assert!(
            !src.contains("static mut EVENTS")
                && !src.contains("static mut EV:")
                && !src.contains("static mut EMPTY"),
            "{name}: still uses a `static mut` event scratch buffer (A5 not applied)"
        );
        assert!(
            src.contains("events: [(i32, f32);"),
            "{name}: missing owned `events` scratch buffer field (A5)"
        );
    }
}
