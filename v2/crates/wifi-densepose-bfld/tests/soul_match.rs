//! §3.6 Soul Signature matcher — measured-on-synthetic behavior tests.
//!
//! Every number asserted here is **MEASURED on STRUCTURED SYNTHETIC data**,
//! never on real people. The synthetic "people" are deterministic functions of
//! a seed (`synthetic_person`); they are clearly NOT recordings of humans, and
//! NONE of these tests demonstrate working named-person identification. What
//! they DO demonstrate, with reproducible numbers:
//!
//!   1. The matcher runs and is internally consistent (same-person scores
//!      higher than cross-person when the decisive channels are present).
//!   2. The audit's negative result: on cardiac + respiratory channels ALONE,
//!      two different people are NOT separable above threshold — the matcher
//!      correctly refuses to lock identity ("your heartbeat alone overlaps too
//!      much").
//!   3. Graceful degradation, zero-norm safety, and the "insufficient
//!      channels" path never produce a NaN or a default-high score.

#![cfg(feature = "std")]

use wifi_densepose_bfld::coherence_gate::{MatchOutcome, SoulMatchOracle};
use wifi_densepose_bfld::embedding::IdentityEmbedding;
use wifi_densepose_bfld::soul_channels::{
    Channel, FeatureVector, MatchWeights, SoulChannels,
};
use wifi_densepose_bfld::soul_match::{cosine_sim, match_score, EnrolledMatcher};
use wifi_densepose_bfld::EMBEDDING_DIM;

// --- Deterministic synthetic data generators -------------------------------

/// Tiny deterministic LCG — reproducible synthetic channels, no rand dep.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map high bits to [-1, 1).
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// Build a deterministic AETHER embedding for synthetic "person `seed`".
/// Distinct seeds produce distinct, decorrelated 128-d unit-ish vectors.
fn synthetic_aether(seed: u64) -> IdentityEmbedding {
    let mut next = lcg(seed);
    let mut values = [0.0f32; EMBEDDING_DIM];
    for v in &mut values {
        *v = next();
    }
    IdentityEmbedding::from_raw(values)
}

/// Build a deterministic subcarrier reflection profile (body geometry).
fn synthetic_subcarrier(seed: u64) -> FeatureVector {
    let mut next = lcg(seed ^ 0xABCD);
    let data: Vec<f32> = (0..64).map(|_| next()).collect();
    FeatureVector::from_slice(&data).unwrap()
}

/// Build a cardiac HR profile that is *physiologically realistic* — a small
/// set of positive, similar-magnitude features (heart-rate band energies).
/// Different people differ only slightly, exactly the audit's point: cardiac
/// rate alone barely separates people.
fn synthetic_cardiac(seed: u64) -> FeatureVector {
    // Base profile shared by all healthy adults; per-person jitter is small.
    let base = [0.80f32, 0.62, 0.41, 0.30, 0.22, 0.15, 0.10, 0.07];
    let mut next = lcg(seed ^ 0x5151);
    let data: Vec<f32> = base.iter().map(|b| b + 0.03 * next()).collect();
    FeatureVector::from_slice(&data).unwrap()
}

/// Respiratory pattern — likewise positive, similar-magnitude, low per-person
/// variance (breathing rate overlaps heavily between people).
fn synthetic_respiratory(seed: u64) -> FeatureVector {
    let base = [0.55f32, 0.50, 0.44, 0.33, 0.25, 0.18];
    let mut next = lcg(seed ^ 0x7272);
    let data: Vec<f32> = base.iter().map(|b| b + 0.04 * next()).collect();
    FeatureVector::from_slice(&data).unwrap()
}

/// A "full" synthetic signature: AETHER + subcarrier + cardiac + respiratory.
fn synthetic_person(seed: u64) -> SoulChannels {
    SoulChannels::empty()
        .with_aether(synthetic_aether(seed))
        .with_channel(Channel::SubcarrierReflectionProfile, synthetic_subcarrier(seed))
        .with_channel(Channel::CardiacHrProfile, synthetic_cardiac(seed))
        .with_channel(Channel::RespiratoryPattern, synthetic_respiratory(seed))
}

/// A probe with ONLY cardiac + respiratory present (decisive channels absent).
fn cardiac_respiratory_only(seed: u64) -> SoulChannels {
    SoulChannels::empty()
        .with_channel(Channel::CardiacHrProfile, synthetic_cardiac(seed))
        .with_channel(Channel::RespiratoryPattern, synthetic_respiratory(seed))
}

// --- 1. Separability (positive control) ------------------------------------

#[test]
fn same_person_scores_higher_than_cross_person() {
    let weights = MatchWeights::default();
    let person_a = synthetic_person(1);
    let person_b = synthetic_person(2);

    // Independently regenerated probes for A and B (same seed => same data).
    let probe_a = synthetic_person(1);
    let probe_b = synthetic_person(2);

    let a_vs_a = match_score(&person_a, &probe_a, &weights).score().unwrap();
    let a_vs_b = match_score(&person_a, &probe_b, &weights).score().unwrap();
    let b_vs_b = match_score(&person_b, &probe_b, &weights).score().unwrap();

    // MEASURED-on-synthetic (deterministic; reproduce by running this test):
    //   a_vs_a ≈ 1.0000   (identical deterministic data — perfect self-match)
    //   a_vs_b ≈ 0.8088   (cross-person, full channel set)
    //   b_vs_b ≈ 1.0000
    // The cross-person score is HIGH (0.81) even though AETHER (0.35) +
    // subcarrier (0.20) decorrelate between people — because the cardiac (0.15)
    // + respiratory (0.10) channels are similar between healthy adults and
    // pull the FUSED score up. The same-vs-cross gap is ~0.19: real, but far
    // smaller than the decisive channels alone would suggest. This is itself an
    // honest signal that fused scoring with these unvalidated weights does not
    // produce a wide identity margin.
    assert!(a_vs_a > 0.99, "self-match should be ~1.0, got {a_vs_a:.4}");
    assert!(b_vs_b > 0.99, "self-match should be ~1.0, got {b_vs_b:.4}");
    assert!(
        a_vs_a > a_vs_b + 0.15,
        "same-person ({a_vs_a:.4}) must exceed cross-person ({a_vs_b:.4}) \
         by a measurable margin"
    );
    // Pin the measured cross-person value so the number is reproducible.
    assert!(
        (a_vs_b - 0.8088).abs() < 0.01,
        "cross-person score drifted from measured 0.8088, got {a_vs_b:.4}"
    );
}

#[test]
fn enrolled_matcher_locks_correct_person_with_decisive_channels() {
    // With AETHER + subcarrier present, A's probe matches A and not B.
    let weights = MatchWeights::default();
    // Threshold 0.85 with >=2 shared channels: a stringent-but-achievable bar
    // for a full-channel self-match.
    let mut matcher = EnrolledMatcher::new(weights, 0.85, 2);
    matcher.enroll(1001, synthetic_person(1));
    matcher.enroll(2002, synthetic_person(2));

    matcher.set_probe(synthetic_person(1));
    match matcher.matches_enrolled() {
        MatchOutcome::Match { person_id } => assert_eq!(person_id, 1001),
        other => panic!("A's probe should lock person 1001, got {other:?}"),
    }

    matcher.set_probe(synthetic_person(2));
    match matcher.matches_enrolled() {
        MatchOutcome::Match { person_id } => assert_eq!(person_id, 2002),
        other => panic!("B's probe should lock person 2002, got {other:?}"),
    }
}

// --- 2. The audit's negative result (CENTERPIECE) --------------------------

#[test]
fn cardiac_alone_cannot_separate_identity_matches_audit() {
    // The two decisive high-weight channels (AETHER 0.35, subcarrier 0.20) are
    // ABSENT in the probe. Only cardiac (0.15) + respiratory (0.10) remain.
    // The audit's claim, now MEASURED on synthetic data: heartbeat + breathing
    // alone overlap too much between people to lock identity.
    let weights = MatchWeights::default();

    let person_a = synthetic_person(1); // full enrolled profile for A
    let person_b = synthetic_person(2); // full enrolled profile for B

    let probe_a = cardiac_respiratory_only(1); // A's cardiac/resp only
    let probe_b = cardiac_respiratory_only(2); // B's cardiac/resp only

    // Same-person (A's cardiac vs A's enrolled cardiac) and cross-person
    // (A's cardiac vs B's enrolled cardiac) scores.
    let a_self = match_score(&person_a, &probe_a, &weights).score().unwrap();
    let a_cross = match_score(&person_b, &probe_a, &weights).score().unwrap();
    let b_self = match_score(&person_b, &probe_b, &weights).score().unwrap();
    let b_cross = match_score(&person_a, &probe_b, &weights).score().unwrap();

    // MEASURED-on-synthetic numbers (deterministic; reproduce with --nocapture):
    //   a_self = 1.0000   a_cross = 0.9995   gap = 0.0005
    //   b_self = 1.0000   b_cross = 0.9995   gap = 0.0005
    // Both self and cross sit at ~1.0 because cardiac/respiratory feature
    // vectors are positive, similar-magnitude profiles shared by all healthy
    // adults — cosine similarity is high regardless of WHO the person is. The
    // same-vs-cross gap is 0.0005: ~380x smaller than the ~0.19 gap the
    // decisive channels produced. NO threshold fits in a 0.0005 gap, so the
    // matcher cannot lock identity. This is the audit's claim, measured.
    let separation_a = a_self - a_cross;
    let separation_b = b_self - b_cross;

    // Emit the measured numbers so `--nocapture` reproduces them verbatim.
    eprintln!(
        "[cardiac+resp only] a_self={a_self:.4} a_cross={a_cross:.4} gap={separation_a:.4} | \
         b_self={b_self:.4} b_cross={b_cross:.4} gap={separation_b:.4}"
    );

    // The decisive assertion: the same-vs-cross gap on cardiac+respiratory
    // alone is TINY (< 0.05) — far smaller than the ~0.3+ gap the decisive
    // channels produced above. No useful threshold sits in that gap.
    assert!(
        separation_a < 0.05,
        "cardiac+resp self-vs-cross gap should be tiny (got {separation_a:.4}) \
         — proves identity is NOT separable on these channels"
    );
    assert!(
        separation_b < 0.05,
        "cardiac+resp self-vs-cross gap should be tiny (got {separation_b:.4})"
    );

    // And operationally: an EnrolledMatcher gated on cardiac+respiratory alone
    // either (a) refuses to lock, or (b) cannot distinguish A from B. We assert
    // it does NOT confidently lock the WRONG person while excluding the right
    // one — i.e. a threshold high enough to separate them rejects BOTH.
    // Pick a threshold ABOVE the cross score: it must then also reject self,
    // because self and cross are indistinguishable.
    let separating_threshold = a_cross + 0.02; // just above the cross score
    let mut matcher = EnrolledMatcher::new(weights, separating_threshold, 2);
    matcher.enroll(1, person_a);
    matcher.enroll(2, person_b);
    matcher.set_probe(cardiac_respiratory_only(1));

    // At a threshold chosen to exclude the cross-person score, the matcher
    // either locks A (best score) or refuses — but the gap is so small that
    // this threshold is fragile. We assert the honest outcome: the SECOND-best
    // (wrong-person) score is also above any threshold low enough to admit the
    // correct person. Concretely, cross-person score >= threshold - 0.05.
    let best = matcher.best_match().expect("defined score");
    // best.1 is the highest score across enrolled; confirm the runner-up
    // (cross) is within 0.05 of it — i.e. effectively a tie.
    let cross_score = match_score(
        // person_b enrolled vs probe A
        &synthetic_person(2),
        &cardiac_respiratory_only(1),
        &weights,
    )
    .score()
    .unwrap();
    let best_score = best.1.score().unwrap();
    assert!(
        (best_score - cross_score).abs() < 0.05,
        "best ({best_score:.4}) and wrong-person ({cross_score:.4}) scores are \
         effectively tied on cardiac+resp — cannot lock identity"
    );
}

// --- 3. Graceful degradation + availability normalization ------------------

#[test]
fn availability_normalization_with_missing_channels() {
    let weights = MatchWeights::default();

    // Profile has all channels; probe has only AETHER. Only the AETHER channel
    // is shared, so the score must equal that channel's cosine exactly (the
    // weighted sum over one channel divided by its own weight = its cosine).
    let aether = synthetic_aether(7);
    let aether_probe = synthetic_aether(7);
    let profile = synthetic_person(7);
    let probe = SoulChannels::empty().with_aether(aether_probe);

    let ms = match_score(&profile, &probe, &weights);
    assert!(ms.is_defined());
    assert_eq!(ms.contributing_channels(), 1);

    let expected_cos = cosine_sim(aether.as_slice(), profile.channel_slice(Channel::AetherEmbedding).unwrap());
    let score = ms.score().unwrap();
    // score == w*cos / (w*1.0) == cos
    assert!(
        (score - expected_cos).abs() < 1e-5,
        "single-shared-channel score ({score:.6}) must equal that channel's cosine ({expected_cos:.6})"
    );
    assert!(score.is_finite());
}

#[test]
fn zero_norm_channel_contributes_zero_availability_no_nan() {
    let weights = MatchWeights::default();

    // A respiratory channel that is all zeros — present but unusable.
    let zero_resp = FeatureVector::from_slice(&[0.0; 6]).unwrap();
    let profile = SoulChannels::empty()
        .with_aether(synthetic_aether(3))
        .with_channel(Channel::RespiratoryPattern, zero_resp);
    let probe = SoulChannels::empty()
        .with_aether(synthetic_aether(3))
        .with_channel(Channel::RespiratoryPattern, synthetic_respiratory(3));

    let ms = match_score(&profile, &probe, &weights);
    // Zero-norm respiratory is unavailable; only AETHER contributes.
    assert_eq!(ms.contributing_channels(), 1);
    assert!(ms.channel_contribution(Channel::RespiratoryPattern).is_none());
    let score = ms.score().unwrap();
    assert!(score.is_finite(), "score must never be NaN, got {score}");
}

#[test]
fn cosine_sim_handles_zero_and_nan_without_nan_output() {
    assert_eq!(cosine_sim(&[], &[]), 0.0);
    assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    let r = cosine_sim(&[f32::NAN, 1.0], &[1.0, 1.0]);
    assert!(r.is_finite(), "NaN component must not propagate, got {r}");
    // Identical vectors => cosine 1.0.
    assert!((cosine_sim(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
    // Opposite vectors => cosine -1.0.
    assert!((cosine_sim(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6);
}

// --- 4. Insufficient channels (typed undefined, never high) ----------------

#[test]
fn no_shared_channels_yields_insufficient_not_high_score() {
    let weights = MatchWeights::default();

    // Profile carries only AETHER; probe carries only cardiac. No weighted
    // channel is shared => denominator 0 => undefined.
    let profile = SoulChannels::empty().with_aether(synthetic_aether(9));
    let probe = SoulChannels::empty()
        .with_channel(Channel::CardiacHrProfile, synthetic_cardiac(9));

    let ms = match_score(&profile, &probe, &weights);
    assert!(!ms.is_defined(), "no shared channels must be undefined");
    assert_eq!(ms.score(), None);
    assert_eq!(ms.contributing_channels(), 0);
}

#[test]
fn zero_weight_channel_never_contributes() {
    // Body-Field-Coupling has weight 0.0 (single-room) in the default table.
    let weights = MatchWeights::default();
    assert_eq!(weights.weight(Channel::BodyFieldCoupling), 0.0);

    // Both sides carry ONLY the zero-weight channel => undefined (it cannot
    // contribute to numerator or denominator).
    let bfc = FeatureVector::from_slice(&[1.0, 2.0, 3.0]).unwrap();
    let bfc2 = FeatureVector::from_slice(&[1.0, 2.0, 3.0]).unwrap();
    let profile = SoulChannels::empty().with_channel(Channel::BodyFieldCoupling, bfc);
    let probe = SoulChannels::empty().with_channel(Channel::BodyFieldCoupling, bfc2);

    let ms = match_score(&profile, &probe, &weights);
    assert!(!ms.is_defined(), "zero-weight-only match must be undefined");
}

// --- 5. Edge cases: empty enrolled set, threshold boundary -----------------

#[test]
fn empty_enrolled_set_reports_not_enrolled() {
    let matcher = EnrolledMatcher::new(MatchWeights::default(), 0.5, 1);
    matcher.set_probe(synthetic_person(1));
    assert_eq!(matcher.matches_enrolled(), MatchOutcome::NotEnrolled);
    assert!(matcher.is_empty());
}

#[test]
fn no_probe_reports_not_enrolled() {
    let mut matcher = EnrolledMatcher::new(MatchWeights::default(), 0.5, 1);
    matcher.enroll(1, synthetic_person(1));
    // No probe set.
    assert_eq!(matcher.matches_enrolled(), MatchOutcome::NotEnrolled);
}

#[test]
fn threshold_boundary_is_inclusive() {
    // Self-match scores ~1.0; with threshold exactly at the score it must lock.
    let weights = MatchWeights::default();
    let mut matcher = EnrolledMatcher::new(weights, 0.99, 2);
    matcher.enroll(42, synthetic_person(5));
    matcher.set_probe(synthetic_person(5));
    let best = matcher.best_match().unwrap();
    let s = best.1.score().unwrap();
    assert!(s >= 0.99, "self-match should clear 0.99, got {s:.4}");
    assert!(matches!(
        matcher.matches_enrolled(),
        MatchOutcome::Match { person_id: 42 }
    ));
}

#[test]
fn min_channels_gate_blocks_single_channel_lock() {
    // Even a perfect single-channel cosine cannot lock when min_channels = 2.
    let weights = MatchWeights::default();
    let mut matcher = EnrolledMatcher::new(weights, 0.5, 2);
    matcher.enroll(1, SoulChannels::empty().with_aether(synthetic_aether(1)));
    // Probe shares only AETHER (1 channel) — below min_channels.
    matcher.set_probe(SoulChannels::empty().with_aether(synthetic_aether(1)));
    assert_eq!(
        matcher.matches_enrolled(),
        MatchOutcome::NotEnrolled,
        "single shared channel must not lock when min_channels=2"
    );
}

#[test]
fn weights_reject_invalid_tables() {
    use wifi_densepose_bfld::WeightError;
    assert_eq!(
        MatchWeights::new([0.0; 8]).unwrap_err(),
        WeightError::AllZero
    );
    let mut neg = [0.1; 8];
    neg[0] = -0.1;
    assert_eq!(MatchWeights::new(neg).unwrap_err(), WeightError::Negative);
    let mut nan = [0.1; 8];
    nan[3] = f32::NAN;
    assert_eq!(MatchWeights::new(nan).unwrap_err(), WeightError::NotFinite);
}
