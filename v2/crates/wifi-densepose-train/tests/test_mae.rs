//! Integration + property tests for [`wifi_densepose_train::mae`]
//! (ADR-152 §2.3 — UNSW MAE pretraining recipe).
//!
//! All deterministic tests use fixed seeds; property tests use `proptest`
//! with its default deterministic-replay machinery.

use proptest::prelude::*;
use wifi_densepose_train::mae::{
    patchify, random_mask, unpatchify, unpatchify_visible, MaePretrainConfig,
};
use wifi_densepose_train::MaeError;

/// Deterministic test window: value = t * 1000 + sc (every cell unique).
fn window(time: usize, subc: usize) -> Vec<f32> {
    (0..time * subc)
        .map(|i| ((i / subc) * 1000 + i % subc) as f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Config defaults + validation
// ---------------------------------------------------------------------------

#[test]
fn default_config_matches_unsw_recipe() {
    let cfg = MaePretrainConfig::default();
    assert!((cfg.mask_ratio - 0.80).abs() < 1e-12);
    assert_eq!(cfg.patch_time, 30);
    assert_eq!(cfg.patch_subc, 3);
    assert_eq!(cfg.seed, 42);
    cfg.validate().expect("default recipe is valid");
}

#[test]
fn config_json_round_trip() {
    let cfg = MaePretrainConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: MaePretrainConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back, cfg);
}

#[test]
fn invalid_mask_ratio_rejected() {
    for ratio in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        let cfg = MaePretrainConfig {
            mask_ratio: ratio,
            ..MaePretrainConfig::default()
        };
        assert!(cfg.validate().is_err(), "ratio {ratio} should be invalid");
    }
}

#[test]
fn zero_patch_dims_rejected() {
    let cfg = MaePretrainConfig {
        patch_time: 0,
        ..MaePretrainConfig::default()
    };
    assert!(cfg.validate().is_err());
    let cfg = MaePretrainConfig {
        patch_subc: 0,
        ..MaePretrainConfig::default()
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Divisibility policy: error, never truncate
// ---------------------------------------------------------------------------

#[test]
fn non_divisible_window_errors_with_crop_hint() {
    let cfg = MaePretrainConfig::default(); // (30, 3)
                                            // Default TrainingConfig window 100 × 56 is NOT divisible by (30, 3).
    let err = cfg.validate_for_window(100, 56).unwrap_err();
    match err {
        MaeError::NotDivisible {
            axis,
            window,
            patch,
            remainder,
            crop,
        } => {
            assert_eq!(axis, "time");
            assert_eq!(window, 100);
            assert_eq!(patch, 30);
            assert_eq!(remainder, 10);
            assert_eq!(crop, 90);
        }
        other => panic!("expected NotDivisible, got {other:?}"),
    }
    assert_eq!(cfg.cropped_window_shape(100, 56), (90, 54));
    // The hinted crop validates cleanly.
    cfg.validate_for_window(90, 54).expect("crop is divisible");
    assert_eq!(cfg.num_patches(90, 54).unwrap(), 3 * 18);
}

#[test]
fn patch_larger_than_window_errors() {
    let cfg = MaePretrainConfig::default();
    let err = cfg.validate_for_window(20, 3).unwrap_err();
    assert!(matches!(
        err,
        MaeError::PatchExceedsWindow { axis: "time", .. }
    ));
}

#[test]
fn window_length_mismatch_errors() {
    let cfg = MaePretrainConfig::default();
    let buf = vec![0.0_f32; 89 * 54]; // declared 90 × 54
    let err = patchify(&buf, 90, 54, &cfg).unwrap_err();
    assert!(matches!(err, MaeError::WindowShapeMismatch { .. }));
}

// ---------------------------------------------------------------------------
// NaN handling
// ---------------------------------------------------------------------------

#[test]
fn nan_and_inf_input_rejected_with_location() {
    let cfg = MaePretrainConfig::default();
    let mut buf = window(90, 54);
    buf[2 * 54 + 7] = f32::NAN;
    match patchify(&buf, 90, 54, &cfg).unwrap_err() {
        MaeError::NonFiniteValue { row, col, .. } => {
            assert_eq!((row, col), (2, 7));
        }
        other => panic!("expected NonFiniteValue, got {other:?}"),
    }
    buf[2 * 54 + 7] = f32::INFINITY;
    assert!(matches!(
        patchify(&buf, 90, 54, &cfg),
        Err(MaeError::NonFiniteValue { .. })
    ));
}

#[test]
fn finite_input_is_nan_free_after_round_trip() {
    let cfg = MaePretrainConfig::default();
    let buf = window(90, 54);
    let grid = patchify(&buf, 90, 54, &cfg).unwrap();
    assert!(grid.patches.iter().flatten().all(|v| v.is_finite()));
    assert!(unpatchify(&grid).iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// Patchify / unpatchify round trip
// ---------------------------------------------------------------------------

#[test]
fn patchify_unpatchify_identity_default_recipe() {
    let cfg = MaePretrainConfig::default();
    let buf = window(90, 54);
    let grid = patchify(&buf, 90, 54, &cfg).unwrap();
    assert_eq!(grid.n_patches(), 54);
    assert_eq!(grid.patch_len(), 90);
    assert_eq!(grid.window_shape(), (90, 54));
    assert_eq!(unpatchify(&grid), buf);
}

#[test]
fn patch_layout_is_time_major() {
    // 4 × 4 window, (2, 2) patches → patch 0 is rows 0–1 × cols 0–1.
    let cfg = MaePretrainConfig {
        patch_time: 2,
        patch_subc: 2,
        ..MaePretrainConfig::default()
    };
    let buf = window(4, 4);
    let grid = patchify(&buf, 4, 4, &cfg).unwrap();
    assert_eq!(grid.patches[0], vec![0.0, 1.0, 1000.0, 1001.0]);
    // Patch index 1 is the next subcarrier block on the same time rows.
    assert_eq!(grid.patches[1], vec![2.0, 3.0, 1002.0, 1003.0]);
    // Patch index n_patches_subc starts the second time row of patches.
    assert_eq!(grid.patches[2], vec![2000.0, 2001.0, 3000.0, 3001.0]);
}

#[test]
fn unpatchify_visible_restores_visible_and_fills_masked() {
    let cfg = MaePretrainConfig::default();
    let buf = window(90, 54);
    let (grid, mask) = cfg.mask_window(&buf, 90, 54).unwrap();
    let fill = -1.0_f32;
    let recon = unpatchify_visible(&grid, &mask.visible, fill);

    // Visible patch regions are identical to the input; masked regions = fill.
    let full = unpatchify(&grid);
    assert_eq!(full, buf);
    let mut n_fill = 0usize;
    for (i, (&r, &orig)) in recon.iter().zip(buf.iter()).enumerate() {
        if r == fill && orig != fill {
            n_fill += 1;
        } else {
            assert_eq!(r, orig, "visible value at flat index {i} must round-trip");
        }
    }
    assert_eq!(n_fill, mask.masked.len() * grid.patch_len());
}

// ---------------------------------------------------------------------------
// Random mask: exact count, determinism, disjointness
// ---------------------------------------------------------------------------

#[test]
fn mask_count_is_exact_for_default_recipe() {
    // 54 patches @ 0.80 → round(43.2) = 43 masked, 11 visible.
    let cfg = MaePretrainConfig::default();
    assert_eq!(cfg.num_masked(54), 43);
    let mask = random_mask(54, cfg.mask_ratio, cfg.seed).unwrap();
    assert_eq!(mask.masked.len(), 43);
    assert_eq!(mask.visible.len(), 11);
}

#[test]
fn same_seed_same_mask_different_seed_differs() {
    let a = random_mask(100, 0.80, 7).unwrap();
    let b = random_mask(100, 0.80, 7).unwrap();
    assert_eq!(a, b, "same (n, ratio, seed) must reproduce the mask");

    let c = random_mask(100, 0.80, 8).unwrap();
    assert_ne!(a.masked, c.masked, "different seeds must differ");
}

#[test]
fn random_mask_rejects_invalid_ratios() {
    // Error-not-silent: NaN must not silently mask 0 patches; ratios outside
    // (0, 1) must not degenerate to all-visible / all-masked grids.
    for ratio in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0,
        1.5,
        0.0,
        -0.1,
    ] {
        let err = random_mask(54, ratio, 42).unwrap_err();
        assert!(
            matches!(err, MaeError::InvalidMaskRatio { .. }),
            "ratio {ratio} must be rejected, got {err:?}"
        );
    }
}

#[test]
fn mask_window_rejects_invalid_ratio_before_masking() {
    let cfg = MaePretrainConfig {
        mask_ratio: f64::NAN,
        ..MaePretrainConfig::default()
    };
    let buf = window(90, 54);
    assert!(matches!(
        cfg.mask_window(&buf, 90, 54),
        Err(MaeError::InvalidMaskRatio { .. })
    ));
}

proptest! {
    /// Exact count, sortedness, range, disjointness, and full coverage hold
    /// for arbitrary grid sizes, ratios, and seeds.
    #[test]
    fn prop_mask_invariants(
        n in 1usize..600,
        ratio in 0.01f64..0.99,
        seed in any::<u64>(),
    ) {
        let mask = random_mask(n, ratio, seed).unwrap();
        let expected_masked = ((ratio * n as f64).round() as usize).min(n);
        prop_assert_eq!(mask.masked.len(), expected_masked);
        prop_assert_eq!(mask.masked.len() + mask.visible.len(), n);

        // In range, sorted, strictly increasing (no duplicates).
        for set in [&mask.masked, &mask.visible] {
            for w in set.windows(2) {
                prop_assert!(w[0] < w[1]);
            }
            if let Some(&last) = set.last() {
                prop_assert!(last < n);
            }
        }
        // Disjoint + complete: merged sets are exactly 0..n.
        let mut all: Vec<usize> = mask.masked.iter().chain(&mask.visible).copied().collect();
        all.sort_unstable();
        prop_assert_eq!(all, (0..n).collect::<Vec<_>>());
    }

    /// Determinism by seed for arbitrary inputs.
    #[test]
    fn prop_mask_deterministic(n in 1usize..400, seed in any::<u64>()) {
        prop_assert_eq!(
            random_mask(n, 0.80, seed).unwrap(),
            random_mask(n, 0.80, seed).unwrap()
        );
    }

    /// Round-trip identity for arbitrary divisible window/patch geometries.
    #[test]
    fn prop_patchify_round_trip(
        pt in 1usize..8,
        ps in 1usize..8,
        nt in 1usize..6,
        ns in 1usize..6,
        seed in any::<u64>(),
    ) {
        let (time, subc) = (pt * nt, ps * ns);
        let cfg = MaePretrainConfig {
            patch_time: pt,
            patch_subc: ps,
            seed,
            ..MaePretrainConfig::default()
        };
        let buf = window(time, subc);
        let grid = patchify(&buf, time, subc, &cfg).unwrap();
        prop_assert_eq!(grid.n_patches(), nt * ns);
        prop_assert_eq!(unpatchify(&grid), buf);
    }
}
