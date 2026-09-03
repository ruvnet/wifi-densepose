//! Independently verify the real-household-data WQL results (two real,
//! differently-split judges) through Autogenous's real regression-candidate
//! promotion path -- a genuine signed verdict, not a self-reported number.
//! LOCAL-DEV-ONLY, mirrors ruforecast-autogenous-bridge/src/main.rs's real
//! envelope::regression usage exactly, but consumes already-measured real
//! WQL numbers instead of running its own synthetic train_and_score.

use agl_types::{Applicability, Authority, Genome, HardGates, Mutation, MutationScope};
use anyhow::Result;
use constitution::{Constitution, RoleKeys};
use envelope::regression::{
    artifact_hash, sign_regression_promotion, sign_regression_receipt, verify_regression_promotion,
    MetricDirection, RegressionCandidateManifest,
};
use witness::{content_hash, SigningAuthority};

struct RealJudge {
    label: &'static str,
    corpus_id: String,
    sample_count: usize,
    candidate_wql: f64,
    parent_wql: f64,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn main() -> Result<()> {
    // Real numbers from two genuinely independent temporal splits of the
    // same 6390-sample real household vitals corpus (train_fraction 0.70
    // vs 0.50, both with a 90s embargo). "candidate" = the trained model's
    // WQL; "parent" = the better of the two trivial baselines on that same
    // judge's held-out set (min(last_value, seasonal_naive)), matching this
    // session's established `primary` convention throughout.
    let judges = vec![
        RealJudge {
            label: "split-70-30",
            corpus_id: "ruforecast-real-home-lab-split-70".into(),
            sample_count: 27,
            candidate_wql: 0.0513991104899006,
            parent_wql: 0.05630054057848733_f64.min(0.10253572536393343),
        },
        RealJudge {
            label: "split-50-50",
            corpus_id: "ruforecast-real-home-lab-split-50".into(),
            sample_count: 46,
            candidate_wql: 0.06704526404505064,
            parent_wql: 0.0618358588096424_f64.min(0.054265071040759255),
        },
    ];

    let now = now_unix();
    let candidate_bytes = b"real-home-lab-default-optimizer-spec-v1".to_vec();
    let parent_bytes = b"trivial-baseline-min-last-value-seasonal-naive".to_vec();
    let parent_genome_hash = artifact_hash(&parent_bytes);

    let judge_keys: Vec<SigningAuthority> = (0..judges.len() as u64)
        .map(|i| {
            let mut seed = [0u8; 32];
            seed[0] = 2;
            seed[1..9].copy_from_slice(&i.to_le_bytes());
            SigningAuthority::from_seed(&format!("ruforecast-real-judge-{i}"), seed)
        })
        .collect();
    let controller_key = SigningAuthority::from_seed("ruforecast-real-controller", [8u8; 32]);

    let constitution = Constitution {
        identity: "ruforecast-real-data-promotion".into(),
        version: 1,
        authority_ceiling: Authority::Governed,
        prohibited_effects: vec!["pii_egress".into()],
        hard_gates: HardGates::default(),
        signers: vec!["ruforecast-real-bridge-operator".into()],
        pinned_keys: RoleKeys {
            judges: judge_keys.iter().map(SigningAuthority::public_hex).collect(),
            controllers: vec![controller_key.public_hex()],
        },
        effective_at: now.saturating_sub(1),
    };

    let parent_genome = Genome {
        hash: parent_genome_hash.clone(),
        identity: "ruforecast-real-data-parent".into(),
        constitution: constitution.hash(),
        capability_ceiling: Authority::Governed,
        hard_invariants: vec![],
        lineage: vec![],
    };

    let mutation = Mutation {
        id: format!("ruforecast-real-data-candidate-{now}"),
        parent_genome_hash: parent_genome.hash.clone(),
        scope: MutationScope::ApplicationCode,
        requested_authority: Authority::Governed,
        applicability: Applicability::default(),
        preserved_invariants: vec![],
        rollback_target: Some(parent_genome.hash.clone()),
        expires_at: Some(now + 3600),
        signature: None,
    };

    let manifest = RegressionCandidateManifest::from_parts(
        mutation,
        &candidate_bytes,
        "weighted_quantile_loss",
        MetricDirection::LowerIsBetter,
        vec![],
        vec![],
        vec![],
    );
    let candidate_hash = manifest.candidate_hash();
    let parent_hash_for_receipts = content_hash(&parent_genome.hash);

    let receipts: Vec<_> = judges
        .iter()
        .zip(judge_keys.iter())
        .map(|(j, judge)| {
            sign_regression_receipt(
                judge,
                &candidate_hash,
                &parent_hash_for_receipts,
                &j.corpus_id,
                j.sample_count,
                j.candidate_wql,
                j.parent_wql,
                "ruforecast-real-data-verify-v1",
                now,
            )
        })
        .collect();

    let promotion_envelope = sign_regression_promotion(
        &controller_key,
        &constitution.hash(),
        &candidate_hash,
        &receipts,
        &format!("ruforecast-real-data-{now}"),
        now,
        3600,
    );

    let min_samples = judges.iter().map(|j| j.sample_count).min().unwrap_or(1);
    let margin = 0.01_f64;

    let rejections = verify_regression_promotion(
        &constitution,
        &parent_genome,
        &manifest,
        &receipts,
        &promotion_envelope,
        &[],
        judges.len(),
        min_samples,
        margin,
        now,
    );

    let decision = if rejections.is_empty() { "PROMOTE" } else { "REJECT" };
    let report = serde_json::json!({
        "decision": decision,
        "candidate_hash": candidate_hash,
        "constitution_hash": constitution.hash(),
        "judges": judges.iter().map(|j| serde_json::json!({
            "label": j.label,
            "corpus_id": j.corpus_id,
            "sample_count": j.sample_count,
            "candidate_wql": j.candidate_wql,
            "parent_wql": j.parent_wql,
            "candidate_beats_parent_by": j.parent_wql - j.candidate_wql,
        })).collect::<Vec<_>>(),
        "min_judges": judges.len(),
        "min_samples": min_samples,
        "non_inferiority_margin": margin,
        "rejections": rejections.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
