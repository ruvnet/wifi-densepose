//! LOCAL-DEV-ONLY bridge from RuForecast's `evaluate` CLI into Autogenous's
//! regression-candidate promotion path (`envelope::regression`).
//!
//! Not a production path: it path-depends on a sibling, unpublished
//! `ruvnet/autogenous` checkout (see Cargo.toml), and the "judges" here are
//! real independent train+evaluate replays over genuinely distinct synthetic
//! corpora (different `--seed` per judge), not independent human/adversarial
//! review — see the module doc in `envelope::regression` for why that
//! distinction matters and why `min_judges`/`min_samples`/margin are passed
//! explicitly rather than assumed.
//!
//! Darwin's own promotion gate (`harness/ruview/flywheel/ruforecast/gate.mjs`)
//! still runs first and is unchanged; this is an additional, stronger,
//! cryptographically-checked verification step layered on top of a candidate
//! Darwin already found — defense in depth, not a gate replacement.

use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use agl_types::{Applicability, Authority, Genome, HardGates, Mutation, MutationScope};
use anyhow::{bail, Context, Result};
use clap::Parser;
use constitution::{Constitution, RoleKeys};
use envelope::regression::{
    artifact_hash, sign_regression_promotion, sign_regression_receipt, verify_regression_promotion,
    MetricDirection, RegressionCandidateManifest,
};
use serde::{Deserialize, Serialize};
use witness::{content_hash, SigningAuthority};

#[derive(Parser, Debug)]
#[command(
    name = "ruforecast-autogenous-bridge",
    about = "LOCAL-DEV-ONLY: sign/verify a RuForecast hyperparameter candidate through Autogenous's regression-candidate promotion path"
)]
struct Cli {
    /// Path to the built `ruforecast` binary (ruforecast-train's CLI).
    #[arg(long)]
    ruforecast_bin: PathBuf,
    /// Candidate hyperparameter genome, JSON: {learning_rate, weight_decay, gradient_clip_norm, batch_size, epochs}.
    #[arg(long)]
    candidate_genome: PathBuf,
    /// Parent (baseline) hyperparameter genome, same JSON shape.
    #[arg(long)]
    parent_genome: PathBuf,
    /// Independent judges: full train+evaluate replays on genuinely distinct synthetic corpora.
    #[arg(long, default_value_t = 2)]
    judges: u64,
    /// Synthetic training windows per corpus.
    #[arg(long, default_value_t = 24)]
    train_windows: u64,
    /// Synthetic held-out windows per corpus.
    #[arg(long, default_value_t = 8)]
    test_windows: u64,
    /// Scratch directory for per-judge corpora/artifacts (created fresh; must not already exist).
    #[arg(long)]
    work_dir: PathBuf,
    /// Non-inferiority margin on weighted quantile loss (lower-is-better; a candidate must
    /// beat its parent by at least this much on every judge's receipt to be admissible).
    #[arg(long, default_value_t = 0.01)]
    margin: f64,
    /// Minimum held-out samples a receipt must report (floor stated honestly for this
    /// fixture's scale, not the detector domain's unrelated 1000-sample bar).
    #[arg(long, default_value_t = 4)]
    min_samples: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HyperparamGenome {
    learning_rate: f64,
    weight_decay: f64,
    gradient_clip_norm: f64,
    batch_size: u16,
    epochs: u16,
}

struct JudgeMeasurement {
    seed: u64,
    corpus_id: String,
    sample_count: usize,
    candidate_metric: f64,
    parent_metric: f64,
}

fn now_unix() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn run_ruforecast(bin: &Path, args: &[&str], label: &str) -> Result<String> {
    let output = Command::new(bin)
        .args(args)
        .output()
        .with_context(|| format!("spawning ruforecast for {label}"))?;
    if !output.status.success() {
        bail!(
            "ruforecast {label} failed (exit {:?}):\nstdout: {}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Prepare a synthetic corpus + train + evaluate one hyperparameter genome
/// against it, returning (weighted_quantile_loss, n_test_windows).
fn train_and_score(
    bin: &Path,
    genome: &HyperparamGenome,
    corpus_dir: &Path,
    seed: u64,
    train_windows: u64,
    test_windows: u64,
) -> Result<(f64, usize)> {
    std::fs::create_dir_all(corpus_dir.parent().unwrap_or(corpus_dir))?;
    run_ruforecast(
        bin,
        &[
            "prepare-synthetic-dataset",
            "--directory",
            corpus_dir.to_str().context("non-utf8 path")?,
            "--seed",
            &seed.to_string(),
            "--train-windows",
            &train_windows.to_string(),
            "--test-windows",
            &test_windows.to_string(),
            "--learning-rate",
            &genome.learning_rate.to_string(),
            "--weight-decay",
            &genome.weight_decay.to_string(),
            "--gradient-clip-norm",
            &genome.gradient_clip_norm.to_string(),
            "--batch-size",
            &genome.batch_size.to_string(),
            "--epochs",
            &genome.epochs.to_string(),
        ],
        "prepare-synthetic-dataset",
    )?;

    let artifacts_dir = corpus_dir.join("artifacts");
    run_ruforecast(
        bin,
        &[
            "train-local",
            "--request",
            corpus_dir.join("train-local.toml").to_str().context("non-utf8 path")?,
            "--dataset-root",
            corpus_dir.to_str().context("non-utf8 path")?,
            "--output",
            artifacts_dir.to_str().context("non-utf8 path")?,
        ],
        "train-local",
    )?;

    let candidate_mpk = artifacts_dir.join("synthetic-dataset").join("model.mpk");
    let eval_stdout = run_ruforecast(
        bin,
        &[
            "evaluate",
            "--candidate",
            candidate_mpk.to_str().context("non-utf8 path")?,
            "--test-jsonl",
            corpus_dir.join("test.jsonl").to_str().context("non-utf8 path")?,
            "--seasonal-period",
            "12",
        ],
        "evaluate",
    )?;
    let report: serde_json::Value =
        serde_json::from_str(&eval_stdout).context("parsing evaluate JSON output")?;
    let wql = report["model"]["weighted_quantile_loss"]
        .as_f64()
        .context("evaluate output missing model.weighted_quantile_loss")?;
    let n = report["n_test_windows"]
        .as_u64()
        .context("evaluate output missing n_test_windows")? as usize;
    Ok((wql, n))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.work_dir.exists() {
        bail!(
            "--work-dir {} already exists; pass a fresh directory",
            cli.work_dir.display()
        );
    }
    if cli.judges == 0 {
        bail!("--judges must be at least 1 (state min_judges: 1 honestly rather than 0)");
    }
    std::fs::create_dir_all(&cli.work_dir)?;

    let candidate: HyperparamGenome = serde_json::from_str(
        &std::fs::read_to_string(&cli.candidate_genome).context("reading --candidate-genome")?,
    )
    .context("parsing --candidate-genome JSON")?;
    let parent: HyperparamGenome = serde_json::from_str(
        &std::fs::read_to_string(&cli.parent_genome).context("reading --parent-genome")?,
    )
    .context("parsing --parent-genome JSON")?;

    let mut measurements = Vec::with_capacity(cli.judges as usize);
    for judge_index in 0..cli.judges {
        // Genuine corpus independence: each judge trains+evaluates BOTH
        // genomes against its own freshly-generated synthetic corpus (same
        // seed for candidate/parent within a judge, so the comparison is
        // apples-to-apples on that judge's held-out set; different seed
        // ACROSS judges, so disagreement between judges is a real signal
        // about generalization, not a re-run of identical arithmetic).
        let seed = 1000 + judge_index * 97;
        let judge_dir = cli.work_dir.join(format!("judge-{judge_index}"));
        let (candidate_wql, candidate_n) = train_and_score(
            &cli.ruforecast_bin,
            &candidate,
            &judge_dir.join("candidate"),
            seed,
            cli.train_windows,
            cli.test_windows,
        )
        .with_context(|| format!("judge {judge_index}: scoring candidate genome"))?;
        let (parent_wql, parent_n) = train_and_score(
            &cli.ruforecast_bin,
            &parent,
            &judge_dir.join("parent"),
            seed,
            cli.train_windows,
            cli.test_windows,
        )
        .with_context(|| format!("judge {judge_index}: scoring parent genome"))?;
        if candidate_n != parent_n {
            bail!(
                "judge {judge_index}: candidate/parent test-window counts disagree ({candidate_n} vs {parent_n}) — corpus was not actually shared"
            );
        }
        measurements.push(JudgeMeasurement {
            seed,
            corpus_id: format!("ruforecast-synthetic-seed-{seed}"),
            sample_count: candidate_n,
            candidate_metric: candidate_wql,
            parent_metric: parent_wql,
        });
    }

    let now = now_unix()?;
    let candidate_bytes = serde_json::to_vec(&candidate)?;
    let parent_bytes = serde_json::to_vec(&parent)?;
    let parent_genome_hash = artifact_hash(&parent_bytes);

    let judge_keys: Vec<SigningAuthority> = (0..cli.judges)
        .map(|i| {
            let mut seed = [0u8; 32];
            seed[0] = 1;
            seed[1..9].copy_from_slice(&i.to_le_bytes());
            SigningAuthority::from_seed(&format!("ruforecast-judge-{i}"), seed)
        })
        .collect();
    let controller_key = SigningAuthority::from_seed("ruforecast-controller", [9u8; 32]);

    let constitution = Constitution {
        identity: "ruforecast-hpo-promotion".into(),
        version: 1,
        authority_ceiling: Authority::Governed,
        prohibited_effects: vec!["pii_egress".into()],
        hard_gates: HardGates::default(),
        signers: vec!["ruforecast-bridge-operator".into()],
        pinned_keys: RoleKeys {
            judges: judge_keys.iter().map(SigningAuthority::public_hex).collect(),
            controllers: vec![controller_key.public_hex()],
        },
        effective_at: now.saturating_sub(1),
    };

    let parent_genome = Genome {
        hash: parent_genome_hash.clone(),
        identity: "ruforecast-hyperparameter-genome".into(),
        constitution: constitution.hash(),
        capability_ceiling: Authority::Governed,
        hard_invariants: vec![],
        lineage: vec![],
    };

    let mutation = Mutation {
        id: format!("ruforecast-hpo-candidate-{now}"),
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

    let receipts: Vec<_> = measurements
        .iter()
        .zip(judge_keys.iter())
        .map(|(m, judge)| {
            sign_regression_receipt(
                judge,
                &candidate_hash,
                &parent_hash_for_receipts,
                &m.corpus_id,
                m.sample_count,
                m.candidate_metric,
                m.parent_metric,
                "ruforecast-autogenous-bridge-v1",
                now,
            )
        })
        .collect();

    let promotion_envelope = sign_regression_promotion(
        &controller_key,
        &constitution.hash(),
        &candidate_hash,
        &receipts,
        &format!("ruforecast-hpo-{now}"),
        now,
        3600,
    );

    let min_samples = cli.min_samples.min(
        measurements
            .iter()
            .map(|m| m.sample_count)
            .min()
            .unwrap_or(cli.min_samples),
    );

    let rejections = verify_regression_promotion(
        &constitution,
        &parent_genome,
        &manifest,
        &receipts,
        &promotion_envelope,
        &[],
        cli.judges as usize,
        min_samples,
        cli.margin,
        now,
    );

    let decision = if rejections.is_empty() { "PROMOTE" } else { "REJECT" };
    let report = serde_json::json!({
        "decision": decision,
        "candidate_hash": candidate_hash,
        "constitution_hash": constitution.hash(),
        "judges": measurements.iter().map(|m| serde_json::json!({
            "seed": m.seed,
            "corpus_id": m.corpus_id,
            "sample_count": m.sample_count,
            "candidate_wql": m.candidate_metric,
            "parent_wql": m.parent_metric,
            "candidate_beats_parent_by": m.parent_metric - m.candidate_metric,
        })).collect::<Vec<_>>(),
        "min_judges": cli.judges,
        "min_samples": min_samples,
        "non_inferiority_margin": cli.margin,
        "rejections": rejections.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if decision == "REJECT" {
        std::process::exit(1);
    }
    Ok(())
}
