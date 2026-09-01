//! Leakage-safe temporal split and training contracts.

use crate::digest::CanonicalWriter;
use crate::series::validate_text;
use crate::{
    CanonicalDigest, DataPolicy, ForecastError, QuantileSet, SourceKind, SourceState,
    MAX_FORECAST_SPAN_MS, MAX_STEP_MS,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Maximum context rows accepted by a training contract.
pub const MAX_CONTEXT_LENGTH: usize = 16_384;
/// Maximum forecast horizon accepted by a training contract.
pub const MAX_HORIZON: usize = 4_096;
/// Maximum windows in each split partition.
pub const MAX_SPLIT_MEMBERS: usize = 4_096;
const MAX_SPLIT_ID_LEN: usize = 128;

/// Room, device, and session identity used only for split isolation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SeriesKey {
    room_id: String,
    device_id: String,
    session_id: String,
}

impl SeriesKey {
    /// Construct a bounded split key. None of its fields may be omitted.
    pub fn new(
        room_id: impl Into<String>,
        device_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self, ForecastError> {
        let room_id = room_id.into();
        let device_id = device_id.into();
        let session_id = session_id.into();
        validate_text("split_room_id", &room_id, MAX_SPLIT_ID_LEN, false)?;
        validate_text("split_device_id", &device_id, MAX_SPLIT_ID_LEN, false)?;
        validate_text("split_session_id", &session_id, MAX_SPLIT_ID_LEN, false)?;
        Ok(Self {
            room_id,
            device_id,
            session_id,
        })
    }

    /// Room identifier.
    #[must_use]
    pub fn room_id(&self) -> &str {
        &self.room_id
    }

    /// Device identifier.
    #[must_use]
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Session or recording identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn display_key(&self) -> String {
        format!("{}/{}/{}", self.room_id, self.device_id, self.session_id)
    }

    fn write_canonical(&self, writer: &mut CanonicalWriter) {
        writer.string(&self.room_id);
        writer.string(&self.device_id);
        writer.string(&self.session_id);
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeriesKeyWire {
    room_id: String,
    device_id: String,
    session_id: String,
}

impl<'de> Deserialize<'de> for SeriesKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SeriesKeyWire::deserialize(deserializer)?;
        Self::new(wire.room_id, wire.device_id, wire.session_id).map_err(serde::de::Error::custom)
    }
}

/// Half-open millisecond time range `[start_ms, end_ms)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TimeRange {
    start_ms: u64,
    end_ms: u64,
}

impl TimeRange {
    /// Construct a non-empty, non-reversed range.
    pub fn new(start_ms: u64, end_ms: u64) -> Result<Self, ForecastError> {
        if start_ms >= end_ms {
            return Err(ForecastError::InvalidTimeRange);
        }
        Ok(Self { start_ms, end_ms })
    }

    /// Inclusive range start.
    #[must_use]
    pub const fn start_ms(&self) -> u64 {
        self.start_ms
    }

    /// Exclusive range end.
    #[must_use]
    pub const fn end_ms(&self) -> u64 {
        self.end_ms
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimeRangeWire {
    start_ms: u64,
    end_ms: u64,
}

impl<'de> Deserialize<'de> for TimeRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TimeRangeWire::deserialize(deserializer)?;
        Self::new(wire.start_ms, wire.end_ms).map_err(serde::de::Error::custom)
    }
}

/// One series range assigned to exactly one logical partition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitMember {
    key: SeriesKey,
    range: TimeRange,
}

impl SplitMember {
    /// Construct one validated member.
    #[must_use]
    pub const fn new(key: SeriesKey, range: TimeRange) -> Self {
        Self { key, range }
    }

    /// Isolation key.
    #[must_use]
    pub const fn key(&self) -> &SeriesKey {
        &self.key
    }

    /// Assigned time range.
    #[must_use]
    pub const fn range(&self) -> TimeRange {
        self.range
    }
}

/// Entity dimension kept disjoint between train/calibration/test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HoldoutKey {
    /// Entire recording/session is isolated.
    Session,
    /// Entire room is isolated.
    Room,
    /// Entire sensing device is isolated.
    Device,
    /// Room, device, and session identifiers are each independently isolated.
    Strict,
}

/// Supported leakage-safe partition strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SplitStrategy {
    /// Same series may occur later only after a horizon-sized embargo.
    RollingOrigin,
    /// Selected entity identity may occur in only one partition.
    EntityHoldout(HoldoutKey),
}

/// Validated train/calibration/test assignment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TemporalSplitPlan {
    strategy: SplitStrategy,
    train: Vec<SplitMember>,
    calibration: Vec<SplitMember>,
    test: Vec<SplitMember>,
    horizon: usize,
    step_ms: u64,
    embargo_ms: u64,
}

impl TemporalSplitPlan {
    /// Construct and audit a split plan before any scaler or trainer is fit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: SplitStrategy,
        train: Vec<SplitMember>,
        calibration: Vec<SplitMember>,
        test: Vec<SplitMember>,
        horizon: usize,
        step_ms: u64,
        embargo_ms: u64,
    ) -> Result<Self, ForecastError> {
        if train.is_empty() {
            return Err(ForecastError::EmptyPartition { partition: "train" });
        }
        if test.is_empty() {
            return Err(ForecastError::EmptyPartition { partition: "test" });
        }
        for (name, members) in [
            ("train", train.as_slice()),
            ("calibration", calibration.as_slice()),
            ("test", test.as_slice()),
        ] {
            if members.len() > MAX_SPLIT_MEMBERS {
                return Err(ForecastError::LimitExceeded {
                    field: name,
                    actual: members.len(),
                    max: MAX_SPLIT_MEMBERS,
                });
            }
            reject_overlaps(name, members)?;
        }
        let required_embargo = validate_horizon_span(horizon, step_ms)?;
        match strategy {
            SplitStrategy::RollingOrigin => {
                if embargo_ms < required_embargo {
                    return Err(ForecastError::EmbargoTooSmall {
                        actual_ms: embargo_ms,
                        required_ms: required_embargo,
                    });
                }
                validate_rolling_order(&train, &calibration, &test, embargo_ms)?;
            }
            SplitStrategy::EntityHoldout(key) => {
                validate_entity_holdout(key, &train, &calibration, &test)?;
            }
        }
        Ok(Self {
            strategy,
            train,
            calibration,
            test,
            horizon,
            step_ms,
            embargo_ms,
        })
    }

    /// Split strategy.
    #[must_use]
    pub const fn strategy(&self) -> SplitStrategy {
        self.strategy
    }

    /// Training members only. Fit normalizers exclusively from this slice.
    #[must_use]
    pub fn train(&self) -> &[SplitMember] {
        &self.train
    }

    /// Optional calibration members.
    #[must_use]
    pub fn calibration(&self) -> &[SplitMember] {
        &self.calibration
    }

    /// Test members, never visible to training or model selection.
    #[must_use]
    pub fn test(&self) -> &[SplitMember] {
        &self.test
    }

    /// Target horizon protected by this plan.
    #[must_use]
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// Sample cadence protected by this plan.
    #[must_use]
    pub const fn step_ms(&self) -> u64 {
        self.step_ms
    }

    /// Gap between rolling-origin partitions.
    #[must_use]
    pub const fn embargo_ms(&self) -> u64 {
        self.embargo_ms
    }

    /// Deterministic digest of the exact split assignment.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"temporal-split-plan-v1");
        write_strategy(&mut writer, self.strategy);
        write_members(&mut writer, &self.train);
        write_members(&mut writer, &self.calibration);
        write_members(&mut writer, &self.test);
        writer.usize(self.horizon);
        writer.u64(self.step_ms);
        writer.u64(self.embargo_ms);
        writer.finish()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalSplitPlanWire {
    strategy: SplitStrategy,
    train: Vec<SplitMember>,
    calibration: Vec<SplitMember>,
    test: Vec<SplitMember>,
    horizon: usize,
    step_ms: u64,
    embargo_ms: u64,
}

impl<'de> Deserialize<'de> for TemporalSplitPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TemporalSplitPlanWire::deserialize(deserializer)?;
        Self::new(
            wire.strategy,
            wire.train,
            wire.calibration,
            wire.test,
            wire.horizon,
            wire.step_ms,
            wire.embargo_ms,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Normalization policy with no leakage-prone whole-dataset option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NormalizationPolicy {
    /// Preserve physical units.
    None,
    /// Fit and record standardization on the training partition only.
    StandardizeTrainOnly,
}

/// Whether a training contract is local or explicitly eligible for fal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TrainingDestinationKind {
    /// Operator-controlled local machine.
    Local,
    /// Remote fal processor generating synthetic data from a bounded recipe.
    FalSynthetic,
}

/// V1 fal contract that exports no dataset and generates synthetic rows from
/// an allowlisted recipe and deterministic seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntheticFalContract {
    generator_recipe_digest: CanonicalDigest,
    generator_seed: u64,
    source_digest: CanonicalDigest,
}

impl SyntheticFalContract {
    fn new(
        source: &SourceState,
        generator_recipe_digest: CanonicalDigest,
        generator_seed: u64,
    ) -> Result<Self, ForecastError> {
        if source.kind() != SourceKind::Synthetic {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal synthetic training",
                reason: "v1 hosted training requires synthetic provenance",
            });
        }
        if generator_recipe_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "generator_recipe_digest",
            });
        }
        Ok(Self {
            generator_recipe_digest,
            generator_seed,
            source_digest: source.canonical_digest(),
        })
    }

    /// Exact allowlisted generator recipe digest.
    #[must_use]
    pub const fn generator_recipe_digest(&self) -> CanonicalDigest {
        self.generator_recipe_digest
    }

    /// Deterministic generator seed.
    #[must_use]
    pub const fn generator_seed(&self) -> u64 {
        self.generator_seed
    }

    /// Exact synthetic provenance digest.
    #[must_use]
    pub const fn source_digest(&self) -> CanonicalDigest {
        self.source_digest
    }

    fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"synthetic-fal-contract-v1");
        writer.digest(self.generator_recipe_digest);
        writer.u64(self.generator_seed);
        writer.digest(self.source_digest);
        writer.finish()
    }
}

/// Validated, content-addressed training contract.
///
/// This trusted type intentionally does not implement `Deserialize`. Remote
/// DTOs must be byte-bounded, decoded into untrusted adapter-owned types, and
/// reconstructed through [`Self::new_local`] or [`Self::new_fal_synthetic`].
#[derive(Clone, Debug, PartialEq)]
pub struct TrainSpec {
    context_length: usize,
    horizon: usize,
    step_ms: u64,
    quantiles: QuantileSet,
    split_plan: TemporalSplitPlan,
    normalization: NormalizationPolicy,
    dataset_digest: CanonicalDigest,
    policy: DataPolicy,
    destination: TrainingDestinationKind,
    fal_synthetic_contract: Option<SyntheticFalContract>,
}

impl TrainSpec {
    /// Construct an operator-controlled local training contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new_local(
        context_length: usize,
        horizon: usize,
        step_ms: u64,
        quantiles: QuantileSet,
        split_plan: TemporalSplitPlan,
        normalization: NormalizationPolicy,
        dataset_digest: CanonicalDigest,
        policy: DataPolicy,
    ) -> Result<Self, ForecastError> {
        Self::from_parts(
            context_length,
            horizon,
            step_ms,
            quantiles,
            split_plan,
            normalization,
            dataset_digest,
            policy,
            TrainingDestinationKind::Local,
            None,
        )
    }

    /// Construct a v1 fal contract that sends no dataset and generates only
    /// synthetic data from an allowlisted recipe.
    #[allow(clippy::too_many_arguments)]
    pub fn new_fal_synthetic(
        context_length: usize,
        horizon: usize,
        step_ms: u64,
        quantiles: QuantileSet,
        split_plan: TemporalSplitPlan,
        normalization: NormalizationPolicy,
        dataset_digest: CanonicalDigest,
        policy: DataPolicy,
        source: &SourceState,
        generator_recipe_digest: CanonicalDigest,
        generator_seed: u64,
    ) -> Result<Self, ForecastError> {
        let synthetic = SyntheticFalContract::new(source, generator_recipe_digest, generator_seed)?;
        Self::from_parts(
            context_length,
            horizon,
            step_ms,
            quantiles,
            split_plan,
            normalization,
            dataset_digest,
            policy,
            TrainingDestinationKind::FalSynthetic,
            Some(synthetic),
        )
    }

    /// Context rows.
    #[must_use]
    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    /// Forecast horizon.
    #[must_use]
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// Sample cadence.
    #[must_use]
    pub const fn step_ms(&self) -> u64 {
        self.step_ms
    }

    /// Forecast probabilities.
    #[must_use]
    pub fn quantiles(&self) -> &QuantileSet {
        &self.quantiles
    }

    /// Audited split plan.
    #[must_use]
    pub fn split_plan(&self) -> &TemporalSplitPlan {
        &self.split_plan
    }

    /// Normalization policy.
    #[must_use]
    pub const fn normalization(&self) -> NormalizationPolicy {
        self.normalization
    }

    /// Exact dataset/corpus digest.
    #[must_use]
    pub const fn dataset_digest(&self) -> CanonicalDigest {
        self.dataset_digest
    }

    /// Exact privacy and governance binding.
    #[must_use]
    pub fn policy(&self) -> &DataPolicy {
        &self.policy
    }

    /// Training destination.
    #[must_use]
    pub const fn destination(&self) -> TrainingDestinationKind {
        self.destination
    }

    /// Synthetic fal contract. This is `None` for every local contract.
    #[must_use]
    pub fn fal_synthetic_contract(&self) -> Option<&SyntheticFalContract> {
        self.fal_synthetic_contract.as_ref()
    }

    /// Fail closed unless this is a synthetic-only fal contract.
    pub fn require_fal_synthetic_contract(&self) -> Result<&SyntheticFalContract, ForecastError> {
        if self.destination != TrainingDestinationKind::FalSynthetic {
            return Err(ForecastError::PrivacyDenied {
                operation: "fal submission",
                reason: "training contract is local-only",
            });
        }
        self.fal_synthetic_contract
            .as_ref()
            .ok_or(ForecastError::MissingReceipt {
                field: "fal export",
            })
    }

    /// Deterministic digest of the complete trusted training contract.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"train-spec-v1");
        writer.usize(self.context_length);
        writer.usize(self.horizon);
        writer.u64(self.step_ms);
        writer.digest(self.quantiles.canonical_digest());
        writer.digest(self.split_plan.canonical_digest());
        writer.tag(match self.normalization {
            NormalizationPolicy::None => 0,
            NormalizationPolicy::StandardizeTrainOnly => 1,
        });
        writer.digest(self.dataset_digest);
        writer.digest(self.policy.canonical_digest());
        writer.tag(match self.destination {
            TrainingDestinationKind::Local => 0,
            TrainingDestinationKind::FalSynthetic => 1,
        });
        match &self.fal_synthetic_contract {
            Some(contract) => {
                writer.bool(true);
                writer.digest(contract.canonical_digest());
            }
            None => writer.bool(false),
        }
        writer.finish()
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        context_length: usize,
        horizon: usize,
        step_ms: u64,
        quantiles: QuantileSet,
        split_plan: TemporalSplitPlan,
        normalization: NormalizationPolicy,
        dataset_digest: CanonicalDigest,
        policy: DataPolicy,
        destination: TrainingDestinationKind,
        fal_synthetic_contract: Option<SyntheticFalContract>,
    ) -> Result<Self, ForecastError> {
        if context_length == 0 {
            return Err(ForecastError::ZeroValue {
                field: "context_length",
            });
        }
        if context_length > MAX_CONTEXT_LENGTH {
            return Err(ForecastError::LimitExceeded {
                field: "context_length",
                actual: context_length,
                max: MAX_CONTEXT_LENGTH,
            });
        }
        validate_horizon_span(horizon, step_ms)?;
        if split_plan.horizon != horizon {
            return Err(ForecastError::SplitPlanMismatch {
                field: "horizon",
                actual: horizon as u64,
                expected: split_plan.horizon as u64,
            });
        }
        if split_plan.step_ms != step_ms {
            return Err(ForecastError::SplitPlanMismatch {
                field: "step_ms",
                actual: step_ms,
                expected: split_plan.step_ms,
            });
        }
        if dataset_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "training_dataset_digest",
            });
        }
        match (destination, fal_synthetic_contract.is_some()) {
            (TrainingDestinationKind::Local, false)
            | (TrainingDestinationKind::FalSynthetic, true) => {}
            (TrainingDestinationKind::Local, true) => {
                return Err(ForecastError::PrivacyDenied {
                    operation: "local training",
                    reason: "local contract must not carry a fal receipt",
                });
            }
            (TrainingDestinationKind::FalSynthetic, false) => {
                return Err(ForecastError::MissingReceipt {
                    field: "fal synthetic contract",
                });
            }
        }
        Ok(Self {
            context_length,
            horizon,
            step_ms,
            quantiles,
            split_plan,
            normalization,
            dataset_digest,
            policy,
            destination,
            fal_synthetic_contract,
        })
    }
}

fn validate_horizon_span(horizon: usize, step_ms: u64) -> Result<u64, ForecastError> {
    if horizon == 0 {
        return Err(ForecastError::ZeroValue { field: "horizon" });
    }
    if horizon > MAX_HORIZON {
        return Err(ForecastError::LimitExceeded {
            field: "horizon",
            actual: horizon,
            max: MAX_HORIZON,
        });
    }
    if step_ms == 0 {
        return Err(ForecastError::ZeroValue { field: "step_ms" });
    }
    if step_ms > MAX_STEP_MS {
        return Err(ForecastError::DurationLimitExceeded {
            field: "step_ms",
            actual_ms: step_ms,
            max_ms: MAX_STEP_MS,
        });
    }
    let horizon = horizon as u64;
    let span = step_ms
        .checked_mul(horizon)
        .ok_or(ForecastError::SizeOverflow {
            field: "split_horizon_span_ms",
        })?;
    if span > MAX_FORECAST_SPAN_MS {
        return Err(ForecastError::DurationLimitExceeded {
            field: "split_horizon_span_ms",
            actual_ms: span,
            max_ms: MAX_FORECAST_SPAN_MS,
        });
    }
    Ok(span)
}

fn reject_overlaps(partition: &'static str, members: &[SplitMember]) -> Result<(), ForecastError> {
    let mut grouped: BTreeMap<&SeriesKey, Vec<TimeRange>> = BTreeMap::new();
    for member in members {
        grouped.entry(&member.key).or_default().push(member.range);
    }
    for (key, ranges) in &mut grouped {
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].end_ms > pair[1].start_ms {
                return Err(ForecastError::OverlappingWindows {
                    partition,
                    series: key.display_key(),
                });
            }
        }
    }
    Ok(())
}

fn validate_entity_holdout(
    key: HoldoutKey,
    train: &[SplitMember],
    calibration: &[SplitMember],
    test: &[SplitMember],
) -> Result<(), ForecastError> {
    for (left, right) in [(train, calibration), (train, test), (calibration, test)] {
        match key {
            HoldoutKey::Session => reject_shared("session", left, right, |k| &k.session_id)?,
            HoldoutKey::Room => reject_shared("room", left, right, |k| &k.room_id)?,
            HoldoutKey::Device => reject_shared("device", left, right, |k| &k.device_id)?,
            HoldoutKey::Strict => {
                reject_shared("session", left, right, |k| &k.session_id)?;
                reject_shared("room", left, right, |k| &k.room_id)?;
                reject_shared("device", left, right, |k| &k.device_id)?;
            }
        }
    }
    Ok(())
}

fn reject_shared<'a, F>(
    dimension: &'static str,
    left: &'a [SplitMember],
    right: &'a [SplitMember],
    value: F,
) -> Result<(), ForecastError>
where
    F: Fn(&'a SeriesKey) -> &'a String,
{
    let left_values: BTreeSet<&String> = left.iter().map(|member| value(&member.key)).collect();
    if let Some(shared) = right
        .iter()
        .map(|member| value(&member.key))
        .find(|item| left_values.contains(item))
    {
        return Err(ForecastError::SplitLeakage {
            dimension,
            value: shared.clone(),
        });
    }
    Ok(())
}

fn validate_rolling_order(
    train: &[SplitMember],
    calibration: &[SplitMember],
    test: &[SplitMember],
    embargo_ms: u64,
) -> Result<(), ForecastError> {
    validate_partition_order(train, calibration, embargo_ms)?;
    validate_partition_order(train, test, embargo_ms)?;
    validate_partition_order(calibration, test, embargo_ms)?;
    Ok(())
}

fn validate_partition_order(
    earlier: &[SplitMember],
    later: &[SplitMember],
    embargo_ms: u64,
) -> Result<(), ForecastError> {
    let mut earlier_end: BTreeMap<&SeriesKey, u64> = BTreeMap::new();
    for member in earlier {
        earlier_end
            .entry(&member.key)
            .and_modify(|end| *end = (*end).max(member.range.end_ms))
            .or_insert(member.range.end_ms);
    }
    let mut later_start: BTreeMap<&SeriesKey, u64> = BTreeMap::new();
    for member in later {
        later_start
            .entry(&member.key)
            .and_modify(|start| *start = (*start).min(member.range.start_ms))
            .or_insert(member.range.start_ms);
    }
    for (key, end) in earlier_end {
        if let Some(start) = later_start.get(key) {
            let safe_end = end
                .checked_add(embargo_ms)
                .ok_or(ForecastError::SizeOverflow {
                    field: "split_embargo_end_ms",
                })?;
            if safe_end > *start {
                return Err(ForecastError::SplitOrder {
                    series: key.display_key(),
                });
            }
        }
    }
    Ok(())
}

fn write_strategy(writer: &mut CanonicalWriter, strategy: SplitStrategy) {
    match strategy {
        SplitStrategy::RollingOrigin => writer.tag(0),
        SplitStrategy::EntityHoldout(key) => {
            writer.tag(1);
            writer.tag(match key {
                HoldoutKey::Session => 0,
                HoldoutKey::Room => 1,
                HoldoutKey::Device => 2,
                HoldoutKey::Strict => 3,
            });
        }
    }
}

fn write_members(writer: &mut CanonicalWriter, members: &[SplitMember]) {
    writer.usize(members.len());
    for member in members {
        member.key.write_canonical(writer);
        writer.u64(member.range.start_ms);
        writer.u64(member.range.end_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrivacyClass;

    fn digest(value: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"split-test", value)
    }

    fn member(room: &str, session: &str, start: u64, end: u64) -> SplitMember {
        SplitMember::new(
            SeriesKey::new(room, format!("device-{room}"), session).unwrap(),
            TimeRange::new(start, end).unwrap(),
        )
    }

    fn policy(retention: u64) -> DataPolicy {
        DataPolicy::new(
            PrivacyClass::P1,
            "tenant",
            "account",
            "workspace",
            "forecast training",
            digest(b"policy"),
            Some(digest(b"consent")),
            Some(digest(b"dpa")),
            Some(digest(b"export")),
            retention,
            true,
        )
        .unwrap()
    }

    #[test]
    fn entity_holdout_rejects_room_leakage() {
        let result = TemporalSplitPlan::new(
            SplitStrategy::EntityHoldout(HoldoutKey::Room),
            vec![member("a", "s1", 0, 10)],
            vec![],
            vec![member("a", "s2", 20, 30)],
            2,
            1,
            0,
        );
        assert!(matches!(result, Err(ForecastError::SplitLeakage { .. })));
    }

    #[test]
    fn rolling_origin_requires_horizon_embargo_and_order() {
        let too_small = TemporalSplitPlan::new(
            SplitStrategy::RollingOrigin,
            vec![member("a", "s", 0, 10)],
            vec![],
            vec![member("a", "s", 12, 20)],
            4,
            1,
            3,
        );
        assert!(matches!(
            too_small,
            Err(ForecastError::EmbargoTooSmall { .. })
        ));
        let valid = TemporalSplitPlan::new(
            SplitStrategy::RollingOrigin,
            vec![member("a", "s", 0, 10)],
            vec![],
            vec![member("a", "s", 14, 20)],
            4,
            1,
            4,
        );
        assert!(valid.is_ok());
    }

    #[test]
    fn local_spec_cannot_be_submitted_to_fal() {
        let split = TemporalSplitPlan::new(
            SplitStrategy::EntityHoldout(HoldoutKey::Room),
            vec![member("train", "s1", 0, 10)],
            vec![],
            vec![member("test", "s2", 20, 30)],
            2,
            1,
            0,
        )
        .unwrap();
        let spec = TrainSpec::new_local(
            4,
            2,
            1,
            QuantileSet::new(vec![0.5]).unwrap(),
            split,
            NormalizationPolicy::StandardizeTrainOnly,
            digest(b"dataset"),
            policy(1_000),
        )
        .unwrap();
        assert!(spec.require_fal_synthetic_contract().is_err());
    }

    #[test]
    fn hosted_spec_accepts_only_explicit_synthetic_recipe() {
        let make_split = || {
            TemporalSplitPlan::new(
                SplitStrategy::EntityHoldout(HoldoutKey::Room),
                vec![member("train", "s1", 0, 10)],
                vec![],
                vec![member("test", "s2", 20, 30)],
                2,
                1,
                0,
            )
            .unwrap()
        };
        let claimed = SourceState::claimed("caller claim").unwrap();
        let denied = TrainSpec::new_fal_synthetic(
            4,
            2,
            1,
            QuantileSet::new(vec![0.5]).unwrap(),
            make_split(),
            NormalizationPolicy::StandardizeTrainOnly,
            digest(b"dataset"),
            policy(1_000),
            &claimed,
            digest(b"recipe"),
            42,
        );
        assert!(matches!(denied, Err(ForecastError::PrivacyDenied { .. })));

        let synthetic = SourceState::synthetic("allowlisted generator").unwrap();
        let allowed = TrainSpec::new_fal_synthetic(
            4,
            2,
            1,
            QuantileSet::new(vec![0.5]).unwrap(),
            make_split(),
            NormalizationPolicy::StandardizeTrainOnly,
            digest(b"dataset"),
            policy(1_000),
            &synthetic,
            digest(b"recipe"),
            42,
        )
        .unwrap();
        assert_eq!(
            allowed
                .require_fal_synthetic_contract()
                .unwrap()
                .generator_seed(),
            42
        );
    }
}
