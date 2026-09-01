//! Error types for validated forecast contracts.

use thiserror::Error;

/// Errors returned when an input violates a forecasting contract.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ForecastError {
    /// A required text field was empty.
    #[error("{field} must not be empty")]
    EmptyField {
        /// Name of the invalid field.
        field: &'static str,
    },

    /// A text field exceeded its byte bound.
    #[error("{field} is {actual} bytes; maximum is {max}")]
    TextTooLong {
        /// Name of the invalid field.
        field: &'static str,
        /// Observed UTF-8 byte length.
        actual: usize,
        /// Maximum accepted byte length.
        max: usize,
    },

    /// Text had surrounding whitespace or control characters.
    #[error("{field} contains non-canonical text")]
    InvalidText {
        /// Name of the invalid field.
        field: &'static str,
    },

    /// A bounded collection or scalar exceeded its contract.
    #[error("{field} is {actual}; maximum is {max}")]
    LimitExceeded {
        /// Name of the bounded value.
        field: &'static str,
        /// Observed value.
        actual: usize,
        /// Maximum accepted value.
        max: usize,
    },

    /// A duration exceeded its millisecond bound.
    #[error("{field} is {actual_ms} ms; maximum is {max_ms} ms")]
    DurationLimitExceeded {
        /// Name of the bounded duration.
        field: &'static str,
        /// Observed milliseconds.
        actual_ms: u64,
        /// Maximum milliseconds.
        max_ms: u64,
    },

    /// A required integer was zero.
    #[error("{field} must be greater than zero")]
    ZeroValue {
        /// Name of the invalid field.
        field: &'static str,
    },

    /// Checked size arithmetic overflowed.
    #[error("size calculation overflowed for {field}")]
    SizeOverflow {
        /// Name of the calculated shape.
        field: &'static str,
    },

    /// A flat array did not match its declared shape.
    #[error("{field} length is {actual}; expected {expected}")]
    ShapeMismatch {
        /// Name of the malformed array.
        field: &'static str,
        /// Expected flat length.
        expected: usize,
        /// Actual flat length.
        actual: usize,
    },

    /// A floating-point input was NaN or infinite.
    #[error("{field}[{index}] must be finite")]
    NonFinite {
        /// Name of the malformed array.
        field: &'static str,
        /// Index of the malformed value.
        index: usize,
    },

    /// Time stamps were not strictly increasing.
    #[error("timestamps must increase strictly at index {index}")]
    NonMonotonicTimestamp {
        /// Index of the later timestamp.
        index: usize,
    },

    /// A schema repeated a feature name.
    #[error("duplicate feature name: {name}")]
    DuplicateFeature {
        /// Repeated feature name.
        name: String,
    },

    /// A quantile was outside the open interval `(0, 1)`.
    #[error("quantile[{index}]={value} must be finite and inside (0, 1)")]
    InvalidQuantile {
        /// Quantile index.
        index: usize,
        /// Invalid quantile value.
        value: f32,
    },

    /// Quantiles were not strictly increasing.
    #[error("quantiles must increase strictly at index {index}")]
    QuantilesNotIncreasing {
        /// Index of the later quantile.
        index: usize,
    },

    /// Predicted quantile values crossed.
    #[error("forecast quantiles cross at flat index {index}")]
    QuantileCrossing {
        /// Index of the later crossing value.
        index: usize,
    },

    /// A series had no observed value for a feature needed by an operation.
    #[error("feature {feature} has no observed value")]
    NoObservedValue {
        /// Feature index.
        feature: usize,
    },

    /// No observed target remained after applying a mask.
    #[error("metric requires at least one observed target")]
    NoObservedTargets,

    /// A metric has no mathematically defined normalization scale.
    #[error("metric is undefined: {reason}")]
    MetricUndefined {
        /// Static reason.
        reason: &'static str,
    },

    /// A receipt used a placeholder all-zero digest.
    #[error("{field} must not be the all-zero digest")]
    ZeroDigest {
        /// Digest field.
        field: &'static str,
    },

    /// A content digest did not match the canonical payload.
    #[error("{field} does not match the canonical payload")]
    DigestMismatch {
        /// Digest field.
        field: &'static str,
    },

    /// A source-state combination was internally inconsistent.
    #[error("invalid source state: {reason}")]
    InvalidSourceState {
        /// Static validation reason.
        reason: &'static str,
    },

    /// A required governance receipt was absent.
    #[error("missing required {field} receipt")]
    MissingReceipt {
        /// Missing receipt field.
        field: &'static str,
    },

    /// A privacy class or governance state is ineligible for a destination.
    #[error("privacy policy denies {operation}: {reason}")]
    PrivacyDenied {
        /// Attempted operation.
        operation: &'static str,
        /// Static denial reason.
        reason: &'static str,
    },

    /// A signature or public key was malformed or did not verify.
    #[error("governance signature verification failed")]
    GovernanceSignatureInvalid,

    /// A verified governance signer was absent from the operator allowlist.
    #[error("governance signer is not operator-allowlisted")]
    GovernanceSignerNotAllowed,

    /// A verifier was configured without any authority key.
    #[error("governance signer allowlist must not be empty")]
    EmptySignerAllowlist,

    /// A derived output attempted to claim stronger evidence than its inputs.
    #[error("output evidence {output:?} exceeds allowed floor {allowed:?}")]
    EvidenceEscalation {
        /// Requested output evidence kind.
        output: crate::SourceKind,
        /// Maximum allowed evidence kind.
        allowed: crate::SourceKind,
    },

    /// A half-open time range was empty or reversed.
    #[error("time range must satisfy start < end")]
    InvalidTimeRange,

    /// A required split partition was empty.
    #[error("{partition} partition must not be empty")]
    EmptyPartition {
        /// Partition name.
        partition: &'static str,
    },

    /// Two split windows overlap within one partition.
    #[error("overlapping windows in {partition} for series {series}")]
    OverlappingWindows {
        /// Partition name.
        partition: &'static str,
        /// Bounded human-readable key.
        series: String,
    },

    /// An entity intended for holdout appeared in multiple partitions.
    #[error("split leakage on {dimension}: {value}")]
    SplitLeakage {
        /// Isolation dimension.
        dimension: &'static str,
        /// Leaked bounded identifier.
        value: String,
    },

    /// Rolling-origin partitions were not chronological for one series.
    #[error("rolling-origin order is invalid for series {series}")]
    SplitOrder {
        /// Bounded human-readable key.
        series: String,
    },

    /// The embargo could not cover the complete target horizon.
    #[error("embargo is {actual_ms} ms; target horizon requires at least {required_ms} ms")]
    EmbargoTooSmall {
        /// Configured embargo.
        actual_ms: u64,
        /// Minimum safe embargo.
        required_ms: u64,
    },

    /// A training specification disagreed with its split plan.
    #[error("training {field}={actual} does not match split plan {expected}")]
    SplitPlanMismatch {
        /// Mismatched dimension.
        field: &'static str,
        /// Value in the training specification.
        actual: u64,
        /// Value in the split plan.
        expected: u64,
    },

    /// A forecast could not be produced with the available history.
    #[error("insufficient history: required {required} rows, found {actual}")]
    InsufficientHistory {
        /// Required rows.
        required: usize,
        /// Available rows.
        actual: usize,
    },

    /// Analogue results crossed a tenant, workspace, or split boundary.
    #[error("analogue result scope does not match its query")]
    RetrievalScopeMismatch,

    /// Analogue results were not ordered deterministically.
    #[error("analogue results must be sorted by (distance, record_id) at index {index}")]
    RetrievalOrder {
        /// Index of the later result.
        index: usize,
    },

    /// Analogue results repeated an opaque record identifier.
    #[error("duplicate analogue record id: {record_id}")]
    DuplicateAnalog {
        /// Repeated opaque identifier.
        record_id: String,
    },
}
