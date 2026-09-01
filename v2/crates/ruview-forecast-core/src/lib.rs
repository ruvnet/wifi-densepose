//! Backend-neutral multivariate forecasting contracts for RuView.
//!
//! This crate is intentionally free of tensor runtimes, GPU libraries, async
//! runtimes, file I/O, clocks, and network access. Callers inject data and
//! provenance explicitly; constructors validate every shape and finite-value
//! boundary before a model sees it.
//!
//! The API does not claim that a forecast is a detection or authorize an
//! action. [`SourceState`] records whether evidence is synthetic, claimed, or
//! measured, and derived forecasts may not raise that evidence level.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod baseline;
mod digest;
mod error;
mod forecast;
mod metrics;
mod privacy;
mod receipt;
mod retrieval;
mod series;
mod split;

pub use baseline::{LastValueForecaster, SeasonalNaiveForecaster};
pub use digest::CanonicalDigest;
pub use error::ForecastError;
pub use forecast::{
    Abstention, AbstentionReason, Forecast, ForecastOutcome, ForecastRequest, Forecaster,
    QuantileSet, MAX_FORECAST_SPAN_MS, MAX_QUANTILES, MAX_STEP_MS,
};
pub use metrics::{interval_coverage, mae, pinball_loss, weighted_quantile_loss, weighted_quantile_loss_by_horizon};
pub use privacy::{
    DataPolicy, FalGovernanceClaims, FalGovernanceVerifier, PrivacyClass,
    SignedFalGovernanceReceipt, VerifiedFalDataset,
};
pub use receipt::{ArtifactReceipt, ForecastReceipt, SourceKind, SourceState};
pub use retrieval::{
    AnalogMatch, AnalogQuery, AnalogRetriever, RetrievalScope, MAX_ANALOG_DIMENSION, MAX_ANALOG_K,
};
pub use series::{
    FeatureSchema, FeatureSpec, StandardScaler, TimeSeries, MAX_FEATURES, MAX_FEATURE_NAME_LEN,
    MAX_SERIES_ROWS, MAX_SERIES_VALUES, MAX_SOURCE_REFERENCE_LEN, MAX_UNIT_LEN,
};
pub use split::{
    HoldoutKey, NormalizationPolicy, SeriesKey, SplitMember, SplitStrategy, SyntheticFalContract,
    TemporalSplitPlan, TimeRange, TrainSpec, TrainingDestinationKind, MAX_CONTEXT_LENGTH,
    MAX_HORIZON, MAX_SPLIT_MEMBERS,
};
