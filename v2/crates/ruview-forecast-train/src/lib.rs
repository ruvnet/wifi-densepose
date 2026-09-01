//! Safe Rust training and fal.ai orchestration for RuView Forecast.
//!
//! The local path can consume governed, hash-addressed RuView data. The fal
//! path is intentionally a different protocol and supports deterministic
//! synthetic pretraining only; no customer data or identity metadata is
//! accepted by its wire types.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod artifact;
pub mod cancel;
pub mod config;
pub mod corpus;

#[cfg(feature = "fal-client")]
pub mod fal;
#[cfg(feature = "training")]
pub mod runner;
#[cfg(feature = "server")]
pub mod server;
