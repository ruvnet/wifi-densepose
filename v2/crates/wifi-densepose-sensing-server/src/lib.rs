//! WiFi-DensePose Sensing Server library.
//!
//! This crate provides:
//! - Vital sign detection from WiFi CSI amplitude data
//! - RVF (RuVector Format) binary container for model weights
//! - Opt-in bearer-token auth for the `/api/v1/*` HTTP surface (`bearer_auth`)
//! - Host-header allowlist / DNS-rebinding defense (`host_validation`)
//! - Generic, leak-free internal-error responses (`error_response`, ADR-080 #2)
//! - Real-time CSI introspection / low-latency tap (`introspection`, ADR-099)

pub mod bearer_auth;
pub mod cli;
pub mod dataset;
pub mod edge_registry;
#[allow(dead_code)]
pub mod embedding;
pub mod error_response;
pub mod graph_transformer;
pub mod host_validation;
pub mod introspection;
pub mod matter;
pub mod model_format;
pub mod mqtt;
pub mod path_safety;
pub mod semantic;
/// ADR-262 P3: the live RuField surface — turns the governed sensing cycle into
/// signed RuField `FieldEvent`s on the additive `/api/field` + `/ws/field`
/// endpoints, via the `wifi-densepose-rufield` anti-corruption bridge.
pub mod rufield_surface;
pub mod rvf_container;
pub mod rvf_pipeline;
pub mod sona;
pub mod sparse_inference;
#[allow(dead_code)]
pub mod trainer;
pub mod vital_signs;
/// ADR-270 Mist and NETGEAR telemetry providers.
pub mod vendor_mist_netgear;
/// ADR-270 Origin AI and Plume/OpenSync providers.
pub mod vendor_origin_plume;
/// ADR-270 scalar, network-only, and fail-closed vendor providers.
pub mod vendor_remaining;
/// ADR-270 provider registry and canonical event helpers.
pub mod vendor_rf;
