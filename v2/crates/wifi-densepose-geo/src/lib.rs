//! wifi-densepose-geo — geospatial satellite integration for RuView.
//!
//! Provides: IP geolocation, satellite tile fetching (Sentinel-2),
//! SRTM elevation, OSM buildings/roads, coordinate transforms,
//! temporal change tracking, and brain memory integration.

pub mod brain;
pub mod cache;
pub mod coord;
pub mod fuse;
pub mod locate;
pub mod osm;
pub mod register;
pub mod temporal;
pub mod terrain;
pub mod tiles;
pub mod types;

pub use types::*;
