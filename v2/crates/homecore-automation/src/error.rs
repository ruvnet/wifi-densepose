//! Crate-wide error type for homecore-automation.

use thiserror::Error;

use homecore::ServiceError;

#[derive(Error, Debug)]
pub enum AutomationError {
    #[error("YAML parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    #[error("template render error: {0}")]
    TemplateRender(String),

    #[error("service call failed: {0}")]
    ServiceCall(#[from] ServiceError),

    #[error("entity id invalid: {0}")]
    EntityId(#[from] homecore::EntityIdError),

    #[error("automation {id} not found")]
    NotFound { id: String },

    #[error("automation action timed out after {secs}s")]
    ActionTimeout { secs: u64 },

    #[error("numeric state parse error for '{entity_id}': {value}")]
    NumericParse { entity_id: String, value: String },
}
