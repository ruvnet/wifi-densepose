//! Typed, deterministic identifier scheme (ADR-303 §1).
//!
//! Every ontology entity carries a stable, caller-provided string id wrapped in
//! a distinct newtype. Ids are *never* randomly generated here: the ontology is
//! a pure representation, so identity is supplied by the producing surface
//! (ADR-302 `DeviceId`, HomeCore `area_id`, tracker `track_id`, …) and only
//! validated at the crate boundary.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum accepted id length, in bytes. Bounds allocation on untrusted input.
pub const MAX_ID_LEN: usize = 256;

/// Reasons a raw id string is rejected at the boundary.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum IdError {
    /// The id was empty after trimming was *not* applied (empty is invalid).
    #[error("identifier must not be empty")]
    Empty,
    /// The id exceeded [`MAX_ID_LEN`] bytes.
    #[error("identifier length {len} exceeds maximum {max}")]
    TooLong {
        /// Actual length in bytes.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
    /// The id contained an ASCII control character (newline, NUL, …).
    #[error("identifier contains a control character at byte {pos}")]
    ControlChar {
        /// Byte offset of the offending control character.
        pos: usize,
    },
}

/// Validate a raw id string: non-empty, bounded length, no control characters.
pub(crate) fn validate_id(raw: &str) -> Result<(), IdError> {
    if raw.is_empty() {
        return Err(IdError::Empty);
    }
    if raw.len() > MAX_ID_LEN {
        return Err(IdError::TooLong {
            len: raw.len(),
            max: MAX_ID_LEN,
        });
    }
    if let Some(pos) = raw.bytes().position(|b| b.is_ascii_control()) {
        return Err(IdError::ControlChar { pos });
    }
    Ok(())
}

macro_rules! typed_id {
    ($(#[$meta:meta])* $name:ident, $kind:literal) => {
        $(#[$meta])*
        ///
        /// A stable, caller-supplied identifier. Construct with [`Self::new`] to
        /// validate untrusted input; serde round-trips it transparently as a
        /// plain JSON string so it is usable as a canonical map key.
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Construct a validated id, rejecting empty, over-long, or
            /// control-character input at the boundary.
            pub fn new(raw: impl Into<String>) -> Result<Self, IdError> {
                let s = raw.into();
                validate_id(&s)?;
                Ok(Self(s))
            }

            /// Borrow the underlying id string.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// The stable type tag for this id kind (e.g. `"site"`).
            #[must_use]
            pub const fn kind() -> &'static str {
                $kind
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

typed_id!(
    /// Identifier for a [`Site`](crate::Site) — the containment-spine root.
    SiteId, "site"
);
typed_id!(
    /// Identifier for a [`Building`](crate::Building).
    BuildingId, "building"
);
typed_id!(
    /// Identifier for a [`Floor`](crate::Floor).
    FloorId, "floor"
);
typed_id!(
    /// Identifier for a [`Space`](crate::Space) (ADR-294 room / HomeCore area).
    SpaceId, "space"
);
typed_id!(
    /// Identifier for a [`Zone`](crate::Zone) — a sub-region of a space.
    ZoneId, "zone"
);
typed_id!(
    /// Identifier for a [`Sensor`](crate::Sensor) (ADR-302 authenticated device).
    SensorId, "sensor"
);
typed_id!(
    /// Identifier for a [`Person`](crate::Person).
    PersonId, "person"
);
typed_id!(
    /// Identifier for an [`Object`](crate::Object).
    ObjectId, "object"
);
typed_id!(
    /// Identifier for an [`Observation`](crate::Observation).
    ObservationId, "observation"
);
typed_id!(
    /// Identifier for a [`Track`](crate::Track) (ADR-304 persistent track).
    TrackId, "track"
);
typed_id!(
    /// Identifier for an [`Event`](crate::Event) (ADR-315/ADR-316 governed output).
    EventId, "event"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_overlong_and_control() {
        assert_eq!(SiteId::new(""), Err(IdError::Empty));
        let long = "x".repeat(MAX_ID_LEN + 1);
        assert!(matches!(SiteId::new(long), Err(IdError::TooLong { .. })));
        assert!(matches!(
            SiteId::new("a\nb"),
            Err(IdError::ControlChar { pos: 1 })
        ));
    }

    #[test]
    fn kind_tags_are_stable() {
        assert_eq!(SiteId::kind(), "site");
        assert_eq!(ZoneId::kind(), "zone");
        assert_eq!(EventId::kind(), "event");
    }
}
