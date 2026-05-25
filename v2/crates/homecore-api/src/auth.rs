//! Bearer-token auth helper. P1 accepts any non-empty bearer; P2 wires
//! in the long-lived token store. Mirrors HA's
//! `Authorization: Bearer <token>` convention.

use axum::http::HeaderMap;
use crate::error::ApiError;

#[derive(Clone, Debug)]
pub struct BearerAuth(pub String);

impl BearerAuth {
    /// Parse the `Authorization: Bearer <token>` header out of the
    /// request. Returns `ApiError::Unauthorized` if missing, malformed,
    /// or the token is empty.
    pub fn from_headers(headers: &HeaderMap) -> Result<Self, ApiError> {
        let header = headers
            .get(axum::http::header::AUTHORIZATION)
            .ok_or(ApiError::Unauthorized)?;
        let value = header.to_str().map_err(|_| ApiError::Unauthorized)?;
        let token = value
            .strip_prefix("Bearer ")
            .ok_or(ApiError::Unauthorized)?
            .trim()
            .to_string();
        if token.is_empty() {
            return Err(ApiError::Unauthorized);
        }
        Ok(Self(token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::AUTHORIZATION;

    #[test]
    fn strips_bearer_prefix() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, "Bearer abc123".parse().unwrap());
        let a = BearerAuth::from_headers(&h).unwrap();
        assert_eq!(a.0, "abc123");
    }

    #[test]
    fn rejects_missing_prefix() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, "abc123".parse().unwrap());
        assert!(matches!(BearerAuth::from_headers(&h), Err(ApiError::Unauthorized)));
    }

    #[test]
    fn rejects_missing_header() {
        let h = HeaderMap::new();
        assert!(matches!(BearerAuth::from_headers(&h), Err(ApiError::Unauthorized)));
    }

    #[test]
    fn rejects_empty_token() {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, "Bearer   ".parse().unwrap());
        assert!(matches!(BearerAuth::from_headers(&h), Err(ApiError::Unauthorized)));
    }
}
