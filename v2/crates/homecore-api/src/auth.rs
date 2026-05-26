//! Bearer-token auth helper. Validates against the
//! [`LongLivedTokenStore`] on `SharedState` (audit fix HC-01/02).
//!
//! - P1 placeholder accepted any non-empty bearer
//! - P2 (this commit) requires the token to be present in the store
//! - DEV escape hatch: `LongLivedTokenStore::allow_any_non_empty()`
//!   preserves the legacy behaviour for users mid-migration, with
//!   a warn log on every check

use axum::http::HeaderMap;
use crate::error::ApiError;
use crate::tokens::LongLivedTokenStore;

#[derive(Clone, Debug)]
pub struct BearerAuth(pub String);

impl BearerAuth {
    /// Parse the `Authorization: Bearer <token>` header out of the
    /// request AND validate it against the supplied token store.
    /// Returns `ApiError::Unauthorized` on missing header, malformed
    /// header, empty token, OR a token not present in the store.
    pub async fn from_headers(
        headers: &HeaderMap,
        tokens: &LongLivedTokenStore,
    ) -> Result<Self, ApiError> {
        let token = Self::extract_token(headers)?;
        if !tokens.is_valid(&token).await {
            return Err(ApiError::Unauthorized);
        }
        Ok(Self(token))
    }

    /// Extract the bearer token from headers without validating it.
    /// Used by the WS handshake which validates inline.
    pub fn extract_token(headers: &HeaderMap) -> Result<String, ApiError> {
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
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::AUTHORIZATION;

    fn mkheaders(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, value.parse().unwrap());
        h
    }

    #[test]
    fn extract_strips_bearer_prefix() {
        let h = mkheaders("Bearer abc123");
        assert_eq!(BearerAuth::extract_token(&h).unwrap(), "abc123");
    }

    #[test]
    fn extract_rejects_missing_prefix() {
        let h = mkheaders("abc123");
        assert!(matches!(BearerAuth::extract_token(&h), Err(ApiError::Unauthorized)));
    }

    #[test]
    fn extract_rejects_missing_header() {
        let h = HeaderMap::new();
        assert!(matches!(BearerAuth::extract_token(&h), Err(ApiError::Unauthorized)));
    }

    #[test]
    fn extract_rejects_empty_token() {
        let h = mkheaders("Bearer   ");
        assert!(matches!(BearerAuth::extract_token(&h), Err(ApiError::Unauthorized)));
    }

    #[tokio::test]
    async fn from_headers_accepts_registered_token() {
        let store = LongLivedTokenStore::empty();
        store.register("good_token").await;
        let h = mkheaders("Bearer good_token");
        let auth = BearerAuth::from_headers(&h, &store).await.unwrap();
        assert_eq!(auth.0, "good_token");
    }

    #[tokio::test]
    async fn from_headers_rejects_unregistered_token() {
        let store = LongLivedTokenStore::empty();
        store.register("good_token").await;
        let h = mkheaders("Bearer wrong_token");
        assert!(matches!(BearerAuth::from_headers(&h, &store).await, Err(ApiError::Unauthorized)));
    }

    #[tokio::test]
    async fn dev_mode_still_accepts_any_non_empty() {
        let store = LongLivedTokenStore::allow_any_non_empty();
        let h = mkheaders("Bearer literally-anything");
        assert!(BearerAuth::from_headers(&h, &store).await.is_ok());
    }

    #[tokio::test]
    async fn dev_mode_still_rejects_empty() {
        let store = LongLivedTokenStore::allow_any_non_empty();
        let h = mkheaders("Bearer ");
        assert!(matches!(BearerAuth::from_headers(&h, &store).await, Err(ApiError::Unauthorized)));
    }
}
