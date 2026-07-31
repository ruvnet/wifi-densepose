//! The authenticated caller, and the scopes it consented to.

use std::collections::BTreeSet;

/// RuView's own scopes, registered on the `ruview` OAuth client
/// (identity migration `0016`, ADR-060).
///
/// Split by **blast radius**, not by endpoint count: the question is whether a
/// leaked token can destroy something, not how many routes it covers.
pub mod scope {
    /// Observe: sensing/pose streams, one-shot inference, reading metadata.
    ///
    /// Not "harmless" — for a presence and vital-signs sensor, read access tells
    /// the holder who is home. It is *non-destructive*, which is a weaker claim.
    pub const SENSING_READ: &str = "sensing:read";

    /// Mutate or destroy: training, model delete, recording delete.
    ///
    /// Irreversible: a deleted model or labelled capture may represent days of
    /// collection, and a training run burns hours of CPU on a Pi.
    pub const SENSING_ADMIN: &str = "sensing:admin";
}

/// A verified caller. Constructed only by
/// [`crate::verify::verify_access_token`] — there is deliberately no public
/// constructor, so a `Principal` in hand always means a signature was checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    /// `sub` — the identity user id.
    pub subject: String,
    /// `account_id` — the billing tenant (the user's Firebase UID; ADR-045).
    /// Required and non-empty; see the verifier for why.
    pub account_id: String,
    pub org_id: String,
    pub workspace_id: String,
    /// `client_id` — which OAuth client obtained this token.
    ///
    /// **Attribution and logging only — never an authorization input.** Clients
    /// borrow each other's registrations when their own has not been deployed
    /// yet (musica ships `DEFAULT_CLIENT_ID = "meta-proxy"`), so this claim does
    /// not reliably identify the product holding the token.
    pub client_id: String,
    /// `jti` — unique per token; use for request-log correlation.
    pub token_id: String,
    /// The consented scopes, split on whitespace.
    scopes: BTreeSet<String>,
    /// `exp`, unix seconds.
    pub expires_at: i64,
}

impl Principal {
    pub(crate) fn new(
        subject: String,
        account_id: String,
        org_id: String,
        workspace_id: String,
        client_id: String,
        token_id: String,
        scope_claim: &str,
        expires_at: i64,
    ) -> Self {
        Self {
            subject,
            account_id,
            org_id,
            workspace_id,
            client_id,
            token_id,
            scopes: scope_claim.split_whitespace().map(str::to_owned).collect(),
            expires_at,
        }
    }

    /// Does this principal hold `scope`?
    ///
    /// Exact match only. There is **no prefix or hierarchy rule** — holding
    /// `sensing:admin` does not imply `sensing:read`, and a token that needs
    /// both must have consented to both. Implying one scope from another is how
    /// a consent screen ends up meaning less than it said.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.contains(scope)
    }

    /// Scopes, sorted — for logging.
    pub fn scopes(&self) -> impl Iterator<Item = &str> {
        self.scopes.iter().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal_with(scope: &str) -> Principal {
        Principal::new(
            "sub".into(),
            "acct".into(),
            "org".into(),
            "ws".into(),
            "ruview".into(),
            "jti".into(),
            scope,
            0,
        )
    }

    #[test]
    fn has_scope_matches_a_single_consented_scope() {
        assert!(principal_with("sensing:read").has_scope(scope::SENSING_READ));
    }

    #[test]
    fn has_scope_matches_within_a_whitespace_separated_list() {
        let p = principal_with("sensing:read sensing:admin");
        assert!(p.has_scope(scope::SENSING_READ));
        assert!(p.has_scope(scope::SENSING_ADMIN));
    }

    #[test]
    fn admin_does_not_imply_read() {
        // Guards the "no hierarchy" rule above. If someone later adds prefix
        // matching to be helpful, this fails and they have to read the comment.
        assert!(!principal_with("sensing:admin").has_scope(scope::SENSING_READ));
    }

    #[test]
    fn unrelated_scope_grants_nothing() {
        let p = principal_with("inference");
        assert!(!p.has_scope(scope::SENSING_READ));
        assert!(!p.has_scope(scope::SENSING_ADMIN));
    }

    #[test]
    fn empty_scope_claim_grants_nothing() {
        assert!(!principal_with("").has_scope(scope::SENSING_READ));
    }

    #[test]
    fn scope_prefix_of_a_real_scope_does_not_match() {
        assert!(!principal_with("sensing").has_scope(scope::SENSING_READ));
    }
}
