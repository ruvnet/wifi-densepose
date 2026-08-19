//! Login orchestration: browser + loopback when possible, OOB paste when not.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Duration;

use super::callback::{looks_headless, open_browser, CallbackServer};
use super::client::{self, OAuthError};
use crate::pkce;
use super::store::{self, Session, StoreError};
use crate::scope;

/// How long to wait for the user to finish in the browser.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, thiserror::Error)]
pub enum LoginError {
    #[error(transparent)]
    OAuth(#[from] OAuthError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("could not bind a loopback callback listener: {0}")]
    Bind(#[source] std::io::Error),
    #[error("waiting for the browser callback failed: {0}")]
    Callback(#[source] std::io::Error),
    #[error("the authorization server returned state {got:?}, expected {expected:?} — this login was not the one you started, so it was discarded")]
    StateMismatch { expected: String, got: String },
    #[error("the authorization server reported: {0}")]
    Denied(String),
    #[error("login cancelled")]
    Cancelled,
    #[error("could not read from the terminal: {0}")]
    Io(#[from] std::io::Error),
}

pub struct LoginOptions {
    /// Where to persist credentials.
    pub credentials_path: PathBuf,
    /// Scopes to request. Least privilege by default — `sensing:read` only.
    pub scope: String,
    /// Force the OOB paste flow even if a browser looks available.
    pub no_browser: bool,
}

impl Default for LoginOptions {
    fn default() -> Self {
        Self {
            credentials_path: store::default_credentials_path(),
            // A client registration is a ceiling, not a default (ADR-060 §5).
            // Routine use asks for read; admin is an explicit escalation.
            scope: scope::SENSING_READ.to_string(),
            no_browser: false,
        }
    }
}

/// Run the login flow and persist the resulting session.
///
/// `out` receives the human-facing prose (URLs, prompts) so a caller can
/// capture it in tests; `input` supplies the pasted code in the OOB path.
pub async fn login<W: Write, R: BufRead>(
    opts: &LoginOptions,
    out: &mut W,
    input: &mut R,
) -> Result<Session, LoginError> {
    let http = reqwest::Client::new();
    let issuer = client::auth_base_url();

    if opts.no_browser || looks_headless() {
        return manual_login(opts, &http, issuer, out, input).await;
    }

    match browser_login(opts, &http, issuer.clone(), out).await {
        Ok(s) => Ok(s),
        // A loopback bind failure is environmental, not user error — fall back
        // rather than dead-ending someone who is one paste away from success.
        Err(LoginError::Bind(e)) => {
            writeln!(
                out,
                "Could not open a local callback listener ({e}); falling back to paste-code sign-in.\n"
            )?;
            manual_login(opts, &http, issuer, out, input).await
        }
        Err(e) => Err(e),
    }
}

async fn browser_login<W: Write>(
    opts: &LoginOptions,
    http: &reqwest::Client,
    issuer: String,
    out: &mut W,
) -> Result<Session, LoginError> {
    let server = CallbackServer::bind().await.map_err(LoginError::Bind)?;
    let req = pkce::generate();
    let url = client::authorize_url(
        &server.redirect_uri,
        &req.state,
        &req.code_challenge,
        &opts.scope,
    );

    writeln!(out, "Opening your browser to sign in to Cognitum…")?;
    writeln!(out, "If it doesn't open, visit:\n\n  {url}\n")?;
    // Best-effort: the URL is already printed, so a missing launcher is not fatal.
    let _ = open_browser(&url);

    let cb = server
        .await_callback(CALLBACK_TIMEOUT)
        .await
        .map_err(LoginError::Callback)?;

    if let Some(err) = cb.error {
        return Err(LoginError::Denied(err));
    }
    // CSRF check before the code is spent: a code arriving with the wrong state
    // did not come from the flow we started.
    let got = cb.state.unwrap_or_default();
    if got != req.state {
        return Err(LoginError::StateMismatch {
            expected: req.state,
            got,
        });
    }
    let code = cb.code.ok_or(LoginError::Cancelled)?;

    let token = client::exchange_code(http, &code, &req.code_verifier, &server.redirect_uri).await?;
    finish(opts, http, token, issuer, out)
}

async fn manual_login<W: Write, R: BufRead>(
    opts: &LoginOptions,
    http: &reqwest::Client,
    issuer: String,
    out: &mut W,
    input: &mut R,
) -> Result<Session, LoginError> {
    let req = pkce::generate();
    let url = client::authorize_url(
        client::OOB_REDIRECT_URI,
        &req.state,
        &req.code_challenge,
        &opts.scope,
    );

    writeln!(
        out,
        "No local browser available (SSH/container detected, or --no-browser).\n"
    )?;
    writeln!(
        out,
        "Open this URL in a browser on any machine and authorize:\n\n  {url}\n"
    )?;
    write!(out, "Paste the code shown after authorizing: ")?;
    out.flush()?;

    let mut line = String::new();
    input.read_line(&mut line)?;
    let code = line.trim();
    if code.is_empty() {
        return Err(LoginError::Cancelled);
    }

    let token = client::exchange_manual_code(http, code, &req.code_verifier).await?;
    finish(opts, http, token, issuer, out)
}

fn finish<W: Write>(
    opts: &LoginOptions,
    http: &reqwest::Client,
    token: client::TokenResponse,
    issuer: String,
    out: &mut W,
) -> Result<Session, LoginError> {
    let granted = token.scope.clone();
    let email = token.account_email.clone();
    let session = Session::from_response(
        opts.credentials_path.clone(),
        http.clone(),
        token,
        issuer,
    )?;

    writeln!(out)?;
    match email {
        Some(e) => writeln!(out, "Signed in as {e}.")?,
        None => writeln!(out, "Signed in.")?,
    }
    // Report what the server actually granted, not what we asked for. They can
    // differ, and a user who thinks they hold `sensing:admin` when they don't
    // will read the eventual 401 as a bug.
    match granted {
        Some(s) => writeln!(out, "Granted scope: {s}")?,
        None => writeln!(out, "Granted scope: (not reported by the server)")?,
    }
    writeln!(
        out,
        "Credentials saved to {}",
        opts.credentials_path.display()
    )?;
    Ok(session)
}

/// Forget the local session. Returns whether anything was removed.
///
/// Local-only by design: this makes the machine unable to act as you. Revoking
/// server-side is a separate, account-level action.
pub fn logout(credentials_path: &std::path::Path) -> Result<bool, StoreError> {
    store::clear(credentials_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_scope_is_read_only() {
        // ADR-060 §5: a registration is a ceiling, not a default. A session that
        // streams poses must not casually hold delete capability.
        assert_eq!(LoginOptions::default().scope, scope::SENSING_READ);
        assert_ne!(LoginOptions::default().scope, scope::SENSING_ADMIN);
    }

    #[test]
    fn logout_on_a_machine_that_never_logged_in_is_not_an_error() {
        let p = std::env::temp_dir().join("ruview-flow-absent-credentials.json");
        let _ = std::fs::remove_file(&p);
        assert_eq!(logout(&p).unwrap(), false);
    }

    #[test]
    fn a_state_mismatch_names_both_values_so_it_can_be_diagnosed() {
        let e = LoginError::StateMismatch {
            expected: "aaa".into(),
            got: "bbb".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("aaa") && msg.contains("bbb"), "{msg}");
        assert!(msg.contains("discarded"), "must say the login was refused: {msg}");
    }
}
