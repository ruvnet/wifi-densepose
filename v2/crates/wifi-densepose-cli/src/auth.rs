//! `wifi-densepose login` / `logout` / `whoami` — Cognitum sign-in (ADR-271).
//!
//! Signing in yields a Cognitum access token that a RuView sensing server
//! verifies offline against `auth.cognitum.one`'s published JWKS. It replaces
//! sharing one `RUVIEW_API_TOKEN` string between everyone who needs access:
//! requests become attributable to a person, and destructive routes can be
//! separated from read-only ones by scope.

use std::path::PathBuf;

use clap::Args;
use ruview_auth::login::{self, LoginOptions};
use ruview_auth::scope;

#[derive(Debug, Args)]
pub struct LoginArgs {
    /// Also request `sensing:admin` — the capability to train models and delete
    /// models and recordings.
    ///
    /// Off by default on purpose. A session that only streams poses has no
    /// business holding delete capability, and a token that carries it is a
    /// bigger loss if it leaks. Ask for it when you are about to do
    /// administrative work, not as a matter of habit.
    #[arg(long)]
    pub admin: bool,

    /// Skip the browser and use the paste-a-code flow.
    ///
    /// Detected automatically over SSH and inside containers; this forces it.
    #[arg(long)]
    pub no_browser: bool,

    /// Where to store credentials. Defaults to `~/.ruview/credentials.json`.
    #[arg(long, env = ruview_auth::login::CREDENTIALS_PATH_ENV)]
    pub credentials_path: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct LogoutArgs {
    #[arg(long, env = ruview_auth::login::CREDENTIALS_PATH_ENV)]
    pub credentials_path: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct WhoamiArgs {
    #[arg(long, env = ruview_auth::login::CREDENTIALS_PATH_ENV)]
    pub credentials_path: Option<PathBuf>,

    /// Refresh the access token now if it has expired, instead of only
    /// reporting that it will be refreshed on next use.
    ///
    /// Refreshing rotates the stored refresh token — identity spends the old
    /// one — so this is a real state change, not a read. That is why it is a
    /// flag rather than something `whoami` does silently.
    #[arg(long)]
    pub refresh: bool,
}

fn path_or_default(p: Option<PathBuf>) -> PathBuf {
    p.unwrap_or_else(login::default_credentials_path)
}

/// What `login` asks the authorization server for.
///
/// Extracted so it is testable on its own. `LoginOptions::default()` has its own
/// least-privilege test in the library, but this command does NOT go through
/// that default — it builds the scope string itself, so the library test says
/// nothing about what the CLI actually requests.
fn requested_scope(admin: bool) -> String {
    if admin {
        // Admin implies read: there is no scope hierarchy server-side, so a
        // session that needs both must consent to both explicitly.
        format!("{} {}", scope::SENSING_READ, scope::SENSING_ADMIN)
    } else {
        scope::SENSING_READ.to_string()
    }
}

pub async fn login_cmd(args: LoginArgs) -> anyhow::Result<()> {
    let scope = requested_scope(args.admin);

    let opts = LoginOptions {
        credentials_path: path_or_default(args.credentials_path),
        scope,
        no_browser: args.no_browser,
    };

    let mut out = std::io::stdout();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();

    login::login(&opts, &mut out, &mut input).await?;
    Ok(())
}

pub async fn logout_cmd(args: LogoutArgs) -> anyhow::Result<()> {
    let path = path_or_default(args.credentials_path);
    if login::logout(&path)? {
        println!("Signed out — {} removed.", path.display());
    } else {
        println!("Not signed in; nothing to remove.");
    }
    // Deliberately local-only. This makes the machine unable to act as you;
    // revoking the session for every device is an account-level action.
    println!("Note: this forgets the local credential only. It does not revoke the session server-side.");
    Ok(())
}

pub async fn whoami_cmd(args: WhoamiArgs) -> anyhow::Result<()> {
    let path = path_or_default(args.credentials_path);
    let mut creds = ruview_auth::login::store::load(&path)?;

    if args.refresh && creds.needs_refresh() {
        println!("Access token expired — refreshing…");
        let session = ruview_auth::login::Session::load_from(path.clone(), reqwest::Client::new())?;
        // Goes through ensure_fresh, so it inherits the single-flight guarantee
        // and the persist-before-return ordering rather than reimplementing a
        // second, subtly different refresh path.
        session.ensure_fresh().await?;
        creds = session.snapshot().await;
        println!("Refreshed.\n");
    }

    println!("Credentials: {}", path.display());
    println!("Issuer:      {}", creds.issuer);
    match &creds.account_email {
        Some(e) => println!("Account:     {e}"),
        None => println!("Account:     (not reported)"),
    }
    // Falls back to the token's own claim, so a file written before that
    // fallback existed still reports its real scope.
    match creds.effective_scope() {
        Some(s) => println!("Scope:       {s}"),
        None => println!("Scope:       (not reported)"),
    }
    // State, not just contents: an expired-looking session is the single most
    // common reason a command starts 401ing, so say it plainly here rather than
    // letting the user infer it from a failure elsewhere.
    if creds.needs_refresh() {
        println!("Status:      access token expired or expiring — pass --refresh to renew it now");
    } else {
        println!("Status:      access token valid");
    }
    if creds.refresh_token.is_none() {
        println!("Warning:     no refresh token stored; you will need to sign in again when this expires");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_login_asks_for_read_only() {
        // The whole point of splitting the scopes (ADR-060) is that streaming
        // poses must not carry the capability to delete recordings. If this
        // ever returns admin by default, every session silently becomes
        // destructive-capable and nothing else in the suite would notice.
        let s = requested_scope(false);
        assert_eq!(s, scope::SENSING_READ);
        assert!(!s.contains(scope::SENSING_ADMIN), "read-only login leaked admin: {s}");
    }

    #[test]
    fn admin_login_asks_for_both_because_there_is_no_hierarchy() {
        // The authorization server grants exactly what is requested; admin does
        // not imply read. Asking for admin alone would produce a session that
        // cannot stream.
        let s = requested_scope(true);
        assert!(s.split_whitespace().any(|x| x == scope::SENSING_READ), "{s}");
        assert!(s.split_whitespace().any(|x| x == scope::SENSING_ADMIN), "{s}");
    }

    #[test]
    fn an_explicit_credentials_path_is_honoured_over_the_default() {
        // `--credentials-path` also carries the RUVIEW_CREDENTIALS_PATH env
        // binding; silently ignoring it would write credentials somewhere the
        // operator did not choose.
        let p = PathBuf::from("/tmp/ruview-cli-explicit-credentials.json");
        assert_eq!(path_or_default(Some(p.clone())), p);
        assert_eq!(path_or_default(None), login::default_credentials_path());
    }
}
