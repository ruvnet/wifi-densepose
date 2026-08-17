//! `wifi-densepose spaces` — Cognitum Spaces activation and read access.

use std::path::PathBuf;

use clap::Args;
use ruview_auth::{login, scope};
use ruview_cognitum_spaces::{Client, Credential};

#[derive(Debug, Args)]
pub struct SpacesArgs {
    /// Cognitum Spaces API origin.
    #[arg(long, default_value = "https://api.cognitum.one")]
    pub base_url: String,

    /// Compatibility API key. If omitted, use the stored OAuth session.
    #[arg(long, env = "COGNITUM_SPACES_API", hide_env_values = true)]
    pub api_key: Option<String>,

    /// OAuth credential file used when no API key is supplied.
    #[arg(long, env = ruview_auth::login::CREDENTIALS_PATH_ENV)]
    pub credentials_path: Option<PathBuf>,

    /// Emit the validated response as JSON.
    #[arg(long)]
    pub json: bool,
}

pub async fn spaces_cmd(args: SpacesArgs) -> anyhow::Result<()> {
    let credential = match args.api_key {
        Some(key) => Credential::api_key(key)?,
        None => {
            let path = args
                .credentials_path
                .unwrap_or_else(login::default_credentials_path);
            let session = login::Session::load_from(path, reqwest::Client::new())?;
            let snapshot = session.snapshot().await;
            let granted = snapshot.effective_scope().unwrap_or_default();
            if !granted
                .split_whitespace()
                .any(|item| item == scope::SPACES_READ)
            {
                anyhow::bail!(
                    "stored OAuth session lacks spaces:read; run `wifi-densepose login --spaces`"
                );
            }
            Credential::oauth(session.ensure_fresh().await?)?
        }
    };
    let response = Client::new(&args.base_url, credential)?.list().await?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&response)?);
        return Ok(());
    }
    println!("Cognitum Spaces: {}", response.data.len());
    println!(
        "Boundary: {} / {}",
        response.boundary.authoritative_state, response.boundary.cloud_role
    );
    for space in response.data {
        let occupancy = space
            .state
            .occupancy
            .map_or_else(|| "unknown".into(), |v| v.to_string());
        let confidence = space
            .state
            .confidence
            .map_or_else(|| "unknown".into(), |v| format!("{v:.3}"));
        println!(
            "{}\t{}\toccupancy={}\tconfidence={}\t{}",
            space.id, space.name, occupancy, confidence, space.status
        );
    }
    Ok(())
}
