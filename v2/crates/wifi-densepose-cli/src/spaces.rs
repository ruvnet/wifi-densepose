//! `wifi-densepose spaces` — Cognitum Spaces activation and read access.

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use ruview_auth::{login, scope};
use ruview_cognitum_spaces::{Client, Credential, PageRequest, SpatialKind};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SpatialResourceKind {
    Sites,
    Buildings,
    Floors,
    Spaces,
    Zones,
    Entities,
    Events,
    Alerts,
}

impl From<SpatialResourceKind> for SpatialKind {
    fn from(value: SpatialResourceKind) -> Self {
        match value {
            SpatialResourceKind::Sites => Self::Sites,
            SpatialResourceKind::Buildings => Self::Buildings,
            SpatialResourceKind::Floors => Self::Floors,
            SpatialResourceKind::Spaces => Self::Spaces,
            SpatialResourceKind::Zones => Self::Zones,
            SpatialResourceKind::Entities => Self::Entities,
            SpatialResourceKind::Events => Self::Events,
            SpatialResourceKind::Alerts => Self::Alerts,
        }
    }
}

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

    /// Versioned hierarchy/event/alert collection. Omit for the legacy flat projection.
    #[arg(long, value_enum)]
    pub resource: Option<SpatialResourceKind>,

    /// Page size for a versioned resource collection (1..=100).
    #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u8).range(1..=100), requires = "resource")]
    pub limit: u8,

    /// Opaque next-page cursor returned by a prior versioned read.
    #[arg(long, requires = "resource")]
    pub cursor: Option<String>,

    /// API-key compatibility only: exact workspace UUID. OAuth derives this from its signed token.
    #[arg(long, requires = "resource")]
    pub workspace_id: Option<String>,

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
    let client = Client::new(&args.base_url, credential)?;
    if let Some(resource) = args.resource {
        let response = client
            .list_spatial(
                resource.into(),
                &PageRequest {
                    limit: args.limit,
                    cursor: args.cursor,
                    workspace_id: args.workspace_id,
                },
            )
            .await?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&response)?);
            return Ok(());
        }
        println!(
            "Cognitum Spatial {}: {}",
            response.kind.as_str(),
            response.data.len()
        );
        println!(
            "Boundary: {} / {}",
            response.boundary.authoritative_state, response.boundary.cloud_role
        );
        for item in response.data {
            println!(
                "{}\tkind={}\tprivacy={}\tsite={}\tspace={}",
                item.id,
                item.kind.as_str(),
                item.privacy,
                item.site_id.as_deref().unwrap_or("-"),
                item.space_id.as_deref().unwrap_or("-")
            );
        }
        if let Some(cursor) = response.next_cursor {
            println!("Next cursor: {cursor}");
        }
        return Ok(());
    }
    let response = client.list().await?;
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
