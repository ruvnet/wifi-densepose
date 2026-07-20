//! WiFi-DensePose CLI Entry Point
//!
//! This is the main entry point for the wifi-densepose command-line tool.

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use wifi_densepose_cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Calibrate(args) => {
            wifi_densepose_cli::calibrate::execute(args).await?;
        }
        Commands::CalibrateServe(args) => {
            wifi_densepose_cli::calibrate_api::execute(args).await?;
        }
        Commands::Enroll(args) => {
            wifi_densepose_cli::room::enroll(args).await?;
        }
        Commands::TrainRoom(args) => {
            wifi_densepose_cli::room::train_room(args).await?;
        }
        Commands::RoomStatus(args) => {
            wifi_densepose_cli::room::room_status(args).await?;
        }
        Commands::RoomWatch(args) => {
            wifi_densepose_cli::room::room_watch(args).await?;
        }
        #[cfg(feature = "mat")]
        Commands::Mat(mat_cmd) => {
            wifi_densepose_cli::mat::execute(mat_cmd).await?;
        }
        Commands::Version => {
            println!("wifi-densepose {}", env!("CARGO_PKG_VERSION"));
            #[cfg(feature = "mat")]
            println!("MAT module version: {}", wifi_densepose_mat::VERSION);
        }
    }

    Ok(())
}
