//! Tracing bootstrap, with optional OTLP log export.
//!
//! Without the `otel` cargo feature (the default) this is exactly the
//! stderr `tracing_subscriber::fmt()` setup the server has always had.
//! With the feature, and only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set
//! in the environment, [`init`] additionally installs an
//! `opentelemetry-appender-tracing` bridge so every `tracing` event is
//! exported as an OpenTelemetry log record (`service.name = "ruview"`)
//! to that endpoint over OTLP/gRPC. Endpoint unset means the OTLP
//! pipeline is never constructed — zero overhead.
//!
//! The curated events named in [`crate::semconv`] carry registry-backed
//! event names and attribute keys (see `semconv/registry/` at the repo
//! root); everything else exports under tracing's default event names.

/// Service name reported in the OTLP resource. Log backends that derive
/// a tenant from `service.name` file all RuView logs under it.
#[cfg(feature = "otel")]
const SERVICE_NAME: &str = "ruview";

/// The server's long-standing default log filter.
const DEFAULT_FILTER: &str = "info,tower_http=debug";

/// Owns the OTLP logger pipeline when one was installed. Hold it for the
/// process lifetime; dropping it flushes pending log records.
pub struct TelemetryGuard {
    #[cfg(feature = "otel")]
    logger: Option<opentelemetry_sdk::logs::SdkLoggerProvider>,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        #[cfg(feature = "otel")]
        if let Some(logger) = &self.logger {
            let _ = logger.shutdown();
        }
    }
}

/// Install the global `tracing` subscriber. Call once, at start-up,
/// inside the tokio runtime (the OTLP exporter runs on it).
#[must_use = "dropping the guard tears the OTLP log pipeline down"]
pub fn init() -> TelemetryGuard {
    #[cfg(feature = "otel")]
    if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some() {
        match init_with_otlp() {
            Ok(logger) => {
                return TelemetryGuard {
                    logger: Some(logger),
                };
            }
            Err(e) => eprintln!("OTLP log export disabled (exporter build failed): {e}"),
        }
    }

    init_fmt_only();
    TelemetryGuard {
        #[cfg(feature = "otel")]
        logger: None,
    }
}

fn init_fmt_only() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| DEFAULT_FILTER.into()),
        )
        .init();
}

#[cfg(feature = "otel")]
fn init_with_otlp(
) -> Result<opentelemetry_sdk::logs::SdkLoggerProvider, opentelemetry_otlp::ExporterBuildError> {
    use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
    use opentelemetry_otlp::LogExporter;
    use opentelemetry_sdk::logs::SdkLoggerProvider;
    use opentelemetry_sdk::Resource;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;
    use tracing_subscriber::{EnvFilter, Layer as _};

    // The exporter reads OTEL_EXPORTER_OTLP_ENDPOINT (and the other
    // OTEL_EXPORTER_* variables) from the environment itself.
    let exporter = LogExporter::builder().with_tonic().build()?;
    let logger = SdkLoggerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name(SERVICE_NAME).build())
        .build();

    // Telemetry-induced-telemetry loop guard: the OTLP exporter is itself
    // a tonic/hyper client, so its internal tracing events must not
    // re-enter the bridge (a failed export would emit records that
    // trigger more exports). The `off` directives win over RUST_LOG.
    let mut bridge_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    for directive in [
        "hyper=off",
        "tonic=off",
        "h2=off",
        "tower=off",
        "opentelemetry=off",
        "opentelemetry_sdk=off",
        "opentelemetry_otlp=off",
    ] {
        if let Ok(directive) = directive.parse() {
            bridge_filter = bridge_filter.add_directive(directive);
        }
    }
    let bridge = OpenTelemetryTracingBridge::new(&logger).with_filter(bridge_filter);
    let fmt = tracing_subscriber::fmt::layer().with_filter(
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER)),
    );
    tracing_subscriber::registry().with(bridge).with(fmt).init();
    Ok(logger)
}
