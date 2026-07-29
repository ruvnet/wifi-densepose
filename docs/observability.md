# Observability: OTLP log export

The sensing server can export every `tracing` log event as an
OpenTelemetry log record over OTLP, with a curated set of sensing events
(presence transitions, vitals estimates, node online/offline, fall
detections, CSI capture stats, MQTT errors, model loads) carrying
registry-backed event names and attributes under the `ruview.*`
namespace.

## The event registry

The names are not ad hoc: they are defined in a weaver-validated
semantic-conventions registry at `semconv/registry/` (attributes and log
event names, OpenTelemetry registry format). The Rust constants module
`v2/crates/wifi-densepose-sensing-server/src/semconv.rs` is **generated**
from that registry (`weaver registry generate`, template under
`templates/registry/rust/`) and CI (`.github/workflows/semconv.yml`)
fails if either the registry stops validating or the generated module
drifts. Executed Rust tests additionally reject any hard-coded
`ruview.*` instrumentation key that is absent from the generated registry.
Exported resources carry the registry's schema URL so downstream consumers
can identify the exact conventions version.

Curated events:

| Event | Emitted when |
| --- | --- |
| `ruview.node.online` | first frame from a sensing node (CSI or edge vitals) |
| `ruview.node.offline` | node evicted after 60 s without frames |
| `ruview.presence.changed` | smoothed presence classification flips (transition-only) |
| `ruview.vitals.estimate` | periodic breathing / heart-rate estimate (every 100 ticks) |
| `ruview.fall.detected` | edge-vitals fall flag rising edge, per node |
| `ruview.csi.stats` | periodic capture snapshot: frames processed, active nodes |
| `ruview.mqtt.error` | MQTT publish/connection error in the HA publisher |
| `ruview.model.loaded` | inference model loaded via the model API |

## Enabling export

Export is doubly gated so the default build and the default runtime are
both unaffected:

1. **Build** with the `otel` cargo feature (compiles in the OTLP
   exporter stack, same gating principle as `mqtt`):

   ```sh
   cargo build --release -p wifi-densepose-sensing-server --features mqtt,otel
   ```

2. **Run** with `OTEL_EXPORTER_OTLP_ENDPOINT` set (unset ⇒ the OTLP
   pipeline is never constructed and logging behaves exactly as before):

   ```sh
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
   ./target/release/sensing-server --source simulated
   ```

Use an `https://` collector endpoint outside a trusted local network. The
`otel` feature includes Rustls and native certificate roots; standard OTLP
environment variables can supply authentication headers. The Compose example
uses plaintext only for container-to-container traffic on its private network.

Logs export with resource attribute `service.name = "ruview"` and schema URL
`https://raw.githubusercontent.com/ruvnet/RuView/main/semconv/schema/ruview-0.1.0.yaml`.
Curated sensing
events are emitted only after the configured exporter initializes
successfully; without it, the pre-existing stderr output is unchanged.

## Full stack: `docker compose`

`docker/otel-compose.yml` brings up the whole pipeline —
sensing server (synthetic CSI by default) → OpenTelemetry Collector →
[Ourios](https://github.com/jensholdgaard/ourios), an OTLP-native log
backend built on Parquet + online log-template mining + DataFusion:

```sh
docker compose -f docker/otel-compose.yml up
```

The collector and backend image tags are pinned to immutable multi-platform
digests so the demo resolves to the reviewed images.

Ourios derives the tenant from `service.name`, so all RuView logs land
in tenant `ruview`.

## Example queries

Ourios mines every log line into a stable `template_id` online at
ingest, which makes template-level questions cheap. Its query endpoint
speaks a small logs DSL:

Which log templates dominate RuView's output?

```sh
curl -s http://localhost:4319/v1/query \
  -H 'X-Ourios-Tenant: ruview' \
  -H 'Content-Type: text/plain' \
  -d 'severity >= trace | range(-1h, now) | count by template_id | sort count desc | limit 10'
```

Recent warnings and errors (fall detections, MQTT failures):

```sh
curl -s http://localhost:4319/v1/query \
  -H 'X-Ourios-Tenant: ruview' \
  -H 'Content-Type: text/plain' \
  -d 'severity >= warn | limit 50'
```

Did a RuView deploy change what the service logs? Template drift between
two time windows (new / vanished / changed templates):

```sh
curl -s http://localhost:4319/v1/query \
  -H 'X-Ourios-Tenant: ruview' \
  -H 'Content-Type: text/plain' \
  -d 'drift from -7d to now'
```
