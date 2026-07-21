# USRP X310 RF-Direct Integration

This path uses the USRP as the RF front end without pretending that RF peaks are
human detections. OpenISAC produces range-Doppler data plus a metadata sidecar;
the bridge pairs both halves by frame ID and sends a versioned `RfObservation`
to RuView. RuView exposes diagnostics and CFAR candidate clusters only. Motion,
presence, person count, pose, vitals, and MQTT automation remain disabled.

## Recommended Topology

Use this topology when the laptop has no SFP/SFP+ NIC and the X310 is attached
to another host.

```text
USRP X310 <SFP/Ethernet> isac (UHD + OpenISAC + bridge + RuView server)
                                      |
                                      | HTTP/WebSocket/SSH
                                      v
                               laptop browser/control
```

Run UHD, the RF worker, and `sensing-server` on `isac`. Use the laptop only for
SSH and the browser. This avoids hauling raw IQ across an extra network hop.

Your probe indicates:

- UHD `4.9.0.HEAD-0-g9ec1f582`
- X310 serial `31167CA`, FPGA `39.3`, FW `6.1`
- internal GPSDO available (`gpsdo` clock/time source)
- UBX-160 daughterboards on both radios
- X310 addresses: `192.168.10.2`, `192.168.20.2`, `192.168.30.2`,
  `192.168.40.2`
- usable RF range: `10 MHz` to `6 GHz`

## Tune The USRP Host

UHD reported Linux UDP socket buffers that are too small. Apply this on `isac`
before high-rate work:

```bash
sudo sysctl -w net.core.rmem_max=24912805
sudo sysctl -w net.core.wmem_max=24912805
```

If the NIC path supports it, also consider jumbo MTU on the X310 interface.

## Start RuView In RF-Direct Mode

On `isac`:

```bash
cd /path/to/RuView/v2
cargo run -p wifi-densepose-sensing-server -- \
  --source rf-direct \
  --rf-bind-addr 127.0.0.1 \
  --rf-udp-port 5020 \
  --http-port 3000 \
  --ws-port 3001 \
  --ui-path ../ui \
  --bind-addr 0.0.0.0
```

From the laptop, open:

```text
http://isac:3000/ui/index.html
```

If you bind to `0.0.0.0`, set `RUVIEW_API_TOKEN` or keep the host on a trusted
lab network.

## Verify Without Hardware

On any host:

```bash
python scripts/openisac_to_ruview_bridge.py \
  --demo \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --verbose
```

This sends synthetic, explicitly labelled `openisac-rd-demo` observations. It
tests the contract and transport only; it is not sensing evidence.

## First Real OpenISAC/X310 Experiment

Use a shielded setup, cables/attenuators, or an allowed lab frequency plan.
Transmitting RF may be regulated in your location.

Run OpenISAC and the bridge on the same host. The loopback restriction is
intentional; unauthenticated remote RF UDP is not supported.

```bash
python scripts/openisac_to_ruview_bridge.py \
  --openisac-host 127.0.0.1 \
  --openisac-port 8888 \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --record-jsonl data/openisac-observations.jsonl \
  --record-raw-dir data/openisac-raw \
  --verbose
```

The bridge sends paired observations like:

```json
{
  "schema": "ruview.rf_observation",
  "protocol_version": 2,
  "source": "openisac-rd",
  "source_instance_id": "0123456789abcdef0123456789abcdef",
  "config_epoch": 0,
  "frame_id": 1,
  "sequence": 1,
  "source_timestamp_ns": null,
  "received_at_ns": 1750000000000000000,
  "config_hash": "sha256:<64 hex characters>",
  "freshness": "fresh",
  "observation": {
    "range_doppler": {
      "peaks": [{"kind": "unclassified_peak", "range_bin": 8, "doppler_bin": 50}]
    },
    "cfar": {"candidate_clusters": []},
    "micro_doppler": null
  }
}
```

RuView exposes this under `rf_observation`. A peak is not called a target; only
metadata-derived CFAR clusters appear as `candidate_clusters`. If raw or
metadata is missing, the bridge forwards nothing. After five seconds without a
valid frame, the source becomes `rf-direct:offline`, freshness becomes `stale`,
and candidate clusters are cleared.

Each bridge process generates a random `source_instance_id`. A new instance can
restart its sequence at zero; RuView retires the previous instance and rejects
later replay from it. RuView retains up to 32 retired instance IDs and fails
closed on an unknown 34th instance rather than forgetting replay history; restart
the RuView service under operator control to open a fresh trust window. Runtime
parameter changes increment `config_epoch` and
atomically discard incomplete chunks and unmatched frame halves. Raw recordings
are stored under run/epoch/sender directories with a SHA-256 manifest and
collision suffixes, so repeated frame IDs never silently overwrite evidence.

## Optimization Roadmap

1. Prove transport integrity first: paired frame IDs, bounded loss/reordering,
   stable configuration hashes, and correct stale/offline transitions.
2. Establish static-reflector repeatability for range-Doppler diagnostics and
   CFAR candidate clusters. Treat these as observations, not people or motion.
3. Pre-register and evaluate any proposed motion, occupancy, pose, or vital-sign
   estimator offline against labelled controls, including empty-room negatives.
4. Add a capability only after its evidence, calibration limits, and failure
   behavior are documented and regression-tested. Until then it remains false
   in the capability manifest.

Keep OpenISAC and the bridge close to UHD. Send bounded diagnostic observations
to RuView, not raw IQ or inference-shaped feature frames, unless a later reviewed
experiment explicitly requires a different contract.
