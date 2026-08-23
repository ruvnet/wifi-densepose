import Foundation
import RuViewNLOSApple
import RuViewNLOSCore
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    maturityCard
                    connectionCard
                    statusCard
                    visualizationCard
                    capabilityCard
                    boundaryCard
                }
                .padding()
            }
            .navigationTitle("RuView NLOS")
            .background(Color(uiColor: .systemGroupedBackground))
            .onChange(of: scenePhase) { phase in
                if phase != .active {
                    model.suspendForPrivacy()
                }
            }
        }
    }

    private var maturityCard: some View {
        card(title: "Software preview") {
            Text("The software path is implemented and locally validated. Research readiness remains blocked on these evidence gates:")
                .font(.subheadline)
            maturityGate("macOS native compilation")
            maturityGate("Physical VL53L8CH reproduction at 27 fps or better")
            maturityGate("Measured CSI fusion improvement of at least 25 percent")
            Text("Builds, simulators, and synthetic replay do not close hardware or measured-fusion evidence gates.")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
        }
    }

    private var connectionCard: some View {
        card(title: "Authenticated stream") {
            TextField("wss://host.example/api/v1/nlos/ws", text: $model.endpointText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .textContentType(.URL)
                .padding(12)
                .background(Color(uiColor: .tertiarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 10))

            SecureField(
                model.storedTokenAvailable ? "Pairing token stored in Keychain" : "Pairing token",
                text: $model.pairingToken
            )
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .textContentType(.password)
            .padding(12)
            .background(Color(uiColor: .tertiarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))

            HStack {
                Button(model.isConnected ? "Reconnect" : "Connect") {
                    model.connect()
                }
                .buttonStyle(.borderedProminent)

                if model.isConnected {
                    Button("Disconnect", role: .cancel) {
                        model.disconnect()
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()

                if model.storedTokenAvailable {
                    Button("Forget token", role: .destructive) {
                        model.forgetPairingToken()
                    }
                    .font(.caption)
                }
            }

            Label(
                "Bearer token stays in this device's Keychain and is sent only over wss.",
                systemImage: "lock.shield"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private var statusCard: some View {
        card(title: "Evidence status") {
            HStack(alignment: .top) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                    .padding(.top, 4)
                Text(model.statusMessage)
                    .font(.subheadline)
                Spacer()
            }

            if let frame = model.frame {
                HStack(spacing: 8) {
                    badge(frame.source.rawValue.uppercased(), color: sourceColor(frame.source))
                    badge(frame.evidenceLevel.rawValue.uppercased(), color: .indigo)
                    badge("SEQ \(frame.sequence)", color: .gray)
                }
                .accessibilityElement(children: .combine)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Sensor: \(frame.provenance.sensorModel)")
                    Text("Transient: \(frame.provenance.transientKind.rawValue)")
                    Text("Histogram preserved: \(frame.provenance.histogramPreserved ? "yes" : "no")")
                    Text("Algorithm: \(frame.algorithmVersion)")
                }
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            }
        }
    }

    private var visualizationCard: some View {
        card(title: "Validated hidden target hypotheses") {
            ZStack {
                TrackCanvas(tracks: model.tracks)
                    .frame(height: 300)
                    .privacySensitive()

                if let watermark = model.frame?.watermark {
                    Text(watermark)
                        .font(.system(size: 38, weight: .black, design: .rounded))
                        .foregroundStyle(.orange.opacity(0.42))
                        .rotationEffect(.degrees(-18))
                        .accessibilityLabel("Synthetic evidence watermark")
                }

                if model.tracks.isEmpty {
                    Text("NO DISPLAYABLE TRACKS")
                        .font(.caption.bold().monospaced())
                        .foregroundStyle(.secondary)
                        .padding(10)
                        .background(.ultraThinMaterial, in: Capsule())
                }
            }

            ForEach(model.tracks) { track in
                HStack {
                    VStack(alignment: .leading) {
                        Text(track.trackId)
                            .font(.subheadline.monospaced())
                            .lineLimit(1)
                        Text(String(
                            format: "x %.2f  y %.2f  z %.2f m",
                            track.positionM.x,
                            track.positionM.y,
                            track.positionM.z
                        ))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("\(Int(track.confidence * 100))%")
                        .font(.headline.monospacedDigit())
                    badge(track.state.rawValue.uppercased(), color: track.state == .degraded ? .orange : .cyan)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .privacySensitive()
    }

    private var capabilityCard: some View {
        card(title: "Apple capability probe") {
            capabilityRow("ARKit scene depth", model.capabilities.sceneDepth)
            capabilityRow("ARKit smoothed depth", model.capabilities.smoothedSceneDepth)
            capabilityRow("ARKit scene mesh", model.capabilities.sceneMesh)
            capabilityRow("ARKit world pose", model.capabilities.worldPose)
            capabilityRow("Raw photon histograms", model.capabilities.rawPhotonHistograms)

            Text(model.capabilities.rawPhotonHistogramReason)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var boundaryCard: some View {
        card(title: "Interpretation boundary") {
            Label(
                "This client visualizes validated NLOS output produced by an external transient histogram pipeline.",
                systemImage: "waveform.path.ecg.rectangle"
            )
            Label(
                "ARKit depth, mesh, and pose are useful context, but this app never labels them as optical NLOS evidence.",
                systemImage: "exclamationmark.shield"
            )
            Text("Unknown, stale, malformed, replayed, oversized, or unauthenticated input is hidden by default.")
                .font(.caption.bold())
        }
        .font(.subheadline)
    }

    private var statusColor: Color {
        switch model.connectionState {
        case .connected: return .green
        case .connecting: return .yellow
        case .blocked: return .red
        case .disconnected: return .secondary
        }
    }

    private func sourceColor(_ source: NLOSSource) -> Color {
        switch source {
        case .live: return .green
        case .replay: return .blue
        case .synthetic: return .orange
        }
    }

    private func capabilityRow(
        _ title: String,
        _ availability: AppleCapabilityAvailability
    ) -> some View {
        HStack {
            Text(title)
            Spacer()
            Label(
                availability.rawValue.capitalized,
                systemImage: availability == .available ? "checkmark.circle.fill" : "xmark.circle.fill"
            )
            .foregroundStyle(availability == .available ? .green : .secondary)
        }
        .font(.subheadline)
    }

    private func maturityGate(_ title: String) -> some View {
        Label("OPEN · \(title)", systemImage: "circle.dashed")
            .font(.subheadline)
            .foregroundStyle(.orange)
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.bold().monospaced())
            .lineLimit(1)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.14), in: Capsule())
            .foregroundStyle(color)
    }

    private func card<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            content()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 18))
    }
}
