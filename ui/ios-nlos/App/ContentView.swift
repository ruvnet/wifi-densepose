import Foundation
import RuViewNLOSApple
import RuViewNLOSCore
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel
    @Environment(\.scenePhase) private var scenePhase
    @State private var exportConsent = false
    @State private var diagnosticURL: URL?
    @State private var exportError: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    onboardingCard
                    visibleDepthCard
                    Text("NLOS MONITOR")
                        .font(.caption.bold().monospaced())
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
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

    private var onboardingCard: some View {
        card(title: "Beta tester setup") {
            Label("Validate your iPhone's public ARKit LiDAR capabilities before opening the NLOS monitor.", systemImage: "iphone.gen3.radiowaves.left.and.right")
                .font(.subheadline)
            Text("This test measures only visible surfaces. Every result is labeled direct_depth and is never presented as around the corner evidence.")
                .font(.caption.bold())
                .foregroundStyle(.orange)
            Link(destination: URL(string: "https://ruview-nlos.ruv.chatgpt.site")!) {
                Label("Open the visual explainer", systemImage: "safari")
            }
            Link(destination: URL(string: "https://github.com/ruvnet/RuView/issues/1690")!) {
                Label("Read the test guide and provide feedback", systemImage: "bubble.left.and.exclamationmark.bubble.right")
            }
        }
    }

    private var visibleDepthCard: some View {
        card(title: "Visible depth validation") {
            VisibleDepthValidationView(
                session: model.visibleDepthSession,
                exportConsent: $exportConsent,
                diagnosticURL: $diagnosticURL,
                exportError: $exportError,
                start: model.startVisibleDepthValidation,
                cancel: model.cancelVisibleDepthValidation,
                prepareExport: model.prepareDiagnosticExport
            )
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

private struct VisibleDepthValidationView: View {
    @ObservedObject var session: VisibleDepthValidationSession
    @Binding var exportConsent: Bool
    @Binding var diagnosticURL: URL?
    @Binding var exportError: String?
    let start: () -> Void
    let cancel: () -> Void
    let prepareExport: (Bool) throws -> URL

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("DIRECT_DEPTH")
                    .font(.caption.bold().monospaced())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.15), in: Capsule())
                    .foregroundStyle(.orange)
                Spacer()
                Text(phaseLabel)
                    .font(.caption.bold().monospacedDigit())
            }

            Text(session.statusMessage)
                .font(.subheadline)

            if isRunning {
                ProgressView(value: phaseProgress)
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                    metric("FPS", String(format: "%.1f", session.metrics.fps))
                    metric("Depth coverage", "\(Int(session.metrics.depthCoverage * 100))%")
                    metric("Tracking", session.metrics.trackingState)
                    metric("Movement", String(format: "%.3f m/s", session.metrics.movementMetersPerSecond))
                    metric("Thermal", session.metrics.thermalState)
                    metric("Remaining", "\(session.metrics.phaseSecondsRemaining)s")
                }
                Button("Cancel validation", role: .cancel, action: cancel)
                    .buttonStyle(.bordered)
            } else {
                Button {
                    exportConsent = false
                    diagnosticURL = nil
                    exportError = nil
                    start()
                } label: {
                    Label(session.diagnostic == nil ? "Start 45 second validation" : "Run again", systemImage: "sensor.tag.radiowaves.forward")
                }
                .buttonStyle(.borderedProminent)
            }

            if session.diagnostic != nil {
                Divider()
                diagnosticPreview
                Toggle(isOn: $exportConsent) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("I choose to export aggregate diagnostics")
                        Text("The JSON contains no images, raw depth, endpoint, token, or raw samples.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .onChange(of: exportConsent) { consent in
                    if !consent { diagnosticURL = nil }
                }

                if let diagnosticURL {
                    ShareLink(item: diagnosticURL) {
                        Label("Share diagnostic JSON", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button("Prepare local JSON") {
                        do {
                            diagnosticURL = try prepareExport(exportConsent)
                            exportError = nil
                        } catch {
                            exportError = error.localizedDescription
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(!exportConsent)
                }
                if let exportError {
                    Text(exportError).font(.caption).foregroundStyle(.red)
                }
                Link("Open issue 1690 to submit feedback", destination: URL(string: "https://github.com/ruvnet/RuView/issues/1690")!)
                    .font(.caption)
            }
        }
    }

    private var isRunning: Bool {
        session.state == .requestingPermission || session.state == .calibration || session.state == .wallScan
    }

    private var phaseLabel: String {
        switch session.state {
        case .idle: return "READY"
        case .requestingPermission: return "PERMISSION"
        case .calibration: return "CALIBRATION 15S"
        case .wallScan: return "WALL SCAN 30S"
        case .completed: return "COMPLETED"
        case .cancelled: return "CANCELLED"
        case .failed: return "FAILED"
        }
    }

    private var phaseProgress: Double {
        let total = session.state == .calibration ? 15.0 : 30.0
        return max(0, min(1, (total - Double(session.metrics.phaseSecondsRemaining)) / total))
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.subheadline.bold().monospacedDigit()).lineLimit(1).minimumScaleFactor(0.7)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private var diagnosticPreview: some View {
        if let diagnostic = session.diagnostic {
            VStack(alignment: .leading, spacing: 4) {
                Text("Diagnostic preview")
                    .font(.subheadline.bold())
                Text("Evidence: \(diagnostic.evidenceLabel.rawValue)")
                Text("Physical NLOS: \(diagnostic.physicalNLOSStatus)")
                Text("Permission: \(diagnostic.cameraPermission)")
                Text("Phases: \(diagnostic.phases.count) aggregate summaries")
                if let byteCount = try? diagnostic.encodedJSON().count {
                    Text("Encoded size: \(byteCount) bytes of 65,536 maximum")
                }
                Text("Raw sensor export: false")
            }
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(uiColor: .tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}
