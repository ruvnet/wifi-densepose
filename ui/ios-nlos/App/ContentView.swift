import Foundation
import RuViewNLOSApple
import RuViewNLOSCore
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel
    @Environment(\.scenePhase) private var scenePhase
    @ScaledMetric(relativeTo: .largeTitle) private var heroFontSize: CGFloat = 42
    @State private var exportConsent = false
    @State private var diagnosticURL: URL?
    @State private var exportError: String?
    @State private var spatialMode: TrackCanvasMode = .targets

    var body: some View {
        NavigationStack {
            ZStack(alignment: .top) {
                RuViewTheme.background.ignoresSafeArea()
                InstrumentGrid()

                ScrollView {
                    LazyVStack(spacing: 16) {
                        instrumentHeader
                        statusCard
                        visualizationCard
                        onboardingCard
                        visibleDepthCard
                        connectionCard
                        capabilityCard
                        boundaryCard
                        evidenceLegend
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .padding(.bottom, 36)
                }
                .scrollContentBackground(.hidden)
            }
            .toolbar(.hidden, for: .navigationBar)
            .tint(RuViewTheme.cyan)
            .preferredColorScheme(.dark)
            .onChange(of: scenePhase) { phase in
                if phase != .active {
                    model.suspendForPrivacy()
                }
            }
        }
    }

    private var instrumentHeader: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 7) {
                    Text("RUVIEW / MOBILE FIELD LAB")
                        .font(RuViewTypography.mono(11))
                        .tracking(1.6)
                        .foregroundStyle(RuViewTheme.muted)
                    HStack(spacing: 7) {
                        Circle()
                            .fill(evidenceState.color)
                            .frame(width: 8, height: 8)
                            .shadow(color: evidenceState.color.opacity(0.8), radius: 5)
                        Text(evidenceState.label)
                            .font(RuViewTypography.mono(11))
                            .foregroundStyle(evidenceState.color)
                            .lineLimit(2)
                    }
                }

                Spacer(minLength: 4)
                OrbitalSignatureView(accent: evidenceState.color)
                    .frame(width: 78, height: 78)
                    .accessibilityHidden(true)
            }

            VStack(alignment: .leading, spacing: 7) {
                Text("NLOS FIELD")
                    .foregroundStyle(.white)
                Text("MONITOR")
                    .foregroundStyle(
                        LinearGradient(
                            colors: [RuViewTheme.cyan, RuViewTheme.green],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
            }
            .font(RuViewTypography.outfit(heroFontSize, weight: .bold))
            .tracking(-1.4)
            .minimumScaleFactor(0.72)
            .accessibilityElement(children: .combine)

            Text("Review authenticated evidence, sensor provenance, and privacy controls in one fail closed instrument.")
                .font(RuViewTypography.outfit(16))
                .foregroundStyle(RuViewTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 10) {
                headerMetric(
                    eyebrow: "STREAM",
                    value: connectionLabel,
                    color: connectionColor
                )
                headerMetric(
                    eyebrow: "EVIDENCE",
                    value: evidenceLabel,
                    color: evidenceState.color
                )
            }

            HStack(spacing: 8) {
                Image(systemName: "lock.shield.fill")
                    .foregroundStyle(RuViewTheme.green)
                Text("PRIVACY GUARD ACTIVE")
                    .font(RuViewTypography.mono(11))
                    .tracking(0.7)
                Spacer()
                Text("FAIL CLOSED")
                    .font(RuViewTypography.mono(10))
                    .foregroundStyle(RuViewTheme.muted)
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 44)
            .background(RuViewTheme.panelStrong, in: Capsule())
            .overlay {
                Capsule().stroke(RuViewTheme.green.opacity(0.3), lineWidth: 1)
            }
            .accessibilityElement(children: .combine)
        }
        .padding(.top, 4)
    }

    private var statusCard: some View {
        instrumentCard(eyebrow: "01 / EVIDENCE", title: "Current signal state", accent: evidenceState.color) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    Circle()
                        .fill(evidenceState.color.opacity(0.14))
                        .frame(width: 48, height: 48)
                    Image(systemName: evidenceState.icon)
                        .font(.system(size: 19, weight: .semibold))
                        .foregroundStyle(evidenceState.color)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(evidenceState.label)
                        .font(RuViewTypography.mono(16))
                        .foregroundStyle(evidenceState.color)
                    Text(evidenceState.detail)
                        .font(RuViewTypography.outfit(12))
                        .foregroundStyle(RuViewTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .accessibilityElement(children: .combine)

            Text(model.statusMessage)
                .font(RuViewTypography.outfit(14))
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 12))

            if let frame = model.frame {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        frameBadges(frame)
                    }
                    VStack(alignment: .leading, spacing: 8) {
                        frameBadges(frame)
                    }
                }
                .accessibilityElement(children: .combine)

                VStack(spacing: 0) {
                    provenanceRow("SENSOR", frame.provenance.sensorModel)
                    provenanceRow("TRANSIENT", frame.provenance.transientKind.rawValue)
                    provenanceRow("HISTOGRAM", frame.provenance.histogramPreserved ? "preserved" : "not preserved")
                    provenanceRow("ALGORITHM", frame.algorithmVersion, isLast: true)
                }
                .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private var visualizationCard: some View {
        instrumentCard(eyebrow: "02 / SPATIAL", title: "Hidden target hypotheses", accent: RuViewTheme.cyan) {
            Picker("Spatial visualization", selection: $spatialMode) {
                Text("TRACKS").tag(TrackCanvasMode.targets)
                Text("POINT CLOUD").tag(TrackCanvasMode.pointCloud)
            }
            .pickerStyle(.segmented)
            .accessibilityIdentifier("nlos-spatial-mode")

            Text(
                spatialMode == .pointCloud
                    ? "The point cloud is a deterministic rendering of gated reconstruction tracks and relay geometry. It is not raw iPhone LiDAR output."
                    : "Only validated, fresh tracks are shown. Uncertainty rings represent the reported position covariance."
            )
                .font(RuViewTypography.outfit(12))
                .foregroundStyle(RuViewTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            ZStack {
                TrackCanvas(tracks: displayableTracks, mode: spatialMode)
                    .frame(height: 310)
                    .privacySensitive()

                if let watermark = model.frame?.watermark {
                    Text(watermark)
                        .font(RuViewTypography.outfit(35, weight: .bold))
                        .tracking(2)
                        .foregroundStyle(RuViewTheme.orange.opacity(0.48))
                        .rotationEffect(.degrees(-18))
                        .accessibilityLabel("\(watermark.capitalized) evidence watermark")
                }

                if displayableTracks.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "scope")
                            .font(.title2)
                        Text("NO DISPLAYABLE TRACKS")
                            .font(RuViewTypography.mono(11))
                    }
                    .foregroundStyle(RuViewTheme.muted)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(RuViewTheme.background.opacity(0.88), in: Capsule())
                    .overlay {
                        Capsule().stroke(RuViewTheme.border, lineWidth: 1)
                    }
                }
            }

            ForEach(displayableTracks) { track in
                trackRow(track)
            }
        }
        .privacySensitive()
    }

    private var onboardingCard: some View {
        instrumentCard(eyebrow: "03 / BETA", title: "Tester flight plan", accent: RuViewTheme.green) {
            Label {
                Text("Validate your iPhone's public ARKit LiDAR capabilities before opening the NLOS monitor.")
            } icon: {
                Image(systemName: "iphone.gen3.radiowaves.left.and.right")
                    .foregroundStyle(RuViewTheme.green)
            }
            .font(.subheadline)
            .fixedSize(horizontal: false, vertical: true)

            boundaryNotice(
                "This test measures visible surfaces only. Every result is labeled DIRECT_DEPTH and is never presented as around the corner evidence.",
                color: RuViewTheme.orange
            )

            Link(destination: URL(string: "https://ruview-nlos.ruv.chatgpt.site")!) {
                Label("Open the visual explainer", systemImage: "safari")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(InstrumentPrimaryButtonStyle(accent: RuViewTheme.green))
            .accessibilityHint("Opens the RuView NLOS explainer in your browser")

            Link(destination: URL(string: "https://github.com/ruvnet/RuView/issues/1690")!) {
                Label("Test guide and feedback", systemImage: "bubble.left.and.exclamationmark.bubble.right")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(InstrumentSecondaryButtonStyle(accent: RuViewTheme.green))
            .accessibilityHint("Opens issue 1690 on GitHub")
        }
    }

    private var visibleDepthCard: some View {
        instrumentCard(eyebrow: "04 / DEVICE", title: "Visible depth validation", accent: RuViewTheme.orange) {
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
        instrumentCard(eyebrow: "05 / TRANSPORT", title: "Authenticated stream", accent: RuViewTheme.cyan) {
            VStack(alignment: .leading, spacing: 6) {
                Text("SECURE ENDPOINT")
                    .font(.caption2.bold().monospaced())
                    .tracking(0.8)
                    .foregroundStyle(RuViewTheme.muted)
                TextField("wss://host.example/api/v1/nlos/ws", text: $model.endpointText)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .textContentType(.URL)
                    .instrumentInput()
                    .accessibilityLabel("Secure WebSocket endpoint")
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("PAIRING TOKEN")
                    .font(.caption2.bold().monospaced())
                    .tracking(0.8)
                    .foregroundStyle(RuViewTheme.muted)
                SecureField(
                    model.storedTokenAvailable ? "Token stored in Keychain" : "Pairing token",
                    text: $model.pairingToken
                )
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .textContentType(.password)
                .instrumentInput()
                .accessibilityLabel("Pairing token")
            }

            VStack(spacing: 10) {
                Button(model.isConnected ? "Reconnect secure stream" : "Connect secure stream") {
                    model.connect()
                }
                .buttonStyle(InstrumentPrimaryButtonStyle(accent: RuViewTheme.cyan))

                if model.isConnected {
                    Button("Disconnect", role: .cancel) {
                        model.disconnect()
                    }
                    .buttonStyle(InstrumentSecondaryButtonStyle(accent: RuViewTheme.cyan))
                }

                if model.storedTokenAvailable {
                    Button("Forget stored token", role: .destructive) {
                        model.forgetPairingToken()
                    }
                    .frame(minHeight: 44)
                    .font(.caption.bold().monospaced())
                    .foregroundStyle(RuViewTheme.red)
                }
            }

            Label(
                "The bearer token stays in this device's Keychain and is sent only over wss.",
                systemImage: "lock.shield"
            )
            .font(.caption)
            .foregroundStyle(RuViewTheme.textSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var capabilityCard: some View {
        instrumentCard(eyebrow: "06 / CAPABILITY", title: "Apple capability probe", accent: RuViewTheme.indigo) {
            VStack(spacing: 0) {
                capabilityRow("ARKit scene depth", model.capabilities.sceneDepth)
                capabilityRow("ARKit smoothed depth", model.capabilities.smoothedSceneDepth)
                capabilityRow("ARKit scene mesh", model.capabilities.sceneMesh)
                capabilityRow("ARKit world pose", model.capabilities.worldPose)
                capabilityRow("Raw photon histograms", model.capabilities.rawPhotonHistograms, isLast: true)
            }
            .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 12))

            Text(model.capabilities.rawPhotonHistogramReason)
                .font(.caption)
                .foregroundStyle(RuViewTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var boundaryCard: some View {
        instrumentCard(eyebrow: "07 / BOUNDARY", title: "Interpretation boundary", accent: RuViewTheme.orange) {
            boundaryNotice(
                "This client visualizes validated NLOS output produced by an external transient histogram pipeline.",
                color: RuViewTheme.cyan,
                icon: "waveform.path.ecg.rectangle"
            )
            boundaryNotice(
                "ARKit depth, mesh, and pose are context only. The app never labels them as optical NLOS evidence.",
                color: RuViewTheme.orange,
                icon: "exclamationmark.shield"
            )
            Text("Unknown, stale, malformed, replayed, oversized, or unauthenticated input is hidden by default.")
                .font(.caption.bold())
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var evidenceLegend: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("EVIDENCE LEGEND")
                .font(.caption2.bold().monospaced())
                .tracking(1.2)
                .foregroundStyle(RuViewTheme.muted)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    legendItem("VERIFIED", RuViewTheme.green)
                    legendItem("UNVERIFIED", RuViewTheme.yellow)
                    legendItem("SYNTHETIC", RuViewTheme.orange)
                }
                VStack(alignment: .leading, spacing: 8) {
                    legendItem("VERIFIED", RuViewTheme.green)
                    legendItem("UNVERIFIED", RuViewTheme.yellow)
                    legendItem("SYNTHETIC", RuViewTheme.orange)
                }
            }
            Text("A label describes evidence provenance, not certainty about the physical world.")
                .font(.caption2)
                .foregroundStyle(RuViewTheme.muted)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 4)
        .accessibilityElement(children: .combine)
    }

    private var evidenceState: EvidencePresentation {
        switch model.connectionState {
        case .disconnected:
            return EvidencePresentation(
                label: "DISCONNECTED",
                detail: "No authenticated transport is active and no track evidence is displayed.",
                color: RuViewTheme.muted,
                icon: "antenna.radiowaves.left.and.right.slash"
            )
        case .connecting:
            return EvidencePresentation(
                label: "LIVE UNVERIFIED",
                detail: "The secure transport is opening. No evidence is accepted yet.",
                color: RuViewTheme.yellow,
                icon: "ellipsis"
            )
        case .blocked:
            let isStale = model.statusMessage.localizedCaseInsensitiveContains("stale")
                || model.statusMessage.localizedCaseInsensitiveContains("expired")
            return EvidencePresentation(
                label: isStale ? "STALE" : (model.isConnected ? "LIVE UNVERIFIED" : "DISCONNECTED"),
                detail: "Input failed validation and has been hidden. Reconnect only after resolving the reported cause.",
                color: RuViewTheme.red,
                icon: "exclamationmark.octagon.fill"
            )
        case .connected:
            guard let frame = model.frame else {
                return EvidencePresentation(
                    label: "LIVE UNVERIFIED",
                    detail: "Transport is authenticated, but no validated evidence frame is available.",
                    color: RuViewTheme.yellow,
                    icon: "shield.lefthalf.filled"
                )
            }
            switch frame.source {
            case .synthetic:
                return EvidencePresentation(
                    label: "SYNTHETIC",
                    detail: "Generated evidence is accepted only at L0 and remains visibly watermarked.",
                    color: RuViewTheme.orange,
                    icon: "testtube.2"
                )
            case .replay:
                return EvidencePresentation(
                    label: "DISCONNECTED",
                    detail: "Replay provenance is retained below, but recorded tracks are withheld from the live evidence surface.",
                    color: RuViewTheme.muted,
                    icon: "arrow.counterclockwise.circle.fill"
                )
            case .live:
                if frame.evidenceLevel >= .l2Calibrated {
                    return EvidencePresentation(
                        label: "LIVE VERIFIED",
                        detail: "A fresh calibrated live frame passed the fail closed validator.",
                        color: RuViewTheme.green,
                        icon: "checkmark.shield.fill"
                    )
                }
                return EvidencePresentation(
                    label: "LIVE UNVERIFIED",
                    detail: "A fresh measured live frame passed validation but is not calibrated.",
                    color: RuViewTheme.yellow,
                    icon: "shield.lefthalf.filled"
                )
            }
        }
    }

    private var displayableTracks: [NLOSTrack] {
        guard model.connectionState == .connected, let frame = model.frame else {
            return []
        }
        switch frame.source {
        case .synthetic:
            return model.tracks
        case .live where frame.evidenceLevel >= .l2Calibrated:
            return model.tracks
        case .live, .replay:
            return []
        }
    }

    private var connectionLabel: String {
        switch model.connectionState {
        case .disconnected: return "OFFLINE"
        case .connecting: return "OPENING"
        case .connected: return "SECURE"
        case .blocked: return "BLOCKED"
        }
    }

    private var connectionColor: Color {
        switch model.connectionState {
        case .disconnected: return RuViewTheme.muted
        case .connecting: return RuViewTheme.yellow
        case .connected: return RuViewTheme.green
        case .blocked: return RuViewTheme.red
        }
    }

    private var evidenceLabel: String {
        guard let frame = model.frame else { return "NONE" }
        return frame.evidenceLevel.rawValue.uppercased()
    }

    @ViewBuilder
    private func frameBadges(_ frame: TrackDisplayFrame) -> some View {
        badge(frame.source.rawValue.uppercased(), color: sourceColor(frame.source))
        badge(frame.evidenceLevel.rawValue.uppercased(), color: evidenceState.color)
        badge("SEQ \(frame.sequence)", color: RuViewTheme.muted)
    }

    private func sourceColor(_ source: NLOSSource) -> Color {
        switch source {
        case .live: return RuViewTheme.cyan
        case .replay: return RuViewTheme.indigo
        case .synthetic: return RuViewTheme.orange
        }
    }

    private func headerMetric(eyebrow: String, value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(eyebrow)
                .font(RuViewTypography.mono(10))
                .tracking(0.8)
                .foregroundStyle(RuViewTheme.muted)
            Text(value)
                .font(RuViewTypography.mono(11))
                .foregroundStyle(color)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .padding(12)
        .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
        .background(RuViewTheme.panel, in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(color.opacity(0.3), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }

    private func trackRow(_ track: NLOSTrack) -> some View {
        HStack(spacing: 12) {
            Circle()
                .fill(track.state == .degraded ? RuViewTheme.orange : RuViewTheme.cyan)
                .frame(width: 8, height: 8)
                .shadow(
                    color: (track.state == .degraded ? RuViewTheme.orange : RuViewTheme.cyan).opacity(0.8),
                    radius: 4
                )
            VStack(alignment: .leading, spacing: 4) {
                Text(track.trackId)
                    .font(RuViewTypography.mono(14))
                    .lineLimit(1)
                Text(String(
                    format: "x %.2f  y %.2f  z %.2f m",
                    track.positionM.x,
                    track.positionM.y,
                    track.positionM.z
                ))
                .font(RuViewTypography.mono(11, medium: false))
                .foregroundStyle(RuViewTheme.textSecondary)
            }
            Spacer(minLength: 4)
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(Int(track.confidence * 100))%")
                    .font(RuViewTypography.mono(17))
                badge(
                    track.state.rawValue.uppercased(),
                    color: track.state == .degraded ? RuViewTheme.orange : RuViewTheme.cyan
                )
            }
        }
        .padding(12)
        .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "Track \(track.trackId), \(track.state.rawValue), confidence \(Int(track.confidence * 100)) percent, position x \(track.positionM.x) y \(track.positionM.y) z \(track.positionM.z) meters"
        )
    }

    private func capabilityRow(
        _ title: String,
        _ availability: AppleCapabilityAvailability,
        isLast: Bool = false
    ) -> some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Text(title)
                    .font(RuViewTypography.outfit(14))
                Spacer()
                Label(
                    availability.rawValue.capitalized,
                    systemImage: availability == .available ? "checkmark.circle.fill" : "xmark.circle.fill"
                )
                .font(RuViewTypography.mono(11))
                .foregroundStyle(availability == .available ? RuViewTheme.green : RuViewTheme.muted)
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 48)
            if !isLast {
                Rectangle()
                    .fill(RuViewTheme.border)
                    .frame(height: 1)
                    .padding(.leading, 12)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func provenanceRow(_ label: String, _ value: String, isLast: Bool = false) -> some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text(label)
                    .font(RuViewTypography.mono(10))
                    .tracking(0.6)
                    .foregroundStyle(RuViewTheme.muted)
                Spacer()
                Text(value)
                    .font(RuViewTypography.mono(11, medium: false))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.trailing)
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 44)
            if !isLast {
                Rectangle()
                    .fill(RuViewTheme.border)
                    .frame(height: 1)
                    .padding(.leading, 12)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func boundaryNotice(_ text: String, color: Color, icon: String = "exclamationmark.shield") -> some View {
        Label {
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: icon)
                .foregroundStyle(color)
        }
        .font(RuViewTypography.outfit(12))
        .foregroundStyle(RuViewTheme.textSecondary)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(color.opacity(0.24), lineWidth: 1)
        }
    }

    private func legendItem(_ label: String, _ color: Color) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
                .font(RuViewTypography.mono(10))
                .foregroundStyle(RuViewTheme.textSecondary)
        }
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(RuViewTypography.mono(10))
            .lineLimit(1)
            .minimumScaleFactor(0.72)
            .padding(.horizontal, 8)
            .frame(minHeight: 28)
            .background(color.opacity(0.12), in: Capsule())
            .overlay {
                Capsule().stroke(color.opacity(0.25), lineWidth: 1)
            }
            .foregroundStyle(color)
    }

    private func instrumentCard<Content: View>(
        eyebrow: String,
        title: String,
        accent: Color,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Rectangle()
                    .fill(accent)
                    .frame(width: 22, height: 2)
                    .shadow(color: accent.opacity(0.8), radius: 4)
                Text(eyebrow)
                    .font(RuViewTypography.mono(10))
                    .tracking(1.1)
                    .foregroundStyle(accent)
            }
            Text(title)
                .font(RuViewTypography.outfit(20, weight: .medium))
                .foregroundStyle(.white)
            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RuViewTheme.panel, in: RoundedRectangle(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .stroke(RuViewTheme.border, lineWidth: 1)
        }
        .shadow(color: Color.black.opacity(0.5), radius: 18, y: 8)
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
            HStack(spacing: 8) {
                statusBadge("DIRECT_DEPTH", color: RuViewTheme.orange)
                Spacer()
                Text(phaseLabel)
                    .font(.caption.bold().monospacedDigit())
                    .foregroundStyle(phaseColor)
            }

            Text(session.statusMessage)
                .font(.subheadline)
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)

            if isRunning {
                ProgressView(value: phaseProgress)
                    .tint(RuViewTheme.orange)
                    .accessibilityLabel("Validation phase progress")
                    .accessibilityValue("\(Int(phaseProgress * 100)) percent")
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                    metric("FPS", String(format: "%.1f", session.metrics.fps))
                    metric("DEPTH COVERAGE", "\(Int(session.metrics.depthCoverage * 100))%")
                    metric("TRACKING", session.metrics.trackingState)
                    metric("MOVEMENT", String(format: "%.3f m/s", session.metrics.movementMetersPerSecond))
                    metric("THERMAL", session.metrics.thermalState)
                    metric("REMAINING", "\(session.metrics.phaseSecondsRemaining)s")
                }
                Button("Cancel validation", role: .cancel, action: cancel)
                    .buttonStyle(InstrumentSecondaryButtonStyle(accent: RuViewTheme.orange))
            } else {
                Button {
                    exportConsent = false
                    diagnosticURL = nil
                    exportError = nil
                    start()
                } label: {
                    Label(
                        session.diagnostic == nil ? "Start 45 second validation" : "Run validation again",
                        systemImage: "sensor.tag.radiowaves.forward"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(InstrumentPrimaryButtonStyle(accent: RuViewTheme.orange))
            }

            if session.diagnostic != nil {
                Rectangle()
                    .fill(RuViewTheme.border)
                    .frame(height: 1)
                diagnosticPreview
                Toggle(isOn: $exportConsent) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("I choose to export aggregate diagnostics")
                            .font(.subheadline)
                        Text("The JSON contains no images, raw depth, endpoint, token, or raw samples.")
                            .font(.caption)
                            .foregroundStyle(RuViewTheme.textSecondary)
                    }
                }
                .tint(RuViewTheme.green)
                .onChange(of: exportConsent) { consent in
                    if !consent { diagnosticURL = nil }
                }

                if let diagnosticURL {
                    ShareLink(item: diagnosticURL) {
                        Label("Share diagnostic JSON", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(InstrumentPrimaryButtonStyle(accent: RuViewTheme.green))
                } else {
                    Button("Prepare local JSON") {
                        do {
                            diagnosticURL = try prepareExport(exportConsent)
                            exportError = nil
                        } catch {
                            exportError = error.localizedDescription
                        }
                    }
                    .buttonStyle(InstrumentSecondaryButtonStyle(accent: RuViewTheme.green))
                    .disabled(!exportConsent)
                    .opacity(exportConsent ? 1 : 0.45)
                }
                if let exportError {
                    Text(exportError)
                        .font(.caption)
                        .foregroundStyle(RuViewTheme.red)
                        .accessibilityLabel("Export error: \(exportError)")
                }
                Link(
                    "Open issue 1690 to submit feedback",
                    destination: URL(string: "https://github.com/ruvnet/RuView/issues/1690")!
                )
                .font(.caption.bold())
                .frame(minHeight: 44)
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

    private var phaseColor: Color {
        switch session.state {
        case .completed: return RuViewTheme.green
        case .failed: return RuViewTheme.red
        case .cancelled: return RuViewTheme.muted
        default: return RuViewTheme.orange
        }
    }

    private var phaseProgress: Double {
        let total = session.state == .calibration ? 15.0 : 30.0
        return max(0, min(1, (total - Double(session.metrics.phaseSecondsRemaining)) / total))
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2.bold().monospaced())
                .foregroundStyle(RuViewTheme.muted)
            Text(value)
                .font(.subheadline.bold().monospacedDigit())
                .lineLimit(1)
                .minimumScaleFactor(0.65)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 66, alignment: .leading)
        .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10).stroke(RuViewTheme.border, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var diagnosticPreview: some View {
        if let diagnostic = session.diagnostic {
            VStack(alignment: .leading, spacing: 6) {
                Text("DIAGNOSTIC PREVIEW")
                    .font(.caption.bold().monospaced())
                    .foregroundStyle(RuViewTheme.green)
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
            .foregroundStyle(RuViewTheme.textSecondary)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12).stroke(RuViewTheme.green.opacity(0.22), lineWidth: 1)
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func statusBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.bold().monospaced())
            .padding(.horizontal, 9)
            .frame(minHeight: 28)
            .background(color.opacity(0.12), in: Capsule())
            .overlay {
                Capsule().stroke(color.opacity(0.3), lineWidth: 1)
            }
            .foregroundStyle(color)
    }
}

private struct EvidencePresentation {
    let label: String
    let detail: String
    let color: Color
    let icon: String
}

private enum RuViewTheme {
    static let background = Color(red: 0.043, green: 0.055, blue: 0.075)
    static let panel = Color(red: 0.078, green: 0.094, blue: 0.122)
    static let panelStrong = Color(red: 0.094, green: 0.114, blue: 0.145)
    static let surface = Color(red: 0.113, green: 0.129, blue: 0.169)
    static let cyan = Color(red: 0.098, green: 0.831, blue: 0.902)
    static let green = Color(red: 0.149, green: 0.851, blue: 0.408)
    static let orange = Color(red: 1.000, green: 0.612, blue: 0.231)
    static let yellow = Color(red: 1.000, green: 0.827, blue: 0.318)
    static let red = Color(red: 1.000, green: 0.353, blue: 0.384)
    static let indigo = Color(red: 0.490, green: 0.584, blue: 1.000)
    static let muted = Color(red: 0.482, green: 0.537, blue: 0.616)
    static let textSecondary = Color(red: 0.482, green: 0.537, blue: 0.616)
    static let border = Color(red: 0.153, green: 0.173, blue: 0.208)
}

private struct InstrumentGrid: View {
    var body: some View {
        Canvas { context, size in
            var grid = Path()
            let spacing: CGFloat = 28
            var x: CGFloat = 0
            while x <= size.width {
                grid.move(to: CGPoint(x: x, y: 0))
                grid.addLine(to: CGPoint(x: x, y: size.height))
                x += spacing
            }
            var y: CGFloat = 0
            while y <= size.height {
                grid.move(to: CGPoint(x: 0, y: y))
                grid.addLine(to: CGPoint(x: size.width, y: y))
                y += spacing
            }
            context.stroke(grid, with: .color(RuViewTheme.cyan.opacity(0.045)), lineWidth: 0.5)

            var topLine = Path()
            topLine.move(to: CGPoint(x: 0, y: 1))
            topLine.addLine(to: CGPoint(x: size.width, y: 1))
            context.stroke(topLine, with: .color(RuViewTheme.cyan.opacity(0.8)), lineWidth: 1)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
        .ignoresSafeArea()
    }
}

private struct OrbitalSignatureView: View {
    let accent: Color

    var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let rings: [CGFloat] = [0.31, 0.55, 0.82]
            for (index, scale) in rings.enumerated() {
                let diameter = min(size.width, size.height) * scale
                let rect = CGRect(
                    x: center.x - diameter / 2,
                    y: center.y - diameter / 2,
                    width: diameter,
                    height: diameter
                )
                context.stroke(
                    Path(ellipseIn: rect),
                    with: .color(accent.opacity(index == rings.count - 1 ? 0.2 : 0.36)),
                    lineWidth: 1
                )
            }

            var crosshair = Path()
            crosshair.move(to: CGPoint(x: center.x, y: 5))
            crosshair.addLine(to: CGPoint(x: center.x, y: size.height - 5))
            crosshair.move(to: CGPoint(x: 5, y: center.y))
            crosshair.addLine(to: CGPoint(x: size.width - 5, y: center.y))
            context.stroke(crosshair, with: .color(accent.opacity(0.2)), lineWidth: 0.5)

            context.fill(
                Path(ellipseIn: CGRect(x: center.x - 3, y: center.y - 3, width: 6, height: 6)),
                with: .color(accent)
            )
            context.fill(
                Path(ellipseIn: CGRect(x: center.x + 21, y: center.y - 20, width: 5, height: 5)),
                with: .color(RuViewTheme.green)
            )
        }
    }
}

private struct InstrumentInputModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(.body.monospaced())
            .padding(.horizontal, 12)
            .frame(minHeight: 50)
            .background(RuViewTheme.surface, in: RoundedRectangle(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(RuViewTheme.border, lineWidth: 1)
            }
    }
}

private extension View {
    func instrumentInput() -> some View {
        modifier(InstrumentInputModifier())
    }
}

private struct InstrumentPrimaryButtonStyle: ButtonStyle {
    let accent: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.bold())
            .foregroundStyle(RuViewTheme.background)
            .padding(.horizontal, 14)
            .frame(maxWidth: .infinity, minHeight: 48)
            .background(accent.opacity(configuration.isPressed ? 0.72 : 1))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .shadow(color: accent.opacity(configuration.isPressed ? 0.08 : 0.22), radius: 10, y: 4)
    }
}

private struct InstrumentSecondaryButtonStyle: ButtonStyle {
    let accent: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.bold())
            .foregroundStyle(accent)
            .padding(.horizontal, 14)
            .frame(maxWidth: .infinity, minHeight: 48)
            .background(accent.opacity(configuration.isPressed ? 0.12 : 0.06))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(accent.opacity(0.42), lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
