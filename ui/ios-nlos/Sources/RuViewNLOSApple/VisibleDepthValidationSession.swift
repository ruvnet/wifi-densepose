import Foundation
import RuViewNLOSCore

#if os(iOS) && canImport(ARKit) && canImport(AVFoundation) && canImport(Combine)
@preconcurrency import ARKit
@preconcurrency import AVFoundation
import Combine
import CoreVideo
import simd

public enum VisibleDepthRunState: Equatable {
    case idle
    case requestingPermission
    case calibration
    case wallScan
    case completed
    case cancelled
    case failed
}

public struct VisibleDepthLiveMetrics: Equatable {
    public let fps: Double
    public let depthCoverage: Double
    public let trackingState: String
    public let movementMetersPerSecond: Double
    public let thermalState: String
    public let phaseSecondsRemaining: Int

    public init(
        fps: Double = 0,
        depthCoverage: Double = 0,
        trackingState: String = "unavailable",
        movementMetersPerSecond: Double = 0,
        thermalState: String = "unknown",
        phaseSecondsRemaining: Int = 15
    ) {
        self.fps = fps
        self.depthCoverage = depthCoverage
        self.trackingState = trackingState
        self.movementMetersPerSecond = movementMetersPerSecond
        self.thermalState = thermalState
        self.phaseSecondsRemaining = phaseSecondsRemaining
    }
}

@MainActor
public final class VisibleDepthValidationSession: NSObject, ObservableObject {
    @Published public private(set) var state: VisibleDepthRunState = .idle
    @Published public private(set) var metrics = VisibleDepthLiveMetrics()
    @Published public private(set) var diagnostic: VisibleDepthDiagnostic?
    @Published public private(set) var statusMessage = "Ready for an explicit local validation run."

    private let session = ARSession()
    private var sessionId = UUID()
    private var phaseStartTimestamp: Double?
    private var previousFrameTimestamp: Double?
    private var previousPosition: SIMD3<Float>?
    private var calibration = VisibleDepthPhaseAccumulator(phase: .calibration, plannedDurationSeconds: 15)
    private var wallScan = VisibleDepthPhaseAccumulator(phase: .wallScan, plannedDurationSeconds: 30)
    private var deviceModelFamily = "Apple mobile device"
    private var osVersion = "unknown"
    private var appVersion = "unknown"
    private var exportConsent = false
    private var cameraPermission = "not_requested"
    private var phaseStartUptime: TimeInterval?
    private var lastFrameUptime: TimeInterval?
    private var timerTask: Task<Void, Never>?

    public override init() {
        super.init()
        session.delegate = self
    }

    public func start(
        deviceModelFamily: String,
        osVersion: String,
        appVersion: String
    ) {
        guard state != .requestingPermission && state != .calibration && state != .wallScan else { return }
        self.deviceModelFamily = deviceModelFamily
        self.osVersion = osVersion
        self.appVersion = appVersion
        resetRun()

        guard ARWorldTrackingConfiguration.isSupported,
              ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) ||
              ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
        else {
            fail("This device does not expose ARKit scene depth. Use a LiDAR equipped iPhone Pro or iPad Pro.")
            return
        }

        state = .requestingPermission
        statusMessage = "Waiting for camera permission. No image will be saved."
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraPermission = "granted"
            beginARSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    self?.cameraPermission = granted ? "granted" : "denied"
                    if granted { self?.beginARSession() }
                    else { self?.fail("Camera permission was declined. Enable it in Settings to run visible depth validation.") }
                }
            }
        case .restricted:
            cameraPermission = "restricted"
            fail("Camera permission is restricted on this device. Ask the device administrator before retrying.")
        default:
            cameraPermission = "denied"
            fail("Camera permission is unavailable. Enable it in Settings to run visible depth validation.")
        }
    }

    public func cancel() {
        guard state == .requestingPermission || state == .calibration || state == .wallScan else { return }
        session.pause()
        timerTask?.cancel()
        timerTask = nil
        state = .cancelled
        statusMessage = "Validation cancelled. Aggregate metrics are available locally; no sensor samples were saved."
        finish(status: "cancelled", failureReason: nil)
    }

    public func setExportConsent(_ consent: Bool) {
        exportConsent = consent
        rebuildDiagnosticIfFinished()
    }

    private func resetRun() {
        session.pause()
        sessionId = UUID()
        phaseStartTimestamp = nil
        previousFrameTimestamp = nil
        previousPosition = nil
        calibration = VisibleDepthPhaseAccumulator(phase: .calibration, plannedDurationSeconds: 15)
        wallScan = VisibleDepthPhaseAccumulator(phase: .wallScan, plannedDurationSeconds: 30)
        metrics = VisibleDepthLiveMetrics()
        diagnostic = nil
        exportConsent = false
        cameraPermission = "not_requested"
        phaseStartUptime = nil
        lastFrameUptime = nil
        timerTask?.cancel()
        timerTask = nil
        state = .idle
    }

    private func beginARSession() {
        guard state == .requestingPermission else { return }
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics = .smoothedSceneDepth
        } else {
            configuration.frameSemantics = .sceneDepth
        }
        configuration.worldAlignment = .gravity
        state = .calibration
        phaseStartUptime = ProcessInfo.processInfo.systemUptime
        statusMessage = "Calibration: point at a visible textured surface and move slowly for 15 seconds."
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        startTimer()
    }

    private func consume(_ frame: ARFrame) {
        guard state == .calibration || state == .wallScan else { return }
        if phaseStartTimestamp == nil { phaseStartTimestamp = frame.timestamp }
        lastFrameUptime = ProcessInfo.processInfo.systemUptime

        let delta = max(0.0001, frame.timestamp - (previousFrameTimestamp ?? frame.timestamp - (1.0 / 60.0)))
        let fps = min(240, 1.0 / delta)
        let position = SIMD3<Float>(
            frame.camera.transform.columns.3.x,
            frame.camera.transform.columns.3.y,
            frame.camera.transform.columns.3.z
        )
        let movement = previousPosition.map { Double(simd_distance(position, $0)) / delta } ?? 0
        let coverage = Self.depthCoverage(frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap)
        let tracking = Self.trackingDescription(frame.camera.trackingState)
        let newMetrics = VisibleDepthLiveMetrics(
            fps: fps,
            depthCoverage: coverage,
            trackingState: tracking,
            movementMetersPerSecond: movement,
            thermalState: Self.thermalDescription(ProcessInfo.processInfo.thermalState),
            phaseSecondsRemaining: metrics.phaseSecondsRemaining
        )
        metrics = newMetrics

        if state == .calibration {
            calibration.add(timestamp: frame.timestamp, fps: fps, depthCoverage: coverage, movementMetersPerSecond: movement, trackingState: tracking, thermalState: newMetrics.thermalState)
        } else {
            wallScan.add(timestamp: frame.timestamp, fps: fps, depthCoverage: coverage, movementMetersPerSecond: movement, trackingState: tracking, thermalState: newMetrics.thermalState)
        }
        previousFrameTimestamp = frame.timestamp
        previousPosition = position
    }

    private func fail(_ reason: String) {
        session.pause()
        timerTask?.cancel()
        timerTask = nil
        state = .failed
        statusMessage = reason
        finish(status: "failed", failureReason: reason)
    }

    private func finish(status: String, failureReason: String?) {
        let report = AppleCapabilityProbe.probe()
        diagnostic = VisibleDepthDiagnostic(
            sessionId: sessionId,
            deviceModelFamily: deviceModelFamily,
            osVersion: osVersion,
            appVersion: appVersion,
            capabilities: report.visibleDepthDiagnosticFlags,
            phases: [calibration.summary(), wallScan.summary()],
            consent: .init(localValidation: true, diagnosticExport: exportConsent),
            cameraPermission: cameraPermission,
            completionStatus: status,
            failureReason: failureReason
        )
    }

    private func rebuildDiagnosticIfFinished() {
        guard let diagnostic else { return }
        finish(status: diagnostic.completionStatus, failureReason: diagnostic.failureReason)
    }

    private func startTimer() {
        timerTask?.cancel()
        timerTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 250_000_000)
                guard let self else { return }
                self.advanceTimer(now: ProcessInfo.processInfo.systemUptime)
                guard self.state == .calibration || self.state == .wallScan else { return }
            }
        }
    }

    private func advanceTimer(now: TimeInterval) {
        guard state == .calibration || state == .wallScan,
              let phaseStartUptime
        else { return }
        let duration = state == .calibration ? 15.0 : 30.0
        let elapsed = max(0, now - phaseStartUptime)
        metrics = VisibleDepthLiveMetrics(
            fps: metrics.fps,
            depthCoverage: metrics.depthCoverage,
            trackingState: metrics.trackingState,
            movementMetersPerSecond: metrics.movementMetersPerSecond,
            thermalState: Self.thermalDescription(ProcessInfo.processInfo.thermalState),
            phaseSecondsRemaining: max(0, Int(ceil(duration - elapsed)))
        )
        guard elapsed >= duration else { return }

        let summary = state == .calibration ? calibration.summary() : wallScan.summary()
        guard summary.frameCount >= 15, summary.averageDepthCoverage > 0 else {
            fail("Visible depth frames were unavailable or empty. Face a matte surface one to two metres away and retry.")
            return
        }
        guard let lastFrameUptime, now - lastFrameUptime <= 2 else {
            fail("The ARKit frame stream stopped before this phase completed. Keep the app active and retry.")
            return
        }

        if state == .calibration {
            state = .wallScan
            self.phaseStartUptime = now
            phaseStartTimestamp = nil
            previousFrameTimestamp = nil
            previousPosition = nil
            metrics = VisibleDepthLiveMetrics(phaseSecondsRemaining: 30)
            statusMessage = "Wall scan: keep the visible wall in frame and move slowly side to side for 30 seconds."
        } else {
            session.pause()
            timerTask?.cancel()
            timerTask = nil
            state = .completed
            statusMessage = "Visible depth validation complete. Results remain labeled direct_depth, never NLOS."
            finish(status: "completed", failureReason: nil)
        }
    }

    private static func depthCoverage(_ buffer: CVPixelBuffer?) -> Double {
        guard let buffer else { return 0 }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_DepthFloat32,
              let address = CVPixelBufferGetBaseAddress(buffer)
        else { return 0 }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float32>.stride
        let pixels = address.assumingMemoryBound(to: Float32.self)
        var valid = 0
        var measured = 0
        for y in Swift.stride(from: 0, to: height, by: 8) {
            for x in Swift.stride(from: 0, to: width, by: 8) {
                let value = pixels[(y * stride) + x]
                if value.isFinite && value > 0 { valid += 1 }
                measured += 1
            }
        }
        return measured == 0 ? 0 : Double(valid) / Double(measured)
    }

    private static func trackingDescription(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "not_available"
        case let .limited(reason):
            switch reason {
            case .excessiveMotion: return "limited_excessive_motion"
            case .insufficientFeatures: return "limited_insufficient_features"
            case .initializing: return "limited_initializing"
            case .relocalizing: return "limited_relocalizing"
            @unknown default: return "limited_unknown"
            }
        }
    }

    private static func thermalDescription(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

extension VisibleDepthValidationSession: ARSessionDelegate {
    nonisolated public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task { @MainActor [weak self] in self?.consume(frame) }
    }

    nonisolated public func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor [weak self] in self?.fail("ARKit validation stopped: \(error.localizedDescription)") }
    }

    nonisolated public func sessionWasInterrupted(_ session: ARSession) {
        Task { @MainActor [weak self] in self?.fail("ARKit validation was interrupted. Start a new run when the app is active.") }
    }
}
#endif
