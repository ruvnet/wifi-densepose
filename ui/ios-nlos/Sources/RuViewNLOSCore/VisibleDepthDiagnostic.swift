import Foundation

public enum VisibleDepthEvidenceLabel: String, Codable, Sendable {
    case directDepth = "direct_depth"
}

public enum VisibleDepthPhase: String, Codable, Sendable {
    case calibration
    case wallScan = "wall_scan"
}

public struct VisibleDepthCapabilityFlags: Codable, Equatable, Sendable {
    public let worldTracking: Bool
    public let sceneDepth: Bool
    public let smoothedSceneDepth: Bool
    public let sceneMesh: Bool
    public let rawPhotonHistograms: Bool

    public init(
        worldTracking: Bool,
        sceneDepth: Bool,
        smoothedSceneDepth: Bool,
        sceneMesh: Bool,
        rawPhotonHistograms: Bool = false
    ) {
        self.worldTracking = worldTracking
        self.sceneDepth = sceneDepth
        self.smoothedSceneDepth = smoothedSceneDepth
        self.sceneMesh = sceneMesh
        self.rawPhotonHistograms = rawPhotonHistograms
    }
}

public struct VisibleDepthPhaseSummary: Codable, Equatable, Sendable {
    public let phase: VisibleDepthPhase
    public let plannedDurationSeconds: Int
    public let observedDurationSeconds: Double
    public let frameCount: Int
    public let averageFPS: Double
    public let averageDepthCoverage: Double
    public let averageMovementMetersPerSecond: Double
    public let finalTrackingState: String
    public let peakThermalState: String

    public init(
        phase: VisibleDepthPhase,
        plannedDurationSeconds: Int,
        observedDurationSeconds: Double,
        frameCount: Int,
        averageFPS: Double,
        averageDepthCoverage: Double,
        averageMovementMetersPerSecond: Double,
        finalTrackingState: String,
        peakThermalState: String
    ) {
        self.phase = phase
        self.plannedDurationSeconds = plannedDurationSeconds
        self.observedDurationSeconds = observedDurationSeconds
        self.frameCount = frameCount
        self.averageFPS = averageFPS
        self.averageDepthCoverage = averageDepthCoverage
        self.averageMovementMetersPerSecond = averageMovementMetersPerSecond
        self.finalTrackingState = finalTrackingState
        self.peakThermalState = peakThermalState
    }
}

public struct VisibleDepthConsentFlags: Codable, Equatable, Sendable {
    public let localValidation: Bool
    public let diagnosticExport: Bool
    public let rawSensorExport: Bool

    public init(localValidation: Bool, diagnosticExport: Bool) {
        self.localValidation = localValidation
        self.diagnosticExport = diagnosticExport
        self.rawSensorExport = false
    }
}

public struct VisibleDepthDiagnostic: Codable, Equatable, Sendable {
    public static let schema = "ruview.ios.visible-depth-diagnostic.v1"

    public let schema: String
    public let sessionId: String
    public let createdAt: Date
    public let deviceModelFamily: String
    public let osVersion: String
    public let appVersion: String
    public let capabilities: VisibleDepthCapabilityFlags
    public let phases: [VisibleDepthPhaseSummary]
    public let consent: VisibleDepthConsentFlags
    public let evidenceLabel: VisibleDepthEvidenceLabel
    public let physicalNLOSStatus: String
    public let cameraPermission: String
    public let completionStatus: String
    public let failureReason: String?

    public init(
        sessionId: UUID,
        createdAt: Date = Date(),
        deviceModelFamily: String,
        osVersion: String,
        appVersion: String,
        capabilities: VisibleDepthCapabilityFlags,
        phases: [VisibleDepthPhaseSummary],
        consent: VisibleDepthConsentFlags,
        cameraPermission: String,
        completionStatus: String,
        failureReason: String? = nil
    ) {
        self.schema = Self.schema
        self.sessionId = sessionId.uuidString.lowercased()
        self.createdAt = createdAt
        self.deviceModelFamily = String(deviceModelFamily.prefix(80))
        self.osVersion = String(osVersion.prefix(40))
        self.appVersion = String(appVersion.prefix(40))
        self.capabilities = capabilities
        self.phases = Array(phases.prefix(2))
        self.consent = consent
        self.evidenceLabel = .directDepth
        self.physicalNLOSStatus = "blocked_raw_transients_unavailable"
        self.cameraPermission = String(cameraPermission.prefix(24))
        self.completionStatus = String(completionStatus.prefix(24))
        self.failureReason = failureReason.map { String($0.prefix(240)) }
    }

    public func encodedJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        guard data.count <= 64 * 1_024 else {
            throw VisibleDepthDiagnosticError.packageTooLarge
        }
        return data
    }
}

public enum VisibleDepthDiagnosticError: Error, Equatable {
    case packageTooLarge
}

public struct VisibleDepthPhaseAccumulator: Sendable {
    public let phase: VisibleDepthPhase
    public let plannedDurationSeconds: Int
    private var firstTimestamp: Double?
    private var lastTimestamp: Double?
    private var frameCount = 0
    private var fpsTotal = 0.0
    private var coverageTotal = 0.0
    private var movementTotal = 0.0
    private var finalTrackingState = "unavailable"
    private var peakThermalState = "unknown"

    public init(phase: VisibleDepthPhase, plannedDurationSeconds: Int) {
        self.phase = phase
        self.plannedDurationSeconds = plannedDurationSeconds
    }

    public mutating func add(
        timestamp: Double,
        fps: Double,
        depthCoverage: Double,
        movementMetersPerSecond: Double,
        trackingState: String,
        thermalState: String
    ) {
        guard timestamp.isFinite else { return }
        if firstTimestamp == nil { firstTimestamp = timestamp }
        lastTimestamp = timestamp
        frameCount += 1
        fpsTotal += Self.clampFinite(fps, upperBound: 240)
        coverageTotal += Self.clampFinite(depthCoverage, upperBound: 1)
        movementTotal += Self.clampFinite(movementMetersPerSecond, upperBound: 20)
        finalTrackingState = String(trackingState.prefix(48))
        if Self.thermalRank(thermalState) > Self.thermalRank(peakThermalState) {
            peakThermalState = String(thermalState.prefix(16))
        }
    }

    public func summary() -> VisibleDepthPhaseSummary {
        let divisor = Double(max(frameCount, 1))
        return VisibleDepthPhaseSummary(
            phase: phase,
            plannedDurationSeconds: plannedDurationSeconds,
            observedDurationSeconds: max(0, (lastTimestamp ?? 0) - (firstTimestamp ?? 0)),
            frameCount: frameCount,
            averageFPS: fpsTotal / divisor,
            averageDepthCoverage: coverageTotal / divisor,
            averageMovementMetersPerSecond: movementTotal / divisor,
            finalTrackingState: finalTrackingState,
            peakThermalState: peakThermalState
        )
    }

    private static func thermalRank(_ state: String) -> Int {
        switch state {
        case "nominal": return 1
        case "fair": return 2
        case "serious": return 3
        case "critical": return 4
        default: return 0
        }
    }

    private static func clampFinite(_ value: Double, upperBound: Double) -> Double {
        guard value.isFinite else { return 0 }
        return max(0, min(value, upperBound))
    }
}
