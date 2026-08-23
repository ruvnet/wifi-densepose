import Foundation

private struct AnyCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        intValue = nil
    }

    init?(intValue: Int) {
        stringValue = String(intValue)
        self.intValue = intValue
    }
}

private func requireExactKeys(_ decoder: Decoder, allowed: Set<String>) throws {
    let container = try decoder.container(keyedBy: AnyCodingKey.self)
    let actual = Set(container.allKeys.map(\.stringValue))
    guard actual == allowed else {
        throw DecodingError.dataCorrupted(
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Object keys do not match ruview.nlos.track.v1."
            )
        )
    }
}

public enum NLOSSource: String, Decodable, Sendable {
    case live
    case replay
    case synthetic
}

public enum EvidenceLevel: String, Decodable, Sendable, Comparable {
    case l0Synthetic = "l0_synthetic"
    case l1Measured = "l1_measured"
    case l2Calibrated = "l2_calibrated"
    case l3Corroborated = "l3_corroborated"

    private var rank: Int {
        switch self {
        case .l0Synthetic: return 0
        case .l1Measured: return 1
        case .l2Calibrated: return 2
        case .l3Corroborated: return 3
        }
    }

    public static func < (lhs: EvidenceLevel, rhs: EvidenceLevel) -> Bool {
        lhs.rank < rhs.rank
    }
}

public enum TransientKind: String, Decodable, Sendable {
    case rawHistogram = "raw_histogram"
    case compactNormalizedHistogram = "compact_normalized_histogram"
    case depthOnly = "depth_only"
    case replay
}

public enum NLOSTransport: String, Decodable, Sendable {
    case usbSerial = "usb_serial"
    case ruviewServer = "ruview_server"
    case replay
}

public enum NLOSTrackState: String, Decodable, Sendable {
    case tracking
    case degraded
    case unknown
}

public struct Vector3: Decodable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let z: Double

    private enum CodingKeys: String, CodingKey {
        case x, y, z
    }

    public init(from decoder: Decoder) throws {
        try requireExactKeys(decoder, allowed: ["x", "y", "z"])
        let container = try decoder.container(keyedBy: CodingKeys.self)
        x = try container.decode(Double.self, forKey: .x)
        y = try container.decode(Double.self, forKey: .y)
        z = try container.decode(Double.self, forKey: .z)
    }
}

public struct ModalityContributions: Decodable, Equatable, Sendable {
    public let lidar: Double
    public let csi: Double

    private enum CodingKeys: String, CodingKey {
        case lidar, csi
    }

    public init(from decoder: Decoder) throws {
        try requireExactKeys(decoder, allowed: ["lidar", "csi"])
        let container = try decoder.container(keyedBy: CodingKeys.self)
        lidar = try container.decode(Double.self, forKey: .lidar)
        csi = try container.decode(Double.self, forKey: .csi)
    }
}

public struct NLOSProvenance: Decodable, Equatable, Sendable {
    public let sensorId: String
    public let sensorModel: String
    public let firmwareVersion: String
    public let transientKind: TransientKind
    public let histogramPreserved: Bool
    public let transport: NLOSTransport

    private enum CodingKeys: String, CodingKey {
        case sensorId
        case sensorModel
        case firmwareVersion
        case transientKind
        case histogramPreserved
        case transport
    }

    public init(from decoder: Decoder) throws {
        try requireExactKeys(
            decoder,
            allowed: [
                "sensorId",
                "sensorModel",
                "firmwareVersion",
                "transientKind",
                "histogramPreserved",
                "transport",
            ]
        )
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sensorId = try container.decode(String.self, forKey: .sensorId)
        sensorModel = try container.decode(String.self, forKey: .sensorModel)
        firmwareVersion = try container.decode(String.self, forKey: .firmwareVersion)
        transientKind = try container.decode(TransientKind.self, forKey: .transientKind)
        histogramPreserved = try container.decode(Bool.self, forKey: .histogramPreserved)
        transport = try container.decode(NLOSTransport.self, forKey: .transport)
    }
}

public struct NLOSTrack: Decodable, Equatable, Identifiable, Sendable {
    public let trackId: String
    public let state: NLOSTrackState
    public let positionM: Vector3
    public let velocityMps: Vector3
    public let covarianceDiagonalM2: Vector3
    public let confidence: Double
    public let posteriorEntropy: Double
    public let signalQuality: Double
    public let modalityContributions: ModalityContributions

    public var id: String { trackId }

    private enum CodingKeys: String, CodingKey {
        case trackId
        case state
        case positionM
        case velocityMps
        case covarianceDiagonalM2
        case confidence
        case posteriorEntropy
        case signalQuality
        case modalityContributions
    }

    public init(from decoder: Decoder) throws {
        try requireExactKeys(
            decoder,
            allowed: [
                "trackId",
                "state",
                "positionM",
                "velocityMps",
                "covarianceDiagonalM2",
                "confidence",
                "posteriorEntropy",
                "signalQuality",
                "modalityContributions",
            ]
        )
        let container = try decoder.container(keyedBy: CodingKeys.self)
        trackId = try container.decode(String.self, forKey: .trackId)
        state = try container.decode(NLOSTrackState.self, forKey: .state)
        positionM = try container.decode(Vector3.self, forKey: .positionM)
        velocityMps = try container.decode(Vector3.self, forKey: .velocityMps)
        covarianceDiagonalM2 = try container.decode(Vector3.self, forKey: .covarianceDiagonalM2)
        confidence = try container.decode(Double.self, forKey: .confidence)
        posteriorEntropy = try container.decode(Double.self, forKey: .posteriorEntropy)
        signalQuality = try container.decode(Double.self, forKey: .signalQuality)
        modalityContributions = try container.decode(
            ModalityContributions.self,
            forKey: .modalityContributions
        )
    }
}

public struct NLOSTrackEnvelope: Decodable, Equatable, Sendable {
    public let schema: String
    public let sessionId: String
    public let sequence: UInt64
    public let capturedAtUnixMs: UInt64
    public let expiresAtUnixMs: UInt64
    public let source: NLOSSource
    public let evidenceLevel: EvidenceLevel
    public let algorithmVersion: String
    public let calibrationHash: String
    public let provenance: NLOSProvenance
    public let tracks: [NLOSTrack]

    private enum CodingKeys: String, CodingKey {
        case schema
        case sessionId
        case sequence
        case capturedAtUnixMs
        case expiresAtUnixMs
        case source
        case evidenceLevel
        case algorithmVersion
        case calibrationHash
        case provenance
        case tracks
    }

    public init(from decoder: Decoder) throws {
        try requireExactKeys(
            decoder,
            allowed: [
                "schema",
                "sessionId",
                "sequence",
                "capturedAtUnixMs",
                "expiresAtUnixMs",
                "source",
                "evidenceLevel",
                "algorithmVersion",
                "calibrationHash",
                "provenance",
                "tracks",
            ]
        )
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schema = try container.decode(String.self, forKey: .schema)
        sessionId = try container.decode(String.self, forKey: .sessionId)
        sequence = try container.decode(UInt64.self, forKey: .sequence)
        capturedAtUnixMs = try container.decode(UInt64.self, forKey: .capturedAtUnixMs)
        expiresAtUnixMs = try container.decode(UInt64.self, forKey: .expiresAtUnixMs)
        source = try container.decode(NLOSSource.self, forKey: .source)
        evidenceLevel = try container.decode(EvidenceLevel.self, forKey: .evidenceLevel)
        algorithmVersion = try container.decode(String.self, forKey: .algorithmVersion)
        calibrationHash = try container.decode(String.self, forKey: .calibrationHash)
        provenance = try container.decode(NLOSProvenance.self, forKey: .provenance)
        tracks = try container.decode([NLOSTrack].self, forKey: .tracks)
    }
}

/// First server message on every authenticated WebSocket connection.
public struct NLOSAuthenticatedSession: Decodable, Equatable, Sendable {
    public let schema: String
    public let sessionId: String
    public let expiresAtUnixMs: UInt64

    private enum CodingKeys: String, CodingKey {
        case schema
        case sessionId
        case expiresAtUnixMs
    }

    public init(from decoder: Decoder) throws {
        try requireExactKeys(
            decoder,
            allowed: ["schema", "sessionId", "expiresAtUnixMs"]
        )
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schema = try container.decode(String.self, forKey: .schema)
        sessionId = try container.decode(String.self, forKey: .sessionId)
        expiresAtUnixMs = try container.decode(UInt64.self, forKey: .expiresAtUnixMs)
    }
}

public struct ValidatedTrackEnvelope: Equatable, Sendable {
    public let value: NLOSTrackEnvelope

    public var visibleTracks: [NLOSTrack] {
        value.tracks.filter { $0.state != .unknown }
    }

    public var watermark: String? {
        switch value.source {
        case .synthetic: return "SYNTHETIC"
        case .replay: return "REPLAY"
        case .live: return nil
        }
    }

    public func isFresh(atUnixMs nowUnixMs: UInt64) -> Bool {
        value.expiresAtUnixMs > nowUnixMs
    }
}

public struct TrackDisplayFrame: Equatable, Sendable {
    public let sessionId: String
    public let sequence: UInt64
    public let capturedAtUnixMs: UInt64
    public let expiresAtUnixMs: UInt64
    public let source: NLOSSource
    public let evidenceLevel: EvidenceLevel
    public let algorithmVersion: String
    public let provenance: NLOSProvenance
    public let tracks: [NLOSTrack]
    public let watermark: String?
}
