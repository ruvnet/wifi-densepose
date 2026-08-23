import Foundation

public enum NLOSValidationError: Error, Equatable, LocalizedError, Sendable {
    case frameTooLarge(actualBytes: Int, maximumBytes: Int)
    case malformedEnvelope
    case invalidField(String)
    case staleFrame
    case futureDatedFrame
    case excessiveLifetime
    case replayedSequence
    case sessionChanged
    case insecureEndpoint
    case invalidPairingToken

    public var errorDescription: String? {
        switch self {
        case let .frameTooLarge(actualBytes, maximumBytes):
            return "Frame is \(actualBytes) bytes; the limit is \(maximumBytes) bytes."
        case .malformedEnvelope:
            return "Frame is not a valid ruview.nlos.track.v1 envelope."
        case let .invalidField(field):
            return "Frame failed validation for \(field)."
        case .staleFrame:
            return "Frame is stale and was hidden."
        case .futureDatedFrame:
            return "Frame timestamp is outside the allowed clock skew."
        case .excessiveLifetime:
            return "Frame lifetime exceeds the 5 second safety window."
        case .replayedSequence:
            return "Frame sequence was repeated or moved backwards."
        case .sessionChanged:
            return "Stream session changed without reconnecting."
        case .insecureEndpoint:
            return "Only a bounded wss endpoint without embedded credentials is allowed."
        case .invalidPairingToken:
            return "Pairing token must be 32 to 512 visible ASCII characters."
        }
    }
}

public struct TrackEnvelopeDecoder: Sendable {
    public static let schema = "ruview.nlos.track.v1"
    public static let authenticatedSchema = "ruview.nlos.authenticated.v1"
    public static let maximumFrameBytes = 256 * 1024
    public static let maximumTracks = 16
    public static let maximumLifetimeMs: UInt64 = 5_000
    public static let maximumFutureSkewMs: UInt64 = 1_000
    public static let maximumAuthenticationLifetimeMs: UInt64 = 60 * 60 * 1_000
    public static let maximumInteroperableSequence: UInt64 = 9_007_199_254_740_991

    public init() {}

    public func decodeAuthenticated(
        _ data: Data,
        nowUnixMs: UInt64
    ) throws -> NLOSAuthenticatedSession {
        guard data.count <= Self.maximumFrameBytes else {
            throw NLOSValidationError.frameTooLarge(
                actualBytes: data.count,
                maximumBytes: Self.maximumFrameBytes
            )
        }
        let message: NLOSAuthenticatedSession
        do {
            message = try JSONDecoder().decode(NLOSAuthenticatedSession.self, from: data)
        } catch {
            throw NLOSValidationError.malformedEnvelope
        }
        guard message.schema == Self.authenticatedSchema else {
            throw NLOSValidationError.invalidField("authenticated.schema")
        }
        guard isSafeIdentifier(message.sessionId, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("authenticated.sessionId")
        }
        guard message.expiresAtUnixMs > nowUnixMs else {
            throw NLOSValidationError.staleFrame
        }
        guard message.expiresAtUnixMs <= Self.maximumInteroperableSequence else {
            throw NLOSValidationError.invalidField("authenticated.expiresAtUnixMs")
        }
        guard message.expiresAtUnixMs - nowUnixMs
            <= Self.maximumAuthenticationLifetimeMs + Self.maximumFutureSkewMs else {
            throw NLOSValidationError.excessiveLifetime
        }
        return message
    }

    public func decode(_ data: Data, nowUnixMs: UInt64) throws -> ValidatedTrackEnvelope {
        guard data.count <= Self.maximumFrameBytes else {
            throw NLOSValidationError.frameTooLarge(
                actualBytes: data.count,
                maximumBytes: Self.maximumFrameBytes
            )
        }

        let envelope: NLOSTrackEnvelope
        do {
            envelope = try JSONDecoder().decode(NLOSTrackEnvelope.self, from: data)
        } catch {
            throw NLOSValidationError.malformedEnvelope
        }

        try validate(envelope, nowUnixMs: nowUnixMs)
        return ValidatedTrackEnvelope(value: envelope)
    }

    private func validate(_ envelope: NLOSTrackEnvelope, nowUnixMs: UInt64) throws {
        guard envelope.schema == Self.schema else {
            throw NLOSValidationError.invalidField("schema")
        }
        guard isSafeIdentifier(envelope.sessionId, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("sessionId")
        }
        guard envelope.sequence <= Self.maximumInteroperableSequence else {
            throw NLOSValidationError.invalidField("sequence")
        }
        guard envelope.capturedAtUnixMs <= Self.maximumInteroperableSequence,
              envelope.expiresAtUnixMs <= Self.maximumInteroperableSequence else {
            throw NLOSValidationError.invalidField("timestamp")
        }
        guard isSafeIdentifier(envelope.algorithmVersion, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("algorithmVersion")
        }
        guard isLowercaseSHA256(envelope.calibrationHash) else {
            throw NLOSValidationError.invalidField("calibrationHash")
        }
        guard envelope.evidenceLevel != .l3Corroborated else {
            throw NLOSValidationError.invalidField("evidenceLevel")
        }

        let zeroHash = String(repeating: "0", count: 64)
        if envelope.source == .synthetic {
            guard envelope.evidenceLevel == .l0Synthetic else {
                throw NLOSValidationError.invalidField("evidenceLevel")
            }
            guard envelope.calibrationHash == zeroHash else {
                throw NLOSValidationError.invalidField("calibrationHash")
            }
            guard envelope.provenance.transport == .replay else {
                throw NLOSValidationError.invalidField("provenance.transport")
            }
            guard envelope.provenance.transientKind == .replay else {
                throw NLOSValidationError.invalidField("provenance.transientKind")
            }
        } else if envelope.evidenceLevel >= .l2Calibrated,
                  envelope.calibrationHash == zeroHash {
            throw NLOSValidationError.invalidField("calibrationHash")
        }

        guard envelope.expiresAtUnixMs > envelope.capturedAtUnixMs else {
            throw NLOSValidationError.invalidField("expiresAtUnixMs")
        }
        guard envelope.expiresAtUnixMs - envelope.capturedAtUnixMs <= Self.maximumLifetimeMs else {
            throw NLOSValidationError.excessiveLifetime
        }
        guard envelope.expiresAtUnixMs > nowUnixMs else {
            throw NLOSValidationError.staleFrame
        }
        if envelope.capturedAtUnixMs > nowUnixMs {
            guard envelope.capturedAtUnixMs - nowUnixMs <= Self.maximumFutureSkewMs else {
                throw NLOSValidationError.futureDatedFrame
            }
        }

        try validate(envelope.provenance, source: envelope.source, evidence: envelope.evidenceLevel)

        guard envelope.tracks.count <= Self.maximumTracks else {
            throw NLOSValidationError.invalidField("tracks")
        }
        var trackIds = Set<String>()
        for track in envelope.tracks {
            try validate(track)
            guard trackIds.insert(track.trackId).inserted else {
                throw NLOSValidationError.invalidField("tracks.trackId")
            }
        }
    }

    private func validate(
        _ provenance: NLOSProvenance,
        source: NLOSSource,
        evidence: EvidenceLevel
    ) throws {
        guard isSafeIdentifier(provenance.sensorId, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("provenance.sensorId")
        }
        guard isSafeIdentifier(provenance.sensorModel, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("provenance.sensorModel")
        }
        guard isSafeIdentifier(provenance.firmwareVersion, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("provenance.firmwareVersion")
        }

        if source == .live {
            guard evidence >= .l1Measured else {
                throw NLOSValidationError.invalidField("evidenceLevel")
            }
            let isLiveHistogram = provenance.transientKind == .rawHistogram
                || provenance.transientKind == .compactNormalizedHistogram
            guard provenance.histogramPreserved,
                  isLiveHistogram,
                  provenance.transport != .replay else {
                throw NLOSValidationError.invalidField("provenance.transientKind")
            }
        } else if source == .replay {
            guard provenance.transport == .replay,
                  provenance.transientKind == .replay,
                  provenance.histogramPreserved else {
                throw NLOSValidationError.invalidField("provenance.transientKind")
            }
        }
    }

    private func validate(_ track: NLOSTrack) throws {
        guard isSafeIdentifier(track.trackId, maximumBytes: 64) else {
            throw NLOSValidationError.invalidField("tracks.trackId")
        }
        guard isFinite(track.positionM, absoluteMaximum: 100) else {
            throw NLOSValidationError.invalidField("tracks.positionM")
        }
        guard isFinite(track.velocityMps, absoluteMaximum: 20) else {
            throw NLOSValidationError.invalidField("tracks.velocityMps")
        }
        guard isFiniteNonnegative(track.covarianceDiagonalM2, maximum: 10) else {
            throw NLOSValidationError.invalidField("tracks.covarianceDiagonalM2")
        }
        guard isUnitInterval(track.confidence) else {
            throw NLOSValidationError.invalidField("tracks.confidence")
        }
        guard track.posteriorEntropy.isFinite, track.posteriorEntropy >= 0 else {
            throw NLOSValidationError.invalidField("tracks.posteriorEntropy")
        }
        guard isUnitInterval(track.signalQuality) else {
            throw NLOSValidationError.invalidField("tracks.signalQuality")
        }
        guard isUnitInterval(track.modalityContributions.lidar),
              isUnitInterval(track.modalityContributions.csi) else {
            throw NLOSValidationError.invalidField("tracks.modalityContributions")
        }
        let contributionSum = track.modalityContributions.lidar
            + track.modalityContributions.csi
        guard contributionSum >= 0.999, contributionSum <= 1.001 else {
            throw NLOSValidationError.invalidField("tracks.modalityContributions")
        }
    }

    private func isUnitInterval(_ value: Double) -> Bool {
        value.isFinite && value >= 0 && value <= 1
    }

    private func isFinite(_ vector: Vector3, absoluteMaximum: Double) -> Bool {
        [vector.x, vector.y, vector.z].allSatisfy {
            $0.isFinite && abs($0) <= absoluteMaximum
        }
    }

    private func isFiniteNonnegative(_ vector: Vector3, maximum: Double) -> Bool {
        [vector.x, vector.y, vector.z].allSatisfy {
            $0.isFinite && $0 >= 0 && $0 <= maximum
        }
    }

    private func isSafeIdentifier(_ value: String, maximumBytes: Int) -> Bool {
        guard !value.isEmpty, value.utf8.count <= maximumBytes else { return false }
        return value.unicodeScalars.allSatisfy { scalar in
            let code = scalar.value
            return (code >= 48 && code <= 57)
                || (code >= 65 && code <= 90)
                || (code >= 97 && code <= 122)
                || code == 45
                || code == 46
                || code == 58
                || code == 95
        }
    }

    private func isLowercaseSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { byte in
            (byte >= 48 && byte <= 57) || (byte >= 97 && byte <= 102)
        }
    }
}
