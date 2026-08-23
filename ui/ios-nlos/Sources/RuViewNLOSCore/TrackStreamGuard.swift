import Foundation

public struct TrackStreamGuard: Sendable {
    private var boundSessionId: String?
    private var lastSequence: UInt64?

    public init() {}

    public mutating func resetForReconnect() {
        boundSessionId = nil
        lastSequence = nil
    }

    public mutating func accept(
        _ envelope: ValidatedTrackEnvelope,
        nowUnixMs: UInt64
    ) throws -> TrackDisplayFrame {
        guard envelope.isFresh(atUnixMs: nowUnixMs) else {
            throw NLOSValidationError.staleFrame
        }

        let value = envelope.value
        if let boundSessionId {
            guard value.sessionId == boundSessionId else {
                throw NLOSValidationError.sessionChanged
            }
        }
        if let lastSequence {
            guard value.sequence > lastSequence else {
                throw NLOSValidationError.replayedSequence
            }
        }

        boundSessionId = value.sessionId
        lastSequence = value.sequence

        return TrackDisplayFrame(
            sessionId: value.sessionId,
            sequence: value.sequence,
            capturedAtUnixMs: value.capturedAtUnixMs,
            expiresAtUnixMs: value.expiresAtUnixMs,
            source: value.source,
            evidenceLevel: value.evidenceLevel,
            algorithmVersion: value.algorithmVersion,
            provenance: value.provenance,
            tracks: envelope.visibleTracks,
            watermark: envelope.watermark
        )
    }
}

public enum WSSConnectionValidator {
    public static func validate(endpoint: URL, pairingToken: String) throws {
        _ = try credentialAccount(for: endpoint)
        try validatePairingToken(pairingToken)
    }

    /// Return a normalized Keychain account only for the exact NLOS socket
    /// endpoint. Credentials are thereby bound to one authority.
    public static func credentialAccount(for endpoint: URL) throws -> String {
        let components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
        guard endpoint.absoluteString.utf8.count <= 2_048,
              endpoint.scheme?.lowercased() == "wss",
              let host = endpoint.host,
              !host.isEmpty,
              endpoint.user == nil,
              endpoint.password == nil,
              endpoint.fragment == nil,
              endpoint.path == "/api/v1/nlos/ws",
              components?.percentEncodedQuery == nil else {
            throw NLOSValidationError.insecureEndpoint
        }

        if let port = endpoint.port, !(1...65_535).contains(port) {
            throw NLOSValidationError.insecureEndpoint
        }

        return "\(host.lowercased()):\(endpoint.port ?? 443)"
    }

    public static func validatePairingToken(_ pairingToken: String) throws {
        guard (32...512).contains(pairingToken.utf8.count),
              pairingToken.utf8.allSatisfy({ $0 >= 0x21 && $0 <= 0x7e }) else {
            throw NLOSValidationError.invalidPairingToken
        }
    }
}
