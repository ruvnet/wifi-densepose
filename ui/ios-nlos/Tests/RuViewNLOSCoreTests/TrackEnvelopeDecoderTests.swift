import Foundation
import XCTest
@testable import RuViewNLOSCore

final class TrackEnvelopeDecoderTests: XCTestCase {
    private let nowUnixMs: UInt64 = 1_800_000_000_000
    private let decoder = TrackEnvelopeDecoder()

    func testValidLiveEnvelopeDecodes() throws {
        let decoded = try decoder.decode(makeEnvelope(), nowUnixMs: nowUnixMs)

        XCTAssertEqual(decoded.value.schema, TrackEnvelopeDecoder.schema)
        XCTAssertEqual(decoded.value.source, .live)
        XCTAssertEqual(decoded.value.evidenceLevel, .l2Calibrated)
        XCTAssertEqual(decoded.visibleTracks.map(\.trackId), ["target-1"])
        XCTAssertNil(decoded.watermark)
    }

    func testAuthenticatedAcknowledgementUsesExactBoundedContract() throws {
        let object: [String: Any] = [
            "schema": TrackEnvelopeDecoder.authenticatedSchema,
            "sessionId": "session-a",
            "expiresAtUnixMs": NSNumber(value: nowUnixMs + 60_000),
        ]
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let authenticated = try decoder.decodeAuthenticated(data, nowUnixMs: nowUnixMs)
        XCTAssertEqual(authenticated.sessionId, "session-a")

        var unexpected = object
        unexpected["extra"] = true
        let malformed = try JSONSerialization.data(
            withJSONObject: unexpected,
            options: [.sortedKeys]
        )
        XCTAssertThrowsError(
            try decoder.decodeAuthenticated(malformed, nowUnixMs: nowUnixMs)
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .malformedEnvelope)
        }

        var excessive = object
        excessive["expiresAtUnixMs"] = NSNumber(
            value: nowUnixMs + TrackEnvelopeDecoder.maximumAuthenticationLifetimeMs + 1_001
        )
        let excessiveData = try JSONSerialization.data(
            withJSONObject: excessive,
            options: [.sortedKeys]
        )
        XCTAssertThrowsError(
            try decoder.decodeAuthenticated(excessiveData, nowUnixMs: nowUnixMs)
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .excessiveLifetime)
        }
    }

    func testUnknownTrackIsAcceptedButNeverDisplayable() throws {
        let decoded = try decoder.decode(
            makeEnvelope(trackState: "unknown"),
            nowUnixMs: nowUnixMs
        )

        XCTAssertEqual(decoded.value.tracks.count, 1)
        XCTAssertTrue(decoded.visibleTracks.isEmpty)
    }

    func testStaleFrameFailsClosed() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(capturedAtUnixMs: nowUnixMs - 2_000, expiresAtUnixMs: nowUnixMs),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .staleFrame)
        }
    }

    func testFutureFrameOutsideSkewFailsClosed() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    capturedAtUnixMs: nowUnixMs + 1_001,
                    expiresAtUnixMs: nowUnixMs + 2_000
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .futureDatedFrame)
        }
    }

    func testLifetimeOverFiveSecondsIsRejected() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    capturedAtUnixMs: nowUnixMs - 100,
                    expiresAtUnixMs: nowUnixMs + 5_000
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .excessiveLifetime)
        }
    }

    func testLiveDepthOnlyInputCannotBecomeNLOSEvidence() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(transientKind: "depth_only", histogramPreserved: false),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(
                error as? NLOSValidationError,
                .invalidField("provenance.transientKind")
            )
        }
    }

    func testLiveFrameRequiresPreservedHistogram() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(histogramPreserved: false),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(
                error as? NLOSValidationError,
                .invalidField("provenance.transientKind")
            )
        }
    }

    func testReplayTransientCannotBeRelabeledLive() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    transientKind: "replay",
                    histogramPreserved: true,
                    transport: "replay"
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(
                error as? NLOSValidationError,
                .invalidField("provenance.transientKind")
            )
        }
    }

    func testReplayTransportCannotBeRelabeledAsLiveHistogram() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    transientKind: "compact_normalized_histogram",
                    histogramPreserved: true,
                    transport: "replay"
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(
                error as? NLOSValidationError,
                .invalidField("provenance.transientKind")
            )
        }
    }

    func testSyntheticFrameRequiresL0AndCarriesWatermark() throws {
        let zeroHash = String(repeating: "0", count: 64)
        let decoded = try decoder.decode(
            makeEnvelope(
                source: "synthetic",
                evidenceLevel: "l0_synthetic",
                calibrationHash: zeroHash,
                transientKind: "replay",
                histogramPreserved: false,
                transport: "replay"
            ),
            nowUnixMs: nowUnixMs
        )
        XCTAssertEqual(decoded.watermark, "SYNTHETIC")
        XCTAssertNoThrow(
            try decoder.decode(
                makeEnvelope(
                    source: "synthetic",
                    evidenceLevel: "l0_synthetic",
                    calibrationHash: zeroHash,
                    transientKind: "replay",
                    histogramPreserved: true,
                    transport: "replay"
                ),
                nowUnixMs: nowUnixMs
            )
        )

        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    source: "synthetic",
                    evidenceLevel: "l1_measured",
                    calibrationHash: zeroHash,
                    transientKind: "replay",
                    histogramPreserved: false,
                    transport: "replay"
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("evidenceLevel"))
        }

        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    source: "synthetic",
                    evidenceLevel: "l0_synthetic",
                    transientKind: "replay",
                    histogramPreserved: false,
                    transport: "replay"
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("calibrationHash"))
        }

        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(
                    source: "synthetic",
                    evidenceLevel: "l0_synthetic",
                    calibrationHash: zeroHash,
                    transientKind: "replay",
                    histogramPreserved: false,
                    transport: "ruview_server"
                ),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("provenance.transport"))
        }
    }

    func testCalibratedNonSyntheticFrameCannotUseZeroCalibrationHash() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(calibrationHash: String(repeating: "0", count: 64)),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("calibrationHash"))
        }
    }

    func testV1RejectsL3WithoutDualModalityLineage() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(evidenceLevel: "l3_corroborated"),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("evidenceLevel"))
        }
    }

    func testMeasuredUncalibratedReplayCanUseZeroCalibrationHash() throws {
        let decoded = try decoder.decode(
            makeEnvelope(
                source: "replay",
                evidenceLevel: "l1_measured",
                calibrationHash: String(repeating: "0", count: 64),
                transientKind: "replay",
                histogramPreserved: true,
                transport: "replay"
            ),
            nowUnixMs: nowUnixMs
        )
        XCTAssertEqual(decoded.value.evidenceLevel, .l1Measured)
        XCTAssertEqual(decoded.watermark, "REPLAY")
    }

    func testOversizedFrameIsRejectedBeforeJSONParsing() {
        let oversized = Data(
            repeating: 0x20,
            count: TrackEnvelopeDecoder.maximumFrameBytes + 1
        )

        XCTAssertThrowsError(try decoder.decode(oversized, nowUnixMs: nowUnixMs)) { error in
            XCTAssertEqual(
                error as? NLOSValidationError,
                .frameTooLarge(
                    actualBytes: TrackEnvelopeDecoder.maximumFrameBytes + 1,
                    maximumBytes: TrackEnvelopeDecoder.maximumFrameBytes
                )
            )
        }
    }

    func testPositionAndTrackCountBoundsAreEnforced() {
        XCTAssertThrowsError(
            try decoder.decode(makeEnvelope(positionX: 100.001), nowUnixMs: nowUnixMs)
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("tracks.positionM"))
        }

        let tracks = (0...TrackEnvelopeDecoder.maximumTracks).map { index in
            makeTrack(trackId: "target-\(index)")
        }
        XCTAssertThrowsError(
            try decoder.decode(makeEnvelope(tracks: tracks), nowUnixMs: nowUnixMs)
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("tracks"))
        }
    }

    func testModalityContributionsMustSumToOne() {
        var track = makeTrack(trackId: "target-1")
        track["modalityContributions"] = ["lidar": 0.8, "csi": 0.8]
        XCTAssertThrowsError(
            try decoder.decode(makeEnvelope(tracks: [track]), nowUnixMs: nowUnixMs)
        ) { error in
            XCTAssertEqual(
                error as? NLOSValidationError,
                .invalidField("tracks.modalityContributions")
            )
        }
    }

    func testSequenceMustRemainInteroperableWithWebClients() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(sequence: TrackEnvelopeDecoder.maximumInteroperableSequence + 1),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("sequence"))
        }
    }

    func testUnknownObjectKeysAreRejected() {
        XCTAssertThrowsError(
            try decoder.decode(
                makeEnvelope(includeUnknownRootKey: true),
                nowUnixMs: nowUnixMs
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .malformedEnvelope)
        }
    }

    func testDuplicateTrackIdsAreRejected() {
        let track = makeTrack(trackId: "duplicate")
        XCTAssertThrowsError(
            try decoder.decode(makeEnvelope(tracks: [track, track]), nowUnixMs: nowUnixMs)
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidField("tracks.trackId"))
        }
    }

    func testStreamGuardRejectsReplayAndSessionSwitch() throws {
        var guardrail = TrackStreamGuard()
        let first = try decoder.decode(
            makeEnvelope(sessionId: "session-a", sequence: 7),
            nowUnixMs: nowUnixMs
        )
        let repeated = try decoder.decode(
            makeEnvelope(sessionId: "session-a", sequence: 7),
            nowUnixMs: nowUnixMs
        )
        let changedSession = try decoder.decode(
            makeEnvelope(sessionId: "session-b", sequence: 8),
            nowUnixMs: nowUnixMs
        )

        XCTAssertEqual(try guardrail.accept(first, nowUnixMs: nowUnixMs).sequence, 7)
        XCTAssertThrowsError(try guardrail.accept(repeated, nowUnixMs: nowUnixMs)) { error in
            XCTAssertEqual(error as? NLOSValidationError, .replayedSequence)
        }
        XCTAssertThrowsError(try guardrail.accept(changedSession, nowUnixMs: nowUnixMs)) { error in
            XCTAssertEqual(error as? NLOSValidationError, .sessionChanged)
        }
    }

    func testReconnectAllowsNewSessionButRetainsMonotonicRule() throws {
        var guardrail = TrackStreamGuard()
        let first = try decoder.decode(
            makeEnvelope(sessionId: "session-a", sequence: 99),
            nowUnixMs: nowUnixMs
        )
        let afterReconnect = try decoder.decode(
            makeEnvelope(sessionId: "session-b", sequence: 0),
            nowUnixMs: nowUnixMs
        )

        _ = try guardrail.accept(first, nowUnixMs: nowUnixMs)
        guardrail.resetForReconnect()
        XCTAssertEqual(try guardrail.accept(afterReconnect, nowUnixMs: nowUnixMs).sessionId, "session-b")
    }

    func testWSSAndPairingTokenValidation() throws {
        let validToken = String(repeating: "A", count: 32)
        let secureURL = try XCTUnwrap(URL(string: "wss://127.0.0.1:9443/api/v1/nlos/ws"))
        XCTAssertNoThrow(
            try WSSConnectionValidator.validate(endpoint: secureURL, pairingToken: validToken)
        )
        XCTAssertEqual(
            try WSSConnectionValidator.credentialAccount(for: secureURL),
            "127.0.0.1:9443"
        )

        let insecureURL = try XCTUnwrap(URL(string: "ws://127.0.0.1:9443/nlos"))
        XCTAssertThrowsError(
            try WSSConnectionValidator.validate(endpoint: insecureURL, pairingToken: validToken)
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .insecureEndpoint)
        }

        let embeddedCredentialURL = try XCTUnwrap(URL(string: "wss://user@example.test/nlos"))
        XCTAssertThrowsError(
            try WSSConnectionValidator.validate(
                endpoint: embeddedCredentialURL,
                pairingToken: validToken
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .insecureEndpoint)
        }

        for unsafe in [
            "wss://example.test/nlos/v1/tracks",
            "wss://example.test/api/v1/nlos/ws?token=secret",
            "wss://example.test/api/v1/nlos/ws#fragment",
        ] {
            let endpoint = try XCTUnwrap(URL(string: unsafe))
            XCTAssertThrowsError(
                try WSSConnectionValidator.validate(endpoint: endpoint, pairingToken: validToken)
            ) { error in
                XCTAssertEqual(error as? NLOSValidationError, .insecureEndpoint)
            }
        }

        XCTAssertThrowsError(
            try WSSConnectionValidator.validate(
                endpoint: secureURL,
                pairingToken: "valid-looking-token\r\nInjected: yes"
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidPairingToken)
        }

        XCTAssertThrowsError(
            try WSSConnectionValidator.validate(
                endpoint: secureURL,
                pairingToken: String(repeating: "A", count: 31)
            )
        ) { error in
            XCTAssertEqual(error as? NLOSValidationError, .invalidPairingToken)
        }
    }

    private func makeEnvelope(
        sessionId: String = "session-a",
        sequence: UInt64 = 7,
        source: String = "live",
        evidenceLevel: String = "l2_calibrated",
        calibrationHash: String = String(repeating: "a", count: 64),
        transientKind: String = "raw_histogram",
        histogramPreserved: Bool = true,
        transport: String = "ruview_server",
        trackState: String = "tracking",
        positionX: Double = 1.25,
        capturedAtUnixMs: UInt64? = nil,
        expiresAtUnixMs: UInt64? = nil,
        tracks: [[String: Any]]? = nil,
        includeUnknownRootKey: Bool = false
    ) -> Data {
        let captured = capturedAtUnixMs ?? nowUnixMs - 100
        let expires = expiresAtUnixMs ?? nowUnixMs + 1_000
        let resolvedTracks = tracks ?? [
            makeTrack(trackId: "target-1", state: trackState, positionX: positionX),
        ]
        var object: [String: Any] = [
            "schema": TrackEnvelopeDecoder.schema,
            "sessionId": sessionId,
            "sequence": NSNumber(value: sequence),
            "capturedAtUnixMs": NSNumber(value: captured),
            "expiresAtUnixMs": NSNumber(value: expires),
            "source": source,
            "evidenceLevel": evidenceLevel,
            "algorithmVersion": "consumer-nlos-0.1.0",
            "calibrationHash": calibrationHash,
            "provenance": [
                "sensorId": "spad-01",
                "sensorModel": "VL53L8CH",
                "firmwareVersion": "1.0.0",
                "transientKind": transientKind,
                "histogramPreserved": histogramPreserved,
                "transport": transport,
            ],
            "tracks": resolvedTracks,
        ]
        if includeUnknownRootKey {
            object["unexpected"] = true
        }
        return try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func makeTrack(
        trackId: String,
        state: String = "tracking",
        positionX: Double = 1.25
    ) -> [String: Any] {
        [
            "trackId": trackId,
            "state": state,
            "positionM": ["x": positionX, "y": 0.4, "z": -2.5],
            "velocityMps": ["x": 0.1, "y": 0, "z": -0.2],
            "covarianceDiagonalM2": ["x": 0.04, "y": 0.09, "z": 0.04],
            "confidence": 0.88,
            "posteriorEntropy": 0.32,
            "signalQuality": 0.74,
            "modalityContributions": ["lidar": 0.7, "csi": 0.3],
        ]
    }
}
