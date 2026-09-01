import Foundation
import XCTest
@testable import RuViewNLOSCore

final class VisibleDepthDiagnosticTests: XCTestCase {
    func testDiagnosticIsBoundedAndAlwaysDirectDepth() throws {
        let diagnostic = VisibleDepthDiagnostic(
            sessionId: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            createdAt: Date(timeIntervalSince1970: 1_800_000_000),
            deviceModelFamily: String(repeating: "iPhone", count: 50),
            osVersion: "iOS 18.0",
            appVersion: "1.0 (1)",
            capabilities: .init(
                worldTracking: true,
                sceneDepth: true,
                smoothedSceneDepth: true,
                sceneMesh: true
            ),
            phases: [],
            consent: .init(localValidation: true, diagnosticExport: false),
            cameraPermission: "granted",
            completionStatus: "completed"
        )

        let data = try diagnostic.encodedJSON()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["evidenceLabel"] as? String, "direct_depth")
        XCTAssertNil(object["endpoint"])
        XCTAssertNil(object["token"])
        XCTAssertNil(object["rawSamples"])
        XCTAssertEqual(object["physicalNLOSStatus"] as? String, "blocked_raw_transients_unavailable")
        XCTAssertEqual(object["cameraPermission"] as? String, "granted")
        XCTAssertLessThanOrEqual(data.count, 64 * 1_024)
        XCTAssertEqual(diagnostic.deviceModelFamily.count, 80)
    }

    func testAccumulatorClampsUntrustedMetrics() {
        var accumulator = VisibleDepthPhaseAccumulator(phase: .calibration, plannedDurationSeconds: 15)
        accumulator.add(
            timestamp: 10,
            fps: 500,
            depthCoverage: 2,
            movementMetersPerSecond: 100,
            trackingState: "normal",
            thermalState: "fair"
        )
        accumulator.add(
            timestamp: 11,
            fps: -1,
            depthCoverage: -1,
            movementMetersPerSecond: -1,
            trackingState: "limited",
            thermalState: "critical"
        )

        let summary = accumulator.summary()
        XCTAssertEqual(summary.frameCount, 2)
        XCTAssertEqual(summary.observedDurationSeconds, 1)
        XCTAssertEqual(summary.averageFPS, 120)
        XCTAssertEqual(summary.averageDepthCoverage, 0.5)
        XCTAssertEqual(summary.averageMovementMetersPerSecond, 10)
        XCTAssertEqual(summary.finalTrackingState, "limited")
        XCTAssertEqual(summary.peakThermalState, "critical")
    }

    func testAccumulatorRejectsNonFiniteTimestampAndNormalizesNonFiniteMetrics() throws {
        var accumulator = VisibleDepthPhaseAccumulator(phase: .wallScan, plannedDurationSeconds: 30)
        accumulator.add(
            timestamp: .nan,
            fps: 30,
            depthCoverage: 1,
            movementMetersPerSecond: 1,
            trackingState: "normal",
            thermalState: "nominal"
        )
        accumulator.add(
            timestamp: 1,
            fps: .infinity,
            depthCoverage: .nan,
            movementMetersPerSecond: -.infinity,
            trackingState: "normal",
            thermalState: "nominal"
        )

        let summary = accumulator.summary()
        XCTAssertEqual(summary.frameCount, 1)
        XCTAssertEqual(summary.averageFPS, 0)
        XCTAssertEqual(summary.averageDepthCoverage, 0)
        XCTAssertEqual(summary.averageMovementMetersPerSecond, 0)
        XCTAssertNoThrow(try JSONEncoder().encode(summary))
    }
}
