import XCTest
@testable import RuViewNLOSApple

final class AppleCapabilityProbeTests: XCTestCase {
    func testRawPhotonHistogramsAreNeverClaimedByPublicAppleProbe() {
        let report = AppleCapabilityProbe.probe()

        XCTAssertEqual(report.rawPhotonHistograms, .unavailable)
        XCTAssertFalse(report.rawPhotonHistogramReason.isEmpty)
        XCTAssertFalse(report.visibleDepthDiagnosticFlags.rawPhotonHistograms)
    }

    func testDiagnosticFlagsPreservePublicCapabilityBoundary() {
        let report = AppleNLOSCapabilityReport(
            sceneDepth: .available,
            smoothedSceneDepth: .unavailable,
            sceneMesh: .available,
            worldPose: .available,
            rawPhotonHistograms: .available,
            rawPhotonHistogramReason: "fixture"
        )

        let flags = report.visibleDepthDiagnosticFlags
        XCTAssertTrue(flags.worldTracking)
        XCTAssertTrue(flags.sceneDepth)
        XCTAssertFalse(flags.smoothedSceneDepth)
        XCTAssertTrue(flags.sceneMesh)
        XCTAssertFalse(flags.rawPhotonHistograms)
    }

    #if !canImport(ARKit)
    func testNonAppleBuildHostReportsARKitSignalsUnavailable() {
        let report = AppleCapabilityProbe.probe()

        XCTAssertEqual(report.sceneDepth, .unavailable)
        XCTAssertEqual(report.smoothedSceneDepth, .unavailable)
        XCTAssertEqual(report.sceneMesh, .unavailable)
        XCTAssertEqual(report.worldPose, .unavailable)
    }
    #endif

    #if !canImport(Security)
    func testNonAppleBuildHostDoesNotFallBackToPlaintextTokenStorage() {
        let store = KeychainPairingTokenStore()

        XCTAssertThrowsError(try store.save(String(repeating: "A", count: 32)))
        XCTAssertThrowsError(try store.load())
    }
    #endif
}
