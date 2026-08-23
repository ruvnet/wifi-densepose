import XCTest
@testable import RuViewNLOSApple

final class AppleCapabilityProbeTests: XCTestCase {
    func testRawPhotonHistogramsAreNeverClaimedByPublicAppleProbe() {
        let report = AppleCapabilityProbe.probe()

        XCTAssertEqual(report.rawPhotonHistograms, .unavailable)
        XCTAssertFalse(report.rawPhotonHistogramReason.isEmpty)
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
