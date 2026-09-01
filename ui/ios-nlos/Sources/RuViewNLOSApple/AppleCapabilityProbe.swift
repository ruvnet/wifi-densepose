import Foundation
import RuViewNLOSCore

#if os(iOS) && canImport(ARKit)
import ARKit
#endif

public enum AppleCapabilityAvailability: String, Equatable, Sendable {
    case available
    case unavailable
}

public struct AppleNLOSCapabilityReport: Equatable, Sendable {
    public let sceneDepth: AppleCapabilityAvailability
    public let smoothedSceneDepth: AppleCapabilityAvailability
    public let sceneMesh: AppleCapabilityAvailability
    public let worldPose: AppleCapabilityAvailability
    public let rawPhotonHistograms: AppleCapabilityAvailability
    public let rawPhotonHistogramReason: String

    public init(
        sceneDepth: AppleCapabilityAvailability,
        smoothedSceneDepth: AppleCapabilityAvailability,
        sceneMesh: AppleCapabilityAvailability,
        worldPose: AppleCapabilityAvailability,
        rawPhotonHistograms: AppleCapabilityAvailability,
        rawPhotonHistogramReason: String
    ) {
        self.sceneDepth = sceneDepth
        self.smoothedSceneDepth = smoothedSceneDepth
        self.sceneMesh = sceneMesh
        self.worldPose = worldPose
        self.rawPhotonHistograms = rawPhotonHistograms
        self.rawPhotonHistogramReason = rawPhotonHistogramReason
    }
}

public extension AppleNLOSCapabilityReport {
    var visibleDepthDiagnosticFlags: VisibleDepthCapabilityFlags {
        VisibleDepthCapabilityFlags(
            worldTracking: worldPose == .available,
            sceneDepth: sceneDepth == .available,
            smoothedSceneDepth: smoothedSceneDepth == .available,
            sceneMesh: sceneMesh == .available,
            rawPhotonHistograms: false
        )
    }
}

public enum AppleCapabilityProbe {
    public static func probe() -> AppleNLOSCapabilityReport {
        #if os(iOS) && canImport(ARKit)
        let sceneDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        let smoothedDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
        let mesh = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
        let pose = ARWorldTrackingConfiguration.isSupported

        return AppleNLOSCapabilityReport(
            sceneDepth: sceneDepth ? .available : .unavailable,
            smoothedSceneDepth: smoothedDepth ? .available : .unavailable,
            sceneMesh: mesh ? .available : .unavailable,
            worldPose: pose ? .available : .unavailable,
            rawPhotonHistograms: .unavailable,
            rawPhotonHistogramReason: "Public ARKit APIs expose derived depth, mesh, and pose, not the raw per-zone photon timing histograms required by this NLOS pipeline."
        )
        #else
        return AppleNLOSCapabilityReport(
            sceneDepth: .unavailable,
            smoothedSceneDepth: .unavailable,
            sceneMesh: .unavailable,
            worldPose: .unavailable,
            rawPhotonHistograms: .unavailable,
            rawPhotonHistogramReason: "ARKit is unavailable on this build host. Raw photon histograms are not exposed by public Apple APIs."
        )
        #endif
    }
}
