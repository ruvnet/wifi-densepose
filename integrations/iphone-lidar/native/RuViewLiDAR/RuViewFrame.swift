import Foundation
import simd

struct RuViewLiDARFrame: Codable, Sendable {
    struct Intrinsics: Codable, Sendable {
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
        let imageWidth: Int
        let imageHeight: Int
    }

    struct Pose: Codable, Sendable {
        let matrix: [Float]
    }

    struct Depth: Codable, Sendable {
        let width: Int
        let height: Int
        let meters: [Float]
        let confidence: [UInt8]
    }

    struct Provenance: Codable, Sendable {
        let sensor: String
        let source: String
        let privacyClass: String
        let sequence: UInt64
        let timestampNs: UInt64
        let schema: String
    }

    let type: String
    let intrinsics: Intrinsics
    let pose: Pose
    let depth: Depth
    let provenance: Provenance

    init(
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        cameraTransform: simd_float4x4,
        depthWidth: Int,
        depthHeight: Int,
        depthMeters: [Float],
        confidence: [UInt8],
        sequence: UInt64,
        timestamp: TimeInterval
    ) {
        self.type = "ruview.lidar.depth.v1"
        self.intrinsics = Intrinsics(
            fx: intrinsics.columns.0.x,
            fy: intrinsics.columns.1.y,
            cx: intrinsics.columns.2.x,
            cy: intrinsics.columns.2.y,
            imageWidth: Int(imageResolution.width),
            imageHeight: Int(imageResolution.height)
        )
        self.pose = Pose(matrix: cameraTransform.columnMajorArray)
        self.depth = Depth(
            width: depthWidth,
            height: depthHeight,
            meters: depthMeters,
            confidence: confidence
        )
        self.provenance = Provenance(
            sensor: "apple-arkit-scene-depth",
            source: "live",
            privacyClass: "geometry-only",
            sequence: sequence,
            timestampNs: UInt64(max(0, timestamp) * 1_000_000_000),
            schema: "ruview.lidar.depth.v1"
        )
    }

    func assigningSequence(_ sequence: UInt64) -> RuViewLiDARFrame {
        RuViewLiDARFrame(
            type: type,
            intrinsics: intrinsics,
            pose: pose,
            depth: depth,
            provenance: Provenance(
                sensor: provenance.sensor,
                source: provenance.source,
                privacyClass: provenance.privacyClass,
                sequence: sequence,
                timestampNs: provenance.timestampNs,
                schema: provenance.schema
            )
        )
    }

    private init(
        type: String,
        intrinsics: Intrinsics,
        pose: Pose,
        depth: Depth,
        provenance: Provenance
    ) {
        self.type = type
        self.intrinsics = intrinsics
        self.pose = pose
        self.depth = depth
        self.provenance = provenance
    }
}

private extension simd_float4x4 {
    var columnMajorArray: [Float] {
        [
            columns.0.x, columns.0.y, columns.0.z, columns.0.w,
            columns.1.x, columns.1.y, columns.1.z, columns.1.w,
            columns.2.x, columns.2.y, columns.2.z, columns.2.w,
            columns.3.x, columns.3.y, columns.3.z, columns.3.w
        ]
    }
}
