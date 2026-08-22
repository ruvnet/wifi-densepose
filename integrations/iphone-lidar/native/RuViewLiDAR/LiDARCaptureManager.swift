import ARKit
import CoreVideo
import Foundation

@MainActor
final class LiDARCaptureManager: NSObject, ObservableObject {
    enum State: Equatable {
        case idle
        case unsupported
        case running
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var lastFrame: RuViewLiDARFrame?
    @Published private(set) var framesPerSecond: Double = 0

    let session = ARSession()
    var onFrame: (@MainActor @Sendable (RuViewLiDARFrame) -> Void)?

    private var sequence: UInt64 = 0
    private var lastTimestamp: TimeInterval?
    private let processingQueue = DispatchQueue(label: "one.ruv.lidar.capture", qos: .userInitiated)

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = processingQueue
    }

    func start(smoothed: Bool = false) {
        let configuration = ARWorldTrackingConfiguration()
        let semantic: ARConfiguration.FrameSemantics = smoothed ? .smoothedSceneDepth : .sceneDepth

        guard ARWorldTrackingConfiguration.supportsFrameSemantics(semantic) else {
            state = .unsupported
            return
        }

        configuration.frameSemantics.insert(semantic)
        configuration.worldAlignment = .gravity
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        state = .running
    }

    func stop() {
        session.pause()
        state = .idle
    }

    nonisolated private func makeFrame(from frame: ARFrame) -> RuViewLiDARFrame? {
        guard let sceneDepth = frame.sceneDepth ?? frame.smoothedSceneDepth else { return nil }
        let depthMap = sceneDepth.depthMap
        let confidenceMap = sceneDepth.confidenceMap

        guard CVPixelBufferLockBaseAddress(depthMap, .readOnly) == kCVReturnSuccess else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32,
              let depthBase = CVPixelBufferGetBaseAddress(depthMap) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let stride = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float>.size
        let pointer = depthBase.assumingMemoryBound(to: Float.self)

        var meters = [Float]()
        meters.reserveCapacity(width * height)
        for y in 0..<height {
            let row = pointer.advanced(by: y * stride)
            for x in 0..<width {
                let value = row[x]
                meters.append(value.isFinite && value > 0 ? value : 0)
            }
        }

        var confidence = [UInt8](repeating: 0, count: width * height)
        if let confidenceMap,
           CVPixelBufferGetPixelFormatType(confidenceMap) == kCVPixelFormatType_OneComponent8,
           CVPixelBufferGetWidth(confidenceMap) == width,
           CVPixelBufferGetHeight(confidenceMap) == height,
           CVPixelBufferLockBaseAddress(confidenceMap, .readOnly) == kCVReturnSuccess {
            defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
            if let confidenceBase = CVPixelBufferGetBaseAddress(confidenceMap) {
                let confidenceStride = CVPixelBufferGetBytesPerRow(confidenceMap)
                let confidencePointer = confidenceBase.assumingMemoryBound(to: UInt8.self)
                for y in 0..<height {
                    let row = confidencePointer.advanced(by: y * confidenceStride)
                    for x in 0..<width {
                        confidence[y * width + x] = row[x]
                    }
                }
            }
        }

        return RuViewLiDARFrame(
            intrinsics: frame.camera.intrinsics,
            imageResolution: frame.camera.imageResolution,
            cameraTransform: frame.camera.transform,
            depthWidth: width,
            depthHeight: height,
            depthMeters: meters,
            confidence: confidence,
            sequence: 0,
            timestamp: Date().timeIntervalSince1970
        )
    }
}

extension LiDARCaptureManager: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let base = makeFrame(from: frame) else { return }
        let frameTimestamp = frame.timestamp

        Task { @MainActor in
            sequence &+= 1
            let corrected = base.assigningSequence(sequence)

            if let previous = lastTimestamp {
                let delta = frameTimestamp - previous
                if delta > 0 { framesPerSecond = 1.0 / delta }
            }
            lastTimestamp = frameTimestamp
            lastFrame = corrected
            onFrame?(corrected)
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor in
            state = .failed(error.localizedDescription)
        }
    }
}
