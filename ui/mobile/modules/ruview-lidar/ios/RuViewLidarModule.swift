import ARKit
import AVFoundation
import CoreImage
import ExpoModulesCore
import Foundation
import RoomPlan
import simd
import UIKit
import Vision

private let pointFrameEvent = "onLidarFrame"
private let depthPacketEvent = "onLidarDepthPacket"
private let cameraPreviewEvent = "onCameraPreview"
private let bodyTeacherEvent = "onBodyTeacherFrame"
private let statusEvent = "onLidarStatus"
private let roomUpdateEvent = "onRoomUpdate"
private let roomCompleteEvent = "onRoomComplete"
private let lidarErrorEvent = "onLidarError"
private let validationMetricsEvent = "onVisibleDepthMetrics"
private let validationDiagnosticEvent = "onVisibleDepthDiagnostic"

struct LidarCaptureOptions: Record {
  @Field var maxPoints: Int = 1024
  @Field var maxFramesPerSecond: Double = 5
  @Field var minimumConfidence: Int = 1
  @Field var maximumDepthMeters: Double = 8
  @Field var useSmoothedDepth: Bool = true
  @Field var includeCameraPreview: Bool = false
  @Field var includeBodyTeacher: Bool = false
  @Field var maxBodyFramesPerSecond: Double = 15
  @Field var preserveRoomCoordinateFrame: Bool = false
}

public final class RuViewLidarModule: Module {
  private let captureController = RuViewLidarCaptureController()

  public func definition() -> ModuleDefinition {
    Name("RuViewLidar")
    Events(pointFrameEvent, depthPacketEvent, cameraPreviewEvent, bodyTeacherEvent, statusEvent, roomUpdateEvent, roomCompleteEvent, lidarErrorEvent, validationMetricsEvent, validationDiagnosticEvent)

    OnCreate { [weak self] in
      self?.captureController.eventSink = { [weak self] event, payload in
        self?.sendEvent(event, payload)
      }
    }

    OnDestroy { [weak self] in
      Task { @MainActor [weak self] in self?.captureController.stopAll() }
      self?.captureController.eventSink = nil
    }

    AsyncFunction("getCapabilities") { [weak self] in
      self?.captureController.capabilities() ?? RuViewLidarCaptureController.unsupportedCapabilities()
    }

    AsyncFunction("startDepthCapture") { [weak self] (options: LidarCaptureOptions) async throws -> [String: Any] in
      guard let self else { throw LidarModuleError.unavailable }
      try await self.captureController.requestCameraAccess()
      return try await MainActor.run {
        try self.captureController.startDepthCapture(options: options)
      }
    }

    AsyncFunction("stopCapture") { [weak self] () async -> [String: Any] in
      guard let self else { return ["state": "idle"] }
      return await MainActor.run { self.captureController.stopAll() }
    }

    AsyncFunction("startRoomCapture") { [weak self] () async throws -> [String: Any] in
      guard let self else { throw LidarModuleError.unavailable }
      try await self.captureController.requestCameraAccess()
      return try await MainActor.run { try self.captureController.startRoomCapture() }
    }

    AsyncFunction("stopRoomCapture") { [weak self] () async -> [String: Any] in
      guard let self else { return ["state": "idle"] }
      return await MainActor.run { self.captureController.stopRoomCapture() }
    }

    AsyncFunction("getLatestRoom") { [weak self] () -> [String: Any]? in
      self?.captureController.latestRoomPayload
    }

    AsyncFunction("getCurrentPose") { [weak self] () -> [String: Any]? in
      self?.captureController.currentPosePayload()
    }

    AsyncFunction("startVisibleDepthValidation") { [weak self] () async throws -> [String: Any] in
      guard let self else { throw LidarModuleError.unavailable }
      try await self.captureController.requestCameraAccess()
      return try await MainActor.run { try self.captureController.startVisibleDepthValidation() }
    }

    AsyncFunction("cancelVisibleDepthValidation") { [weak self] () async -> [String: Any] in
      guard let self else { return ["state": "idle"] }
      return await MainActor.run { self.captureController.cancelVisibleDepthValidation() }
    }
  }
}

private enum LidarModuleError: Error, LocalizedError {
  case cameraDenied
  case lidarUnsupported
  case roomPlanUnsupported
  case unavailable
  case roomFrameUnavailable

  var errorDescription: String? {
    switch self {
    case .cameraDenied: return "Camera access is required for ARKit LiDAR capture."
    case .lidarUnsupported: return "This device does not support ARKit scene depth."
    case .roomPlanUnsupported: return "RoomPlan is not supported on this device."
    case .unavailable: return "The RuView LiDAR module is unavailable."
    case .roomFrameUnavailable: return "Complete a RoomPlan scan before starting room-aligned pose teaching."
    }
  }
}

private final class RuViewLidarCaptureController: NSObject, ARSessionDelegate {
  var eventSink: ((String, [String: Any]) -> Void)?
  private let arSession = ARSession()
  private let processingQueue = DispatchQueue(label: "com.ruvnet.lidar.processing", qos: .userInitiated)
  private let stateLock = NSLock()
  private var captureOptions = LidarCaptureOptions()
  private var sequence = 0
  private var lastDeliveredTimestamp: TimeInterval = 0
  private var lastCameraPreviewTimestamp: TimeInterval = 0
  private var lastBodyTeacherTimestamp: TimeInterval = 0
  private let imageContext = CIContext(options: [.cacheIntermediates: false])
  private var wallClockOffsetMs: Double?
  private var sessionId = UUID().uuidString.lowercased()
  private var coordinateFrameId = UUID().uuidString.lowercased()
  private var roomController: AnyObject?
  private var storedLatestRoomPayload: [String: Any]?
  private var validationPhase: String?
  private var validationSessionId = UUID().uuidString.lowercased()
  private var validationPhaseStartUptime: TimeInterval = 0
  private var validationLastFrameUptime: TimeInterval = 0
  private var validationPreviousTimestamp: TimeInterval?
  private var validationPreviousPosition: SIMD3<Float>?
  private var validationCalibration = ValidationAccumulator(phase: "calibration", plannedDurationSeconds: 15)
  private var validationWallScan = ValidationAccumulator(phase: "wall_scan", plannedDurationSeconds: 30)
  private var validationTimer: DispatchSourceTimer?

  var latestRoomPayload: [String: Any]? {
    stateLock.lock()
    defer { stateLock.unlock() }
    return storedLatestRoomPayload
  }

  override init() {
    super.init()
    arSession.delegateQueue = processingQueue
  }

  static func unsupportedCapabilities() -> [String: Any] {
    [
      "platform": "ios",
      "lidarSupported": false,
      "sceneDepthSupported": false,
      "smoothedSceneDepthSupported": false,
      "sceneReconstructionSupported": false,
      "roomPlanSupported": false,
      "rawTransientTimingAvailable": false,
    ]
  }

  func capabilities() -> [String: Any] {
    let sceneDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    let smoothedDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
    let mesh = ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
    let roomPlan: Bool
    if #available(iOS 16.0, *) { roomPlan = RoomCaptureSession.isSupported } else { roomPlan = false }
    return [
      "platform": "ios",
      "lidarSupported": sceneDepth,
      "sceneDepthSupported": sceneDepth,
      "smoothedSceneDepthSupported": smoothedDepth,
      "sceneReconstructionSupported": mesh,
      "roomPlanSupported": roomPlan,
      "rawTransientTimingAvailable": false,
    ]
  }

  func requestCameraAccess() async throws {
    let status = AVCaptureDevice.authorizationStatus(for: .video)
    if status == .authorized { return }
    if status == .denied || status == .restricted { throw LidarModuleError.cameraDenied }
    emit(statusEvent, ["state": "requesting_permission"])
    let allowed = await withCheckedContinuation { continuation in
      AVCaptureDevice.requestAccess(for: .video) { continuation.resume(returning: $0) }
    }
    if !allowed { throw LidarModuleError.cameraDenied }
  }

  func currentPosePayload() -> [String: Any]? {
    if #available(iOS 16.0, *), let controller = roomController as? RuViewRoomCaptureController {
      return controller.currentPosePayload()
    }
    guard let frame = arSession.currentFrame else { return nil }
    return spatialPosePayload(
      transform: frame.camera.transform,
      coordinateFrameId: coordinateFrameId,
      trackingState: trackingLabel(frame.camera.trackingState)
    )
  }

  @MainActor
  func startDepthCapture(options: LidarCaptureOptions) throws -> [String: Any] {
    guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
      emit(statusEvent, ["state": "unsupported", "message": LidarModuleError.lidarUnsupported.localizedDescription])
      throw LidarModuleError.lidarUnsupported
    }

    let preserveRoomFrame = options.preserveRoomCoordinateFrame
      && (storedLatestRoomPayload?["coordinateFrameId"] as? String) == coordinateFrameId
    if options.preserveRoomCoordinateFrame && !preserveRoomFrame { throw LidarModuleError.roomFrameUnavailable }
    _ = stopAll()
    captureOptions = bounded(options)
    sessionId = UUID().uuidString.lowercased()
    if !preserveRoomFrame { coordinateFrameId = UUID().uuidString.lowercased() }
    sequence = 0
    lastDeliveredTimestamp = 0
    lastCameraPreviewTimestamp = 0
    lastBodyTeacherTimestamp = 0
    wallClockOffsetMs = nil
    arSession.delegate = self

    let configuration = ARWorldTrackingConfiguration()
    configuration.worldAlignment = .gravity
    configuration.planeDetection = [.horizontal, .vertical]
    if captureOptions.useSmoothedDepth,
       ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
      configuration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
    } else {
      configuration.frameSemantics = [.sceneDepth]
    }
    if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
      configuration.sceneReconstruction = .meshWithClassification
    }
    arSession.run(configuration, options: preserveRoomFrame ? [] : [.resetTracking, .removeExistingAnchors])
    let payload: [String: Any] = [
      "state": "capturing_depth",
      "sessionId": sessionId,
      "coordinateFrameId": coordinateFrameId,
    ]
    emit(statusEvent, payload)
    return payload
  }

  @MainActor
  func startRoomCapture() throws -> [String: Any] {
    guard #available(iOS 16.0, *), RoomCaptureSession.isSupported else {
      emit(statusEvent, ["state": "unsupported", "message": LidarModuleError.roomPlanUnsupported.localizedDescription])
      throw LidarModuleError.roomPlanUnsupported
    }
    _ = stopAll()
    coordinateFrameId = UUID().uuidString.lowercased()
    stateLock.lock()
    storedLatestRoomPayload = nil
    stateLock.unlock()
    let controller = RuViewRoomCaptureController(arSession: arSession, coordinateFrameId: coordinateFrameId) { [weak self] event, payload in
      if event == roomCompleteEvent {
        self?.stateLock.lock()
        self?.storedLatestRoomPayload = payload
        self?.stateLock.unlock()
        DispatchQueue.main.async { [weak self] in self?.roomController = nil }
      }
      self?.emit(event, payload)
    }
    roomController = controller
    controller.start()
    let payload: [String: Any] = [
      "state": "capturing_room",
      "coordinateFrameId": coordinateFrameId,
    ]
    emit(statusEvent, payload)
    return payload
  }

  @MainActor
  @discardableResult
  func stopRoomCapture() -> [String: Any] {
    if #available(iOS 16.0, *), let controller = roomController as? RuViewRoomCaptureController {
      controller.stop()
      let payload: [String: Any] = ["state": "processing_room"]
      emit(statusEvent, payload)
      return payload
    }
    return ["state": "idle"]
  }

  @MainActor
  @discardableResult
  func stopAll() -> [String: Any] {
    if #available(iOS 16.0, *), let controller = roomController as? RuViewRoomCaptureController {
      controller.stop()
      roomController = nil
    }
    arSession.pause()
    arSession.delegate = nil
    validationTimer?.cancel()
    validationTimer = nil
    validationPhase = nil
    let payload: [String: Any] = ["state": "idle"]
    emit(statusEvent, payload)
    return payload
  }

  @MainActor
  func startVisibleDepthValidation() throws -> [String: Any] {
    guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
      || ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) else {
      throw LidarModuleError.lidarUnsupported
    }
    _ = stopAll()
    validationSessionId = UUID().uuidString.lowercased()
    validationPhase = "calibration"
    validationPhaseStartUptime = ProcessInfo.processInfo.systemUptime
    validationLastFrameUptime = validationPhaseStartUptime
    validationPreviousTimestamp = nil
    validationPreviousPosition = nil
    validationCalibration = ValidationAccumulator(phase: "calibration", plannedDurationSeconds: 15)
    validationWallScan = ValidationAccumulator(phase: "wall_scan", plannedDurationSeconds: 30)
    arSession.delegate = self
    let configuration = ARWorldTrackingConfiguration()
    configuration.worldAlignment = .gravity
    configuration.frameSemantics = ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
      ? [.smoothedSceneDepth]
      : [.sceneDepth]
    arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    startValidationTimer()
    let payload: [String: Any] = ["state": "validating_calibration", "message": "Point at a visible textured surface and move slowly for 15 seconds."]
    emit(statusEvent, payload)
    return payload
  }

  @MainActor
  func cancelVisibleDepthValidation() -> [String: Any] {
    guard validationPhase != nil else { return ["state": "idle"] }
    finishValidation(status: "cancelled", failureReason: nil)
    return ["state": "idle", "message": "Visible-depth validation cancelled."]
  }

  private func startValidationTimer() {
    validationTimer?.cancel()
    let timer = DispatchSource.makeTimerSource(queue: processingQueue)
    timer.schedule(deadline: .now() + 0.25, repeating: 0.25)
    timer.setEventHandler { [weak self] in self?.advanceValidationTimer() }
    validationTimer = timer
    timer.resume()
  }

  func session(_ session: ARSession, didUpdate frame: ARFrame) {
    if validationPhase != nil {
      consumeValidationFrame(frame)
      return
    }
    let options = captureOptions
    if options.includeCameraPreview, frame.timestamp - lastCameraPreviewTimestamp >= 0.5 {
      lastCameraPreviewTimestamp = frame.timestamp
      emitCameraPreview(frame)
    }
    if options.includeBodyTeacher, frame.timestamp - lastBodyTeacherTimestamp >= 1.0 / options.maxBodyFramesPerSecond {
      lastBodyTeacherTimestamp = frame.timestamp
      emitBodyTeacher(frame, options: options)
    }
    let minimumInterval = 1.0 / options.maxFramesPerSecond
    guard frame.timestamp - lastDeliveredTimestamp >= minimumInterval else { return }
    guard let depthData = options.useSmoothedDepth ? frame.smoothedSceneDepth ?? frame.sceneDepth : frame.sceneDepth else { return }
    lastDeliveredTimestamp = frame.timestamp
    sequence += 1
    guard let payload = buildPointFrame(frame: frame, depthData: depthData, options: options) else { return }
    emit(pointFrameEvent, payload)
  }

  /// Emits a bounded, low-rate JPEG for the explicit Live video-overlay mode.
  /// The native module never writes the image to disk and the JS store keeps
  /// only the latest frame in memory.
  private func emitCameraPreview(_ frame: ARFrame) {
    let oriented = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
    let maxDimension = max(oriented.extent.width, oriented.extent.height)
    guard maxDimension > 0 else { return }
    let scale = min(1, 480 / maxDimension)
    let image = oriented.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    guard let cgImage = imageContext.createCGImage(image, from: image.extent),
          let jpeg = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.45) else { return }
    emit(cameraPreviewEvent, [
      "schema": "ruview.camera.preview.v1",
      "sessionId": sessionId,
      "coordinateFrameId": coordinateFrameId,
      "capturedAtUnixMs": Int64(Date().timeIntervalSince1970 * 1000),
      "width": cgImage.width,
      "height": cgImage.height,
      "jpegBase64": jpeg.base64EncodedString(),
      "rawPersisted": false,
    ])
  }

  /// Detects only coarse, visible body joints and lifts them with the depth map
  /// from the same ARFrame. Camera pixels and depth buffers remain transient.
  private func emitBodyTeacher(_ frame: ARFrame, options: LidarCaptureOptions) {
    guard trackingLabel(frame.camera.trackingState) == "normal",
          let depthData = options.useSmoothedDepth ? frame.smoothedSceneDepth ?? frame.sceneDepth : frame.sceneDepth else { return }
    let request = VNDetectHumanBodyPoseRequest()
    let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: .right, options: [:])
    do {
      try handler.perform([request])
      guard let observation = request.results?.max(by: { $0.confidence < $1.confidence }) else { return }
      let candidates: [(VNHumanBodyPoseObservation.JointName, String)] = [
        (.nose, "nose"), (.neck, "neck"),
        (.leftShoulder, "left_shoulder"), (.rightShoulder, "right_shoulder"),
        (.leftElbow, "left_elbow"), (.rightElbow, "right_elbow"),
        (.leftWrist, "left_wrist"), (.rightWrist, "right_wrist"),
        (.root, "pelvis"), (.leftHip, "left_hip"), (.rightHip, "right_hip"),
        (.leftKnee, "left_knee"), (.rightKnee, "right_knee"),
        (.leftAnkle, "left_ankle"), (.rightAnkle, "right_ankle")
      ]
      let depthMap = depthData.depthMap
      let confidenceMap = depthData.confidenceMap
      CVPixelBufferLockBaseAddress(depthMap, .readOnly)
      if let confidenceMap { CVPixelBufferLockBaseAddress(confidenceMap, .readOnly) }
      defer {
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        if let confidenceMap { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
      }
      guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return }
      let depthWidth = CVPixelBufferGetWidth(depthMap)
      let depthHeight = CVPixelBufferGetHeight(depthMap)
      let depthStride = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.size
      let depthValues = depthBase.assumingMemoryBound(to: Float32.self)
      let confidenceValues = confidenceMap.flatMap(CVPixelBufferGetBaseAddress)?.assumingMemoryBound(to: UInt8.self)
      let confidenceStride = confidenceMap.map(CVPixelBufferGetBytesPerRow) ?? 0
      let imageResolution = frame.camera.imageResolution
      let scaleX = Float(depthWidth) / Float(max(1, imageResolution.width))
      let scaleY = Float(depthHeight) / Float(max(1, imageResolution.height))
      let intrinsics = frame.camera.intrinsics
      let fx = intrinsics.columns.0.x * scaleX
      let fy = intrinsics.columns.1.y * scaleY
      let cx = intrinsics.columns.2.x * scaleX
      let cy = intrinsics.columns.2.y * scaleY
      guard fx > 0, fy > 0 else { return }
      var joints: [[String: Any]] = []
      for (jointName, outputName) in candidates {
        guard let point = try? observation.recognizedPoint(jointName), point.confidence >= 0.5 else { continue }
        // Vision is evaluated in portrait-right coordinates. Map that normalized
        // point back to the native landscape camera buffer used by ARKit depth.
        let column = min(depthWidth - 1, max(0, Int((1 - point.location.y) * CGFloat(depthWidth))))
        let row = min(depthHeight - 1, max(0, Int((1 - point.location.x) * CGFloat(depthHeight))))
        guard let depth = medianDepth(
          aroundColumn: column, row: row, values: depthValues, stride: depthStride,
          confidence: confidenceValues, confidenceStride: confidenceStride,
          width: depthWidth, height: depthHeight, minimumConfidence: options.minimumConfidence,
          maximumDepthMeters: Float(options.maximumDepthMeters)
        ) else { continue }
        let cameraPoint = SIMD4<Float>(
          (Float(column) - cx) * depth / fx,
          -(Float(row) - cy) * depth / fy,
          -depth,
          1
        )
        let worldPoint = frame.camera.transform * cameraPoint
        joints.append([
          "name": outputName,
          "positionM": [worldPoint.x, worldPoint.y, worldPoint.z],
          "confidence": Double(point.confidence),
          "depthMeters": depth,
        ])
      }
      guard joints.count >= 6 else { return }
      if wallClockOffsetMs == nil {
        wallClockOffsetMs = Date().timeIntervalSince1970 * 1000 - frame.timestamp * 1000
      }
      emit(bodyTeacherEvent, [
        "schema": "ruview.teacher.body.v1",
        "sessionId": sessionId,
        "coordinateFrameId": coordinateFrameId,
        "capturedAtUnixMs": Int64((wallClockOffsetMs ?? 0) + frame.timestamp * 1000),
        "monotonicTimestampSeconds": frame.timestamp,
        "clockModelId": "arkit-monotonic+session-wall-offset-v1",
        "trackingState": "normal",
        "source": "vision-2d+same-frame-scene-depth",
        "evidence": "MEASURED",
        "visible": true,
        "joints": joints,
        "rawCameraPersisted": false,
        "rawDepthPersisted": false,
        "biometricIdentityDerived": false,
      ])
    } catch {
      emit(lidarErrorEvent, ["code": "body_teacher_failed", "message": String(error.localizedDescription.prefix(240))])
    }
  }

  func session(_ session: ARSession, didFailWithError error: Error) {
    emit(lidarErrorEvent, ["code": "arkit_session_failed", "message": error.localizedDescription])
    emit(statusEvent, ["state": "error", "message": error.localizedDescription])
  }

  func sessionWasInterrupted(_ session: ARSession) {
    if validationPhase != nil {
      finishValidation(status: "failed", failureReason: "ARKit capture was interrupted.")
      return
    }
    emit(statusEvent, ["state": "error", "message": "ARKit capture was interrupted."])
  }

  private func consumeValidationFrame(_ frame: ARFrame) {
    guard let phase = validationPhase,
          let depthMap = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap else { return }
    let timestamp = frame.timestamp
    validationLastFrameUptime = ProcessInfo.processInfo.systemUptime
    let delta = max(0.0001, timestamp - (validationPreviousTimestamp ?? timestamp - (1.0 / 60.0)))
    let fps = min(240, 1.0 / delta)
    let position = SIMD3<Float>(frame.camera.transform.columns.3.x, frame.camera.transform.columns.3.y, frame.camera.transform.columns.3.z)
    let movement = validationPreviousPosition.map { Double(simd_distance(position, $0)) / delta } ?? 0
    let coverage = depthCoverage(depthMap)
    let tracking = trackingLabel(frame.camera.trackingState)
    let thermal = thermalLabel(ProcessInfo.processInfo.thermalState)
    if phase == "calibration" {
      validationCalibration.add(timestamp: timestamp, fps: fps, coverage: coverage, movement: movement, tracking: tracking, thermal: thermal)
    } else {
      validationWallScan.add(timestamp: timestamp, fps: fps, coverage: coverage, movement: movement, tracking: tracking, thermal: thermal)
    }
    validationPreviousTimestamp = timestamp
    validationPreviousPosition = position
    let duration = phase == "calibration" ? 15.0 : 30.0
    emit(validationMetricsEvent, [
      "phase": phase,
      "fps": fps,
      "depthCoverage": coverage,
      "trackingState": tracking,
      "movementMetersPerSecond": movement,
      "thermalState": thermal,
      "phaseSecondsRemaining": max(0, Int(ceil(duration - (ProcessInfo.processInfo.systemUptime - validationPhaseStartUptime)))),
    ])
  }

  private func advanceValidationTimer() {
    guard let phase = validationPhase else { return }
    if ProcessInfo.processInfo.systemUptime - validationLastFrameUptime > 2 {
      finishValidation(status: "failed", failureReason: "ARKit visible-depth frames stopped for more than two seconds.")
      return
    }
    let duration = phase == "calibration" ? 15.0 : 30.0
    guard ProcessInfo.processInfo.systemUptime - validationPhaseStartUptime >= duration else { return }
    let summary = phase == "calibration" ? validationCalibration.summary() : validationWallScan.summary()
    guard (summary["frameCount"] as? Int ?? 0) >= 15,
          (summary["averageDepthCoverage"] as? Double ?? 0) > 0 else {
      finishValidation(status: "failed", failureReason: "Visible depth frames were unavailable or empty.")
      return
    }
    if phase == "calibration" {
      validationPhase = "wall_scan"
      validationPhaseStartUptime = ProcessInfo.processInfo.systemUptime
      validationLastFrameUptime = validationPhaseStartUptime
      validationPreviousTimestamp = nil
      validationPreviousPosition = nil
      emit(statusEvent, ["state": "validating_wall_scan", "message": "Keep the visible wall in frame and move slowly side to side for 30 seconds."])
    } else {
      finishValidation(status: "completed", failureReason: nil)
    }
  }

  private func finishValidation(status: String, failureReason: String?) {
    arSession.pause()
    arSession.delegate = nil
    validationTimer?.cancel()
    validationTimer = nil
    validationPhase = nil
    var diagnostic: [String: Any] = [
      "schema": "ruview.ios.visible-depth-diagnostic.v1",
      "sessionId": validationSessionId,
      "createdAt": ISO8601DateFormatter().string(from: Date()),
      "deviceModelFamily": UIDevice.current.model,
      "osVersion": UIDevice.current.systemVersion,
      "appVersion": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
      "capabilities": [
        "worldTracking": ARWorldTrackingConfiguration.isSupported,
        "sceneDepth": ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
        "smoothedSceneDepth": ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth),
        "sceneMesh": ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification),
        "rawPhotonHistograms": false,
      ],
      "phases": [validationCalibration.summary(), validationWallScan.summary()],
      "consent": [
        "localValidation": true,
        "diagnosticExport": false,
        "rawSensorExport": false,
      ],
      "evidenceLabel": "direct_depth",
      "physicalNLOSStatus": "blocked_raw_transients_unavailable",
      "cameraPermission": "granted",
      "completionStatus": status,
    ]
    if let failureReason { diagnostic["failureReason"] = String(failureReason.prefix(240)) }
    emit(validationDiagnosticEvent, diagnostic)
    emit(statusEvent, [
      "state": status == "completed" ? "validation_complete" : "idle",
      "message": status == "completed" ? "Visible-depth validation complete. This is direct depth, never NLOS." : "Visible-depth validation ended.",
    ])
  }

  private func buildPointFrame(
    frame: ARFrame,
    depthData: ARDepthData,
    options: LidarCaptureOptions
  ) -> [String: Any]? {
    let depthMap = depthData.depthMap
    let confidenceMap = depthData.confidenceMap
    let width = CVPixelBufferGetWidth(depthMap)
    let height = CVPixelBufferGetHeight(depthMap)
    guard width > 0, height > 0 else { return nil }

    CVPixelBufferLockBaseAddress(depthMap, .readOnly)
    if let confidenceMap { CVPixelBufferLockBaseAddress(confidenceMap, .readOnly) }
    defer {
      CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
      if let confidenceMap { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
    }
    guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

    let depthStride = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.size
    let depthValues = depthBase.assumingMemoryBound(to: Float32.self)
    let confidenceValues = CVPixelBufferGetBaseAddress(confidenceMap ?? depthMap)?.assumingMemoryBound(to: UInt8.self)
    let confidenceStride = confidenceMap.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
    let targetSamples = max(1, min(options.maxPoints, width * height))
    let sampleStep = max(1, Int(ceil(sqrt(Double(width * height) / Double(targetSamples)))))

    let imageResolution = frame.camera.imageResolution
    let scaleX = Float(width) / Float(max(1, imageResolution.width))
    let scaleY = Float(height) / Float(max(1, imageResolution.height))
    let intrinsics = frame.camera.intrinsics
    let fx = intrinsics.columns.0.x * scaleX
    let fy = intrinsics.columns.1.y * scaleY
    let cx = intrinsics.columns.2.x * scaleX
    let cy = intrinsics.columns.2.y * scaleY
    guard fx > 0, fy > 0 else { return nil }

    var points: [Float] = []
    var confidences: [Int] = []
    points.reserveCapacity(targetSamples * 3)
    confidences.reserveCapacity(targetSamples)
    let cameraTransform = frame.camera.transform

    outer: for row in stride(from: 0, to: height, by: sampleStep) {
      for column in stride(from: 0, to: width, by: sampleStep) {
        if confidences.count >= options.maxPoints { break outer }
        let depth = depthValues[row * depthStride + column]
        guard depth.isFinite, depth >= 0.15, depth <= Float(options.maximumDepthMeters) else { continue }
        let confidence = confidenceValues.map { Int($0[row * confidenceStride + column]) } ?? 0
        guard confidence >= options.minimumConfidence else { continue }

        let cameraPoint = SIMD4<Float>(
          (Float(column) - cx) * depth / fx,
          -(Float(row) - cy) * depth / fy,
          -depth,
          1
        )
        let worldPoint = cameraTransform * cameraPoint
        points.append(contentsOf: [worldPoint.x, worldPoint.y, worldPoint.z])
        confidences.append(confidence)
      }
    }

    if wallClockOffsetMs == nil {
      wallClockOffsetMs = Date().timeIntervalSince1970 * 1000 - frame.timestamp * 1000
    }
    let capturedAtUnixMs = Int64((wallClockOffsetMs ?? 0) + frame.timestamp * 1000)
    emit(depthPacketEvent, buildDepthWirePacket(
      frame: frame,
      depthValues: depthValues,
      depthStride: depthStride,
      confidenceValues: confidenceValues,
      confidenceStride: confidenceStride,
      width: width,
      height: height,
      capturedAtUnixMs: capturedAtUnixMs
    ))
    return [
      "schema": "ruview.lidar.points.v1",
      "sessionId": sessionId,
      "coordinateFrameId": coordinateFrameId,
      "sequence": sequence,
      "capturedAtUnixMs": capturedAtUnixMs,
      "monotonicTimestampSeconds": frame.timestamp,
      "points": points,
      "confidences": confidences,
      "pointCount": confidences.count,
      "cameraTransform": flatten(frame.camera.transform),
      "cameraIntrinsics": flatten(frame.camera.intrinsics),
      "depthWidth": width,
      "depthHeight": height,
      "smoothed": options.useSmoothedDepth && frame.smoothedSceneDepth != nil,
      "trackingState": trackingLabel(frame.camera.trackingState),
      "rawDepthPersisted": false,
      "capturedImagePersisted": false,
    ]
  }

  private func buildDepthWirePacket(
    frame: ARFrame,
    depthValues: UnsafePointer<Float32>,
    depthStride: Int,
    confidenceValues: UnsafePointer<UInt8>?,
    confidenceStride: Int,
    width: Int,
    height: Int,
    capturedAtUnixMs: Int64
  ) -> [String: Any] {
    let sampleStep = 2
    let outputWidth = (width + sampleStep - 1) / sampleStep
    let outputHeight = (height + sampleStep - 1) / sampleStep
    var millimeters = Data(capacity: outputWidth * outputHeight * 2)
    var confidence = Data(capacity: outputWidth * outputHeight)
    for row in stride(from: 0, to: height, by: sampleStep) {
      for column in stride(from: 0, to: width, by: sampleStep) {
        let meters = depthValues[row * depthStride + column]
        let boundedMeters = meters.isFinite && meters > 0 ? min(meters, 65.535) : 0
        var mm = UInt16(clamping: Int((boundedMeters * 1000).rounded())).littleEndian
        withUnsafeBytes(of: &mm) { millimeters.append(contentsOf: $0) }
        confidence.append(confidenceValues.map { $0[row * confidenceStride + column] } ?? 0)
      }
    }
    let intrinsics = frame.camera.intrinsics
    return [
      "type": "ruview.lidar.depth.v1",
      "intrinsics": [
        "fx": intrinsics.columns.0.x,
        "fy": intrinsics.columns.1.y,
        "cx": intrinsics.columns.2.x,
        "cy": intrinsics.columns.2.y,
        "imageWidth": Int(frame.camera.imageResolution.width),
        "imageHeight": Int(frame.camera.imageResolution.height),
      ],
      "pose": ["matrix": flatten(frame.camera.transform)],
      "depth": [
        "width": outputWidth,
        "height": outputHeight,
        "encoding": "u16le-mm+u8-confidence",
        "millimetersBase64": millimeters.base64EncodedString(),
        "confidenceBase64": confidence.base64EncodedString(),
      ],
      "provenance": [
        "sensor": "apple-arkit-scene-depth",
        "sessionId": sessionId,
        "coordinateFrameId": coordinateFrameId,
        "source": "live",
        "privacyClass": "geometry-only",
        "sequence": sequence,
        "timestampNs": capturedAtUnixMs * 1_000_000,
        "captureTimeNs": UInt64(max(0, frame.timestamp) * 1_000_000_000),
        "clockModelId": "arkit-monotonic+session-wall-offset-v1",
        "calibrationId": "coordinate-frame:\(coordinateFrameId)",
        "trackingState": trackingLabel(frame.camera.trackingState),
        "evidence": "MEASURED",
        "schema": "ruview.lidar.depth.v1",
      ],
    ]
  }

  private func bounded(_ input: LidarCaptureOptions) -> LidarCaptureOptions {
    let output = input
    output.maxPoints = max(128, min(input.maxPoints, 4096))
    output.maxFramesPerSecond = max(1, min(input.maxFramesPerSecond, 10))
    output.minimumConfidence = max(0, min(input.minimumConfidence, 2))
    output.maximumDepthMeters = max(0.5, min(input.maximumDepthMeters, 12))
    output.maxBodyFramesPerSecond = max(5, min(input.maxBodyFramesPerSecond, 30))
    return output
  }

  private func emit(_ event: String, _ payload: [String: Any]) {
    DispatchQueue.main.async { [weak self] in self?.eventSink?(event, payload) }
  }
}

private struct ValidationAccumulator {
  let phase: String
  let plannedDurationSeconds: Int
  private var firstTimestamp: Double?
  private var lastTimestamp: Double?
  private var frameCount = 0
  private var fpsTotal = 0.0
  private var coverageTotal = 0.0
  private var movementTotal = 0.0
  private var finalTrackingState = "unavailable"
  private var peakThermalState = "unknown"

  init(phase: String, plannedDurationSeconds: Int) {
    self.phase = phase
    self.plannedDurationSeconds = plannedDurationSeconds
  }

  mutating func add(timestamp: Double, fps: Double, coverage: Double, movement: Double, tracking: String, thermal: String) {
    if firstTimestamp == nil { firstTimestamp = timestamp }
    lastTimestamp = timestamp
    frameCount += 1
    fpsTotal += max(0, min(fps.isFinite ? fps : 0, 240))
    coverageTotal += max(0, min(coverage.isFinite ? coverage : 0, 1))
    movementTotal += max(0, min(movement.isFinite ? movement : 0, 20))
    finalTrackingState = String(tracking.prefix(48))
    if thermalRank(thermal) > thermalRank(peakThermalState) { peakThermalState = thermal }
  }

  func summary() -> [String: Any] {
    let divisor = Double(max(frameCount, 1))
    return [
      "phase": phase,
      "plannedDurationSeconds": plannedDurationSeconds,
      "observedDurationSeconds": max(0, (lastTimestamp ?? 0) - (firstTimestamp ?? 0)),
      "frameCount": frameCount,
      "averageFPS": fpsTotal / divisor,
      "averageDepthCoverage": coverageTotal / divisor,
      "averageMovementMetersPerSecond": movementTotal / divisor,
      "finalTrackingState": finalTrackingState,
      "peakThermalState": peakThermalState,
    ]
  }
}

private func thermalRank(_ state: String) -> Int {
  switch state { case "nominal": return 1; case "fair": return 2; case "serious": return 3; case "critical": return 4; default: return 0 }
}

private func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
  switch state { case .nominal: return "nominal"; case .fair: return "fair"; case .serious: return "serious"; case .critical: return "critical"; @unknown default: return "unknown" }
}

private func depthCoverage(_ buffer: CVPixelBuffer) -> Double {
  CVPixelBufferLockBaseAddress(buffer, .readOnly)
  defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
  guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_DepthFloat32,
        let address = CVPixelBufferGetBaseAddress(buffer) else { return 0 }
  let width = CVPixelBufferGetWidth(buffer)
  let height = CVPixelBufferGetHeight(buffer)
  let rowStride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float32>.stride
  let values = address.assumingMemoryBound(to: Float32.self)
  var valid = 0
  var measured = 0
  for row in stride(from: 0, to: height, by: 8) {
    for column in stride(from: 0, to: width, by: 8) {
      let value = values[row * rowStride + column]
      if value.isFinite && value > 0 { valid += 1 }
      measured += 1
    }
  }
  return measured == 0 ? 0 : Double(valid) / Double(measured)
}

private func medianDepth(
  aroundColumn column: Int,
  row: Int,
  values: UnsafePointer<Float32>,
  stride: Int,
  confidence: UnsafePointer<UInt8>?,
  confidenceStride: Int,
  width: Int,
  height: Int,
  minimumConfidence: Int,
  maximumDepthMeters: Float
) -> Float? {
  var samples: [Float] = []
  for sampleRow in max(0, row - 2)...min(height - 1, row + 2) {
    for sampleColumn in max(0, column - 2)...min(width - 1, column + 2) {
      if let confidence, Int(confidence[sampleRow * confidenceStride + sampleColumn]) < minimumConfidence { continue }
      let depth = values[sampleRow * stride + sampleColumn]
      if depth.isFinite, depth >= 0.15, depth <= maximumDepthMeters { samples.append(depth) }
    }
  }
  guard !samples.isEmpty else { return nil }
  samples.sort()
  return samples[samples.count / 2]
}

@available(iOS 16.0, *)
private final class RuViewRoomCaptureController: NSObject, RoomCaptureSessionDelegate {
  private let session: RoomCaptureSession
  private let roomBuilder = RoomBuilder(options: [.beautifyObjects])
  private let coordinateFrameId: String
  private let eventSink: (String, [String: Any]) -> Void

  init(arSession: ARSession, coordinateFrameId: String, eventSink: @escaping (String, [String: Any]) -> Void) {
    if #available(iOS 17.0, *) { session = RoomCaptureSession(arSession: arSession) }
    else { session = RoomCaptureSession() }
    self.coordinateFrameId = coordinateFrameId
    self.eventSink = eventSink
    super.init()
    session.delegate = self
  }

  func start() {
    var configuration = RoomCaptureSession.Configuration()
    configuration.isCoachingEnabled = true
    session.run(configuration: configuration)
  }

  func stop() {
    if #available(iOS 17.0, *) { session.stop(pauseARSession: false) }
    else { session.stop() }
  }

  func currentPosePayload() -> [String: Any]? {
    guard let frame = session.arSession.currentFrame else { return nil }
    return spatialPosePayload(
      transform: frame.camera.transform,
      coordinateFrameId: coordinateFrameId,
      trackingState: trackingLabel(frame.camera.trackingState)
    )
  }

  func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
    eventSink(roomUpdateEvent, roomSummary(room, state: "capturing_room"))
  }

  func captureSession(_ session: RoomCaptureSession, didProvide instruction: RoomCaptureSession.Instruction) {
    eventSink(statusEvent, ["state": "capturing_room", "instruction": instructionLabel(instruction)])
  }

  func captureSession(_ session: RoomCaptureSession, didEndWith data: CapturedRoomData, error: Error?) {
    if let error {
      eventSink(lidarErrorEvent, ["code": "room_capture_failed", "message": error.localizedDescription])
      eventSink(statusEvent, ["state": "error", "message": error.localizedDescription])
      return
    }
    eventSink(statusEvent, ["state": "processing_room"])
    Task { [roomBuilder, coordinateFrameId, eventSink] in
      do {
        let room = try await roomBuilder.capturedRoom(from: data)
        var payload = roomPayload(room, coordinateFrameId: coordinateFrameId)
        payload["capturedAtUnixMs"] = Int64(Date().timeIntervalSince1970 * 1000)
        eventSink(roomCompleteEvent, payload)
        eventSink(statusEvent, ["state": "idle", "message": "Room geometry is ready for calibration."])
      } catch {
        eventSink(lidarErrorEvent, ["code": "room_processing_failed", "message": error.localizedDescription])
        eventSink(statusEvent, ["state": "error", "message": error.localizedDescription])
      }
    }
  }

  private func roomSummary(_ room: CapturedRoom, state: String) -> [String: Any] {
    [
      "state": state,
      "roomId": room.identifier.uuidString.lowercased(),
      "surfaceCount": room.walls.count + room.doors.count + room.windows.count + room.openings.count,
      "objectCount": room.objects.count,
    ]
  }
}

@available(iOS 16.0, *)
private func roomPayload(_ room: CapturedRoom, coordinateFrameId: String) -> [String: Any] {
  var surfaces: [[String: Any]] = []
  surfaces.append(contentsOf: room.walls.map { surfacePayload($0, kind: "wall") })
  surfaces.append(contentsOf: room.doors.map { surfacePayload($0, kind: "door") })
  surfaces.append(contentsOf: room.windows.map { surfacePayload($0, kind: "window") })
  surfaces.append(contentsOf: room.openings.map { surfacePayload($0, kind: "opening") })
  if #available(iOS 17.0, *) {
    surfaces.append(contentsOf: room.floors.map { surfacePayload($0, kind: "floor") })
  }
  let objects = room.objects.map { object in
    [
      "id": object.identifier.uuidString.lowercased(),
      "category": String(describing: object.category),
      "confidence": String(describing: object.confidence),
      "dimensionsM": [object.dimensions.x, object.dimensions.y, object.dimensions.z],
      "transform": flatten(object.transform),
    ] as [String: Any]
  }
  return [
    "schema": "ruview.roomplan.geometry.v1",
    "roomId": room.identifier.uuidString.lowercased(),
    "coordinateFrameId": coordinateFrameId,
    "surfaces": surfaces,
    "objects": objects,
    "surfaceCount": surfaces.count,
    "objectCount": objects.count,
    "rawCameraPersisted": false,
    "rawDepthPersisted": false,
  ]
}

@available(iOS 16.0, *)
private func surfacePayload(_ surface: CapturedRoom.Surface, kind: String) -> [String: Any] {
  [
    "id": surface.identifier.uuidString.lowercased(),
    "kind": kind,
    "category": String(describing: surface.category),
    "confidence": String(describing: surface.confidence),
    "dimensionsM": [surface.dimensions.x, surface.dimensions.y, surface.dimensions.z],
    "transform": flatten(surface.transform),
  ]
}

@available(iOS 16.0, *)
private func instructionLabel(_ instruction: RoomCaptureSession.Instruction) -> String {
  switch instruction {
  case .moveCloseToWall: return "Move closer to the wall."
  case .moveAwayFromWall: return "Move away from the wall."
  case .slowDown: return "Move the iPhone more slowly."
  case .turnOnLight: return "Turn on more lights."
  case .normal: return "Continue scanning the room."
  case .lowTexture: return "Aim at an area with more visual detail."
  @unknown default: return "Continue scanning the room."
  }
}

private func trackingLabel(_ state: ARCamera.TrackingState) -> String {
  switch state {
  case .normal: return "normal"
  case .notAvailable: return "unavailable"
  case .limited(let reason): return "limited_\(String(describing: reason))"
  }
}

private func spatialPosePayload(
  transform: simd_float4x4,
  coordinateFrameId: String,
  trackingState: String
) -> [String: Any] {
  [
    "coordinateFrameId": coordinateFrameId,
    "capturedAtUnixMs": Int64(Date().timeIntervalSince1970 * 1000),
    "positionM": [transform.columns.3.x, transform.columns.3.y, transform.columns.3.z],
    "transform": flatten(transform),
    "trackingState": trackingState,
  ]
}

private func flatten(_ matrix: simd_float4x4) -> [Float] {
  [
    matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z, matrix.columns.0.w,
    matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z, matrix.columns.1.w,
    matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z, matrix.columns.2.w,
    matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z, matrix.columns.3.w,
  ]
}

private func flatten(_ matrix: simd_float3x3) -> [Float] {
  [
    matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z,
    matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z,
    matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z,
  ]
}
