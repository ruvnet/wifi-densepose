import Foundation

actor WebSocketStreamer {
    enum StreamError: Error {
        case invalidURL
    }

    struct WirePacket: Codable {
        struct Depth: Codable {
            let width: Int
            let height: Int
            let encoding: String
            let millimetersBase64: String
            let confidenceBase64: String
        }

        let type: String
        let intrinsics: RuViewLiDARFrame.Intrinsics
        let pose: RuViewLiDARFrame.Pose
        let depth: Depth
        let provenance: RuViewLiDARFrame.Provenance
    }

    private var task: URLSessionWebSocketTask?
    private let encoder = JSONEncoder()
    private var lastSentNs: UInt64 = 0

    func connect(to endpoint: String) throws {
        guard let url = URL(string: endpoint),
              url.scheme == "ws" || url.scheme == "wss" else {
            throw StreamError.invalidURL
        }
        task?.cancel(with: .goingAway, reason: nil)
        let socket = URLSession.shared.webSocketTask(with: url)
        socket.resume()
        task = socket
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    func send(_ frame: RuViewLiDARFrame, maxFPS: UInt64 = 15, sampleStep: Int = 2) async throws {
        guard let task else { return }

        let timestamp = frame.provenance.timestampNs
        let minDelta = 1_000_000_000 / max(1, maxFPS)
        guard timestamp >= lastSentNs + minDelta else { return }
        lastSentNs = timestamp

        let packet = Self.makeWirePacket(frame, sampleStep: max(1, sampleStep))
        let data = try encoder.encode(packet)
        guard let string = String(data: data, encoding: .utf8) else { return }
        try await task.send(.string(string))
    }

    static func makeWirePacket(_ frame: RuViewLiDARFrame, sampleStep: Int) -> WirePacket {
        let step = max(1, sampleStep)
        let sourceWidth = frame.depth.width
        let sourceHeight = frame.depth.height
        let width = (sourceWidth + step - 1) / step
        let height = (sourceHeight + step - 1) / step

        var millimeters = Data(capacity: width * height * 2)
        var confidence = Data(capacity: width * height)

        for y in stride(from: 0, to: sourceHeight, by: step) {
            for x in stride(from: 0, to: sourceWidth, by: step) {
                let index = y * sourceWidth + x
                let meters = frame.depth.meters[index]
                let mm = UInt16(clamping: Int((meters * 1000).rounded()))
                var littleEndian = mm.littleEndian
                withUnsafeBytes(of: &littleEndian) { millimeters.append(contentsOf: $0) }
                confidence.append(frame.depth.confidence[index])
            }
        }

        return WirePacket(
            type: frame.type,
            intrinsics: frame.intrinsics,
            pose: frame.pose,
            depth: WirePacket.Depth(
                width: width,
                height: height,
                encoding: "u16le-mm+u8-confidence",
                millimetersBase64: millimeters.base64EncodedString(),
                confidenceBase64: confidence.base64EncodedString()
            ),
            provenance: frame.provenance
        )
    }
}
