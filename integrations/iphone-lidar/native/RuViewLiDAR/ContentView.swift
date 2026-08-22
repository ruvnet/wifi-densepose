import SwiftUI

struct ContentView: View {
    @StateObject private var capture = LiDARCaptureManager()
    @State private var endpoint = "ws://HOST:8787/ws/lidar?token=TOKEN"
    @State private var streaming = false
    @State private var status = "Idle"

    private let streamer = WebSocketStreamer()

    var body: some View {
        NavigationStack {
            Form {
                Section("Sensor") {
                    HStack {
                        Text("State")
                        Spacer()
                        Text(stateText)
                            .foregroundStyle(stateColor)
                    }
                    HStack {
                        Text("Capture FPS")
                        Spacer()
                        Text(capture.framesPerSecond.formatted(.number.precision(.fractionLength(1))))
                    }
                    if let frame = capture.lastFrame {
                        HStack {
                            Text("Depth")
                            Spacer()
                            Text("\(frame.depth.width) x \(frame.depth.height)")
                        }
                        HStack {
                            Text("Sequence")
                            Spacer()
                            Text("\(frame.provenance.sequence)")
                        }
                    }

                    Button("Start LiDAR") {
                        capture.start(smoothed: false)
                    }
                    .disabled(capture.state == .running)

                    Button("Stop") {
                        capture.stop()
                    }
                    .disabled(capture.state != .running)
                }

                Section("RuView Stream") {
                    TextField("ws://host:port/ws/lidar", text: $endpoint)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Toggle("Stream geometry", isOn: $streaming)
                        .onChange(of: streaming) { _, enabled in
                            Task {
                                if enabled {
                                    do {
                                        try await streamer.connect(to: endpoint)
                                        status = "Connected"
                                    } catch {
                                        streaming = false
                                        status = error.localizedDescription
                                    }
                                } else {
                                    await streamer.disconnect()
                                    status = "Disconnected"
                                }
                            }
                        }

                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Privacy") {
                    Text("This implementation transmits depth geometry, confidence, camera intrinsics, and device pose. RGB camera frames are not transmitted.")
                        .font(.footnote)
                }
            }
            .navigationTitle("RuView LiDAR")
            .onAppear {
                capture.onFrame = { frame in
                    guard streaming else { return }
                    Task {
                        do {
                            try await streamer.send(frame, maxFPS: 15, sampleStep: 2)
                        } catch {
                            await MainActor.run {
                                status = error.localizedDescription
                            }
                        }
                    }
                }
            }
            .onDisappear {
                capture.stop()
                Task { await streamer.disconnect() }
            }
        }
    }

    private var stateText: String {
        switch capture.state {
        case .idle: return "Idle"
        case .unsupported: return "No LiDAR"
        case .running: return "Live"
        case .failed(let message): return "Error: \(message)"
        }
    }

    private var stateColor: Color {
        switch capture.state {
        case .running: return .green
        case .failed, .unsupported: return .red
        case .idle: return .secondary
        }
    }
}
