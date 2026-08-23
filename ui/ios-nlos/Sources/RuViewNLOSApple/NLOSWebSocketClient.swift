import Foundation
import RuViewNLOSCore

public enum NLOSStreamEvent: Sendable {
    case connecting
    case connected
    case frame(TrackDisplayFrame)
    case failClosed(String)
    case disconnected(String)
}

#if canImport(Darwin)

private final class RejectRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

private enum ProcessedMessage: Sendable {
    case authenticated(NLOSAuthenticatedSession)
    case frame(TrackDisplayFrame)
}

private actor FrameProcessor {
    private let decoder = TrackEnvelopeDecoder()
    private var streamGuard = TrackStreamGuard()
    private var authenticatedSession: NLOSAuthenticatedSession?

    func process(_ data: Data, nowUnixMs: UInt64) throws -> ProcessedMessage {
        guard let authenticatedSession else {
            let authenticated = try decoder.decodeAuthenticated(
                data,
                nowUnixMs: nowUnixMs
            )
            self.authenticatedSession = authenticated
            return .authenticated(authenticated)
        }
        guard authenticatedSession.expiresAtUnixMs > nowUnixMs else {
            throw NLOSValidationError.staleFrame
        }
        let envelope = try decoder.decode(data, nowUnixMs: nowUnixMs)
        guard envelope.value.sessionId == authenticatedSession.sessionId else {
            throw NLOSValidationError.sessionChanged
        }
        return .frame(try streamGuard.accept(envelope, nowUnixMs: nowUnixMs))
    }
}

@MainActor
public final class NLOSWebSocketClient {
    public var onEvent: ((NLOSStreamEvent) -> Void)?

    private var session: URLSession?
    private var redirectDelegate: RejectRedirectDelegate?
    private var socket: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var expiryTask: Task<Void, Never>?
    private var authenticationTask: Task<Void, Never>?
    private var sessionExpiryTask: Task<Void, Never>?
    private var connectionId: UUID?

    public init() {}

    deinit {
        receiveTask?.cancel()
        expiryTask?.cancel()
        authenticationTask?.cancel()
        sessionExpiryTask?.cancel()
        socket?.cancel(with: .goingAway, reason: nil)
        session?.invalidateAndCancel()
    }

    public func connect(endpoint: URL, pairingToken: String) throws {
        try WSSConnectionValidator.validate(endpoint: endpoint, pairingToken: pairingToken)
        disconnect(emitEvent: false)

        let currentConnectionId = UUID()
        let processor = FrameProcessor()
        connectionId = currentConnectionId

        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 15
        configuration.timeoutIntervalForResource = 86_400

        let redirectDelegate = RejectRedirectDelegate()
        let session = URLSession(
            configuration: configuration,
            delegate: redirectDelegate,
            delegateQueue: nil
        )
        var request = URLRequest(url: endpoint)
        request.timeoutInterval = 15
        request.setValue("Bearer \(pairingToken)", forHTTPHeaderField: "Authorization")
        request.setValue(TrackEnvelopeDecoder.schema, forHTTPHeaderField: "Sec-WebSocket-Protocol")

        let socket = session.webSocketTask(with: request)
        socket.maximumMessageSize = TrackEnvelopeDecoder.maximumFrameBytes
        self.session = session
        self.redirectDelegate = redirectDelegate
        self.socket = socket
        onEvent?(.connecting)
        socket.resume()

        authenticationTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: 5_000_000_000)
            } catch {
                return
            }
            guard let self, self.connectionId == currentConnectionId else { return }
            self.failClosed("Authentication acknowledgement timed out; all tracks were hidden.")
            self.disconnect(emitEvent: true)
        }

        receiveTask = Task { [weak self, weak socket] in
            guard let socket else { return }
            while !Task.isCancelled {
                do {
                    let message = try await socket.receive()
                    guard let self, self.connectionId == currentConnectionId else { return }
                    await self.handle(
                        message,
                        connectionId: currentConnectionId,
                        processor: processor
                    )
                } catch {
                    guard !Task.isCancelled, let self,
                          self.connectionId == currentConnectionId else { return }
                    self.failClosed("Secure stream ended; all tracks were hidden.")
                    self.disconnect(emitEvent: true)
                    return
                }
            }
        }
    }

    public func disconnect() {
        disconnect(emitEvent: true)
    }

    private func disconnect(emitEvent: Bool) {
        connectionId = nil
        receiveTask?.cancel()
        receiveTask = nil
        expiryTask?.cancel()
        expiryTask = nil
        authenticationTask?.cancel()
        authenticationTask = nil
        sessionExpiryTask?.cancel()
        sessionExpiryTask = nil
        socket?.cancel(with: .normalClosure, reason: nil)
        socket = nil
        session?.invalidateAndCancel()
        session = nil
        redirectDelegate = nil
        if emitEvent {
            onEvent?(.disconnected("Disconnected; all tracks are hidden."))
        }
    }

    private func handle(
        _ message: URLSessionWebSocketTask.Message,
        connectionId: UUID,
        processor: FrameProcessor
    ) async {
        let data: Data
        switch message {
        case let .data(binary):
            data = binary
        case let .string(text):
            guard let encoded = text.data(using: .utf8) else {
                failClosed("A non UTF-8 frame was rejected.")
                return
            }
            data = encoded
        @unknown default:
            failClosed("An unsupported WebSocket frame was rejected.")
            return
        }

        let nowUnixMs = Self.nowUnixMs()
        do {
            let processed = try await processor.process(data, nowUnixMs: nowUnixMs)
            guard self.connectionId == connectionId else { return }
            guard case let .frame(displayFrame) = processed else {
                guard case let .authenticated(session) = processed else { return }
                authenticationTask?.cancel()
                authenticationTask = nil
                scheduleSessionExpiry(session, connectionId: connectionId)
                onEvent?(.connected)
                return
            }
            guard displayFrame.expiresAtUnixMs > Self.nowUnixMs() else {
                throw NLOSValidationError.staleFrame
            }
            onEvent?(.frame(displayFrame))
            scheduleExpiry(for: displayFrame, connectionId: connectionId)
        } catch let validationError as NLOSValidationError {
            failClosed(validationError.localizedDescription)
        } catch {
            failClosed("Frame validation failed; all tracks were hidden.")
        }
    }

    private func scheduleExpiry(for frame: TrackDisplayFrame, connectionId: UUID) {
        expiryTask?.cancel()
        let now = Self.nowUnixMs()
        let delayMs = frame.expiresAtUnixMs > now ? frame.expiresAtUnixMs - now : 0
        let sequence = frame.sequence
        let sessionId = frame.sessionId

        expiryTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: delayMs * 1_000_000)
            } catch {
                return
            }
            guard let self, self.connectionId == connectionId else { return }
            self.onEvent?(.failClosed(
                "Frame \(sessionId)#\(sequence) expired; all tracks were hidden."
            ))
        }
    }

    private func scheduleSessionExpiry(
        _ session: NLOSAuthenticatedSession,
        connectionId: UUID
    ) {
        sessionExpiryTask?.cancel()
        let now = Self.nowUnixMs()
        let delayMs = session.expiresAtUnixMs > now ? session.expiresAtUnixMs - now : 0
        sessionExpiryTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: delayMs * 1_000_000)
            } catch {
                return
            }
            guard let self, self.connectionId == connectionId else { return }
            self.failClosed("Authenticated session expired; all tracks were hidden.")
            self.disconnect(emitEvent: true)
        }
    }

    private func failClosed(_ reason: String) {
        expiryTask?.cancel()
        expiryTask = nil
        onEvent?(.failClosed(reason))
    }

    private static func nowUnixMs() -> UInt64 {
        UInt64(Date().timeIntervalSince1970 * 1_000)
    }
}

#else

@MainActor
public final class NLOSWebSocketClient {
    public var onEvent: ((NLOSStreamEvent) -> Void)?

    public init() {}

    public func connect(endpoint: URL, pairingToken: String) throws {
        try WSSConnectionValidator.validate(endpoint: endpoint, pairingToken: pairingToken)
        onEvent?(.failClosed("Apple URLSession WebSocket support is unavailable on this platform."))
    }

    public func disconnect() {
        onEvent?(.disconnected("Disconnected; all tracks are hidden."))
    }
}

#endif
