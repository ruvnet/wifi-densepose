import Combine
import Foundation
import RuViewNLOSApple
import RuViewNLOSCore

@MainActor
final class AppModel: ObservableObject {
    enum ConnectionState: Equatable {
        case disconnected
        case connecting
        case connected
        case blocked
    }

    @Published var endpointText = "" {
        didSet {
            if endpointText != oldValue {
                storedTokenAvailable = false
            }
        }
    }
    @Published var pairingToken = ""
    @Published private(set) var storedTokenAvailable = false
    @Published private(set) var transportActive = false
    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var statusMessage = "Disconnected; no track evidence is displayed."
    @Published private(set) var frame: TrackDisplayFrame?

    let capabilities = AppleCapabilityProbe.probe()

    private let client = NLOSWebSocketClient()
    private let tokenStore = KeychainPairingTokenStore()

    init() {
        client.onEvent = { [weak self] event in
            self?.handle(event)
        }
    }

    var tracks: [NLOSTrack] { frame?.tracks ?? [] }
    var isConnected: Bool { transportActive }

    func connect() {
        if transportActive {
            client.disconnect()
        }
        let trimmedEndpoint = endpointText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let endpoint = URL(string: trimmedEndpoint) else {
            block("Enter a valid wss endpoint.")
            return
        }

        do {
            let enteredToken = pairingToken
            let token: String
            if enteredToken.isEmpty {
                guard let stored = try tokenStore.load(for: endpoint) else {
                    storedTokenAvailable = false
                    block("Enter a pairing token. None is stored for this server authority.")
                    return
                }
                storedTokenAvailable = true
                token = stored
            } else {
                token = enteredToken
            }

            try WSSConnectionValidator.validate(endpoint: endpoint, pairingToken: token)
            if !enteredToken.isEmpty {
                try tokenStore.save(enteredToken, for: endpoint)
                storedTokenAvailable = true
            }
            try client.connect(endpoint: endpoint, pairingToken: token)
            pairingToken = ""
        } catch let error as LocalizedError {
            block(error.errorDescription ?? "Connection setup failed closed.")
        } catch {
            block("Connection setup failed closed.")
        }
    }

    func disconnect() {
        client.disconnect()
    }

    func suspendForPrivacy() {
        guard isConnected || frame != nil else { return }
        client.disconnect()
    }

    func forgetPairingToken() {
        do {
            if transportActive {
                client.disconnect()
            }
            try tokenStore.deleteAll()
            pairingToken = ""
            storedTokenAvailable = false
            statusMessage = "Pairing token removed from Keychain."
        } catch {
            block("Keychain token could not be removed.")
        }
    }

    private func handle(_ event: NLOSStreamEvent) {
        switch event {
        case .connecting:
            transportActive = true
            connectionState = .connecting
            clearFrame()
            statusMessage = "Opening authenticated secure stream…"
        case .connected:
            transportActive = true
            connectionState = .connected
            statusMessage = "Secure stream connected; waiting for validated evidence."
        case let .frame(displayFrame):
            transportActive = true
            connectionState = .connected
            frame = displayFrame
            if displayFrame.tracks.isEmpty {
                statusMessage = "Frame accepted, but no displayable tracks were present. Unknown tracks remain hidden."
            } else {
                statusMessage = "Validated frame \(displayFrame.sequence) with \(displayFrame.tracks.count) displayable track(s)."
            }
        case let .failClosed(reason):
            connectionState = .blocked
            clearFrame()
            statusMessage = reason
        case let .disconnected(reason):
            transportActive = false
            connectionState = .disconnected
            clearFrame()
            statusMessage = reason
        }
    }

    private func block(_ reason: String) {
        transportActive = false
        connectionState = .blocked
        clearFrame()
        statusMessage = reason
    }

    private func clearFrame() {
        frame = nil
    }
}
