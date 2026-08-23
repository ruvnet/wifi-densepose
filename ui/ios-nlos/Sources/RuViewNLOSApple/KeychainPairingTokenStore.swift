import Foundation
import RuViewNLOSCore

#if canImport(Security)
import Security

public enum PairingTokenStoreError: Error, LocalizedError, Sendable {
    case keychainFailure(OSStatus)
    case invalidStoredValue

    public var errorDescription: String? {
        switch self {
        case .keychainFailure:
            return "The pairing token could not be accessed in Keychain."
        case .invalidStoredValue:
            return "The stored pairing token is invalid."
        }
    }
}

public final class KeychainPairingTokenStore: @unchecked Sendable {
    private let service: String

    public init(service: String = "org.ruvnet.RuViewNLOS.pairing") {
        self.service = service
    }

    public func save(_ token: String, for endpoint: URL) throws {
        try WSSConnectionValidator.validatePairingToken(token)
        let account = try WSSConnectionValidator.credentialAccount(for: endpoint)
        guard let data = token.data(using: .utf8) else {
            throw PairingTokenStoreError.invalidStoredValue
        }

        let identity: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]

        let updateStatus = SecItemUpdate(identity as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecSuccess { return }
        guard updateStatus == errSecItemNotFound else {
            throw PairingTokenStoreError.keychainFailure(updateStatus)
        }

        var insert = identity
        attributes.forEach { insert[$0.key] = $0.value }
        let insertStatus = SecItemAdd(insert as CFDictionary, nil)
        guard insertStatus == errSecSuccess else {
            throw PairingTokenStoreError.keychainFailure(insertStatus)
        }
    }

    public func load(for endpoint: URL) throws -> String? {
        let account = try WSSConnectionValidator.credentialAccount(for: endpoint)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = item as? Data,
              let token = String(data: data, encoding: .utf8) else {
            if status == errSecSuccess {
                throw PairingTokenStoreError.invalidStoredValue
            }
            throw PairingTokenStoreError.keychainFailure(status)
        }
        do {
            try WSSConnectionValidator.validatePairingToken(token)
        } catch {
            throw PairingTokenStoreError.invalidStoredValue
        }
        return token
    }

    public func deleteAll() throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw PairingTokenStoreError.keychainFailure(status)
        }
    }
}

#else

public enum PairingTokenStoreError: Error, LocalizedError, Sendable {
    case unavailable

    public var errorDescription: String? {
        "Apple Keychain is unavailable on this platform."
    }
}

public final class KeychainPairingTokenStore: @unchecked Sendable {
    public init(service: String = "org.ruvnet.RuViewNLOS.pairing") {}

    public func save(_ token: String, for endpoint: URL) throws {
        throw PairingTokenStoreError.unavailable
    }
    public func load(for endpoint: URL) throws -> String? {
        throw PairingTokenStoreError.unavailable
    }
    public func deleteAll() throws { throw PairingTokenStoreError.unavailable }
}

#endif
