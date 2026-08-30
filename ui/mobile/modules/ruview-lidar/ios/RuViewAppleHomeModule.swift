import ExpoModulesCore
import Foundation

private let appleHomeDiscoveryEvent = "onAppleHomeDiscovery"

public final class RuViewAppleHomeModule: Module {
  private let discovery = AppleHomeDiscoveryController()

  public func definition() -> ModuleDefinition {
    Name("RuViewAppleHome")
    Events(appleHomeDiscoveryEvent)

    OnCreate { [weak self] in
      self?.discovery.eventSink = { [weak self] payload in
        self?.sendEvent(appleHomeDiscoveryEvent, payload)
      }
    }

    OnDestroy { [weak self] in
      self?.discovery.stop()
      self?.discovery.eventSink = nil
    }

    AsyncFunction("startDiscovery") { [weak self] () -> [String: Any] in
      self?.discovery.start() ?? ["state": "unavailable", "bridges": []]
    }

    AsyncFunction("stopDiscovery") { [weak self] () -> [String: Any] in
      self?.discovery.stop() ?? ["state": "idle", "bridges": []]
    }

    AsyncFunction("getDiscoveredBridges") { [weak self] () -> [[String: Any]] in
      self?.discovery.bridgePayloads() ?? []
    }
  }
}

private final class AppleHomeDiscoveryController: NSObject, NetServiceBrowserDelegate, NetServiceDelegate {
  var eventSink: (([String: Any]) -> Void)?
  private var browser: NetServiceBrowser?
  private var services: [String: NetService] = [:]
  private var state = "idle"

  func start() -> [String: Any] {
    stop()
    state = "searching"
    let browser = NetServiceBrowser()
    browser.delegate = self
    self.browser = browser
    browser.searchForServices(ofType: "_hap._tcp.", inDomain: "local.")
    emit()
    return payload()
  }

  @discardableResult
  func stop() -> [String: Any] {
    browser?.stop()
    browser = nil
    services.values.forEach { $0.stop() }
    services.removeAll()
    state = "idle"
    emit()
    return payload()
  }

  func bridgePayloads() -> [[String: Any]] {
    services.values.sorted { $0.name < $1.name }.map(servicePayload)
  }

  func netServiceBrowserWillSearch(_ browser: NetServiceBrowser) {
    state = "searching"
    emit()
  }

  func netServiceBrowserDidStopSearch(_ browser: NetServiceBrowser) {
    if state == "searching" { state = "idle" }
    emit()
  }

  func netServiceBrowser(_ browser: NetServiceBrowser, didNotSearch errorDict: [String: NSNumber]) {
    state = "error"
    let errorCode = errorDict["NSNetServicesErrorCode"]?.intValue ?? 0
    emit(error: "Bonjour search failed (\(errorCode)). Check Local Network permission.")
  }

  func netServiceBrowser(_ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
    let key = "\(service.domain)|\(service.type)|\(service.name)"
    services[key] = service
    service.delegate = self
    service.resolve(withTimeout: 5)
    if !moreComing { emit() }
  }

  func netServiceBrowser(_ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool) {
    let key = "\(service.domain)|\(service.type)|\(service.name)"
    services.removeValue(forKey: key)
    if !moreComing { emit() }
  }

  func netServiceDidResolveAddress(_ sender: NetService) { emit() }
  func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) { emit() }

  private func servicePayload(_ service: NetService) -> [String: Any] {
    let txt = service.txtRecordData().map(NetService.dictionary(fromTXTRecord:)) ?? [:]
    let model = txt["md"].flatMap { String(data: $0, encoding: .utf8) }
    let category = txt["ci"].flatMap { String(data: $0, encoding: .utf8) }
    let paired = txt["sf"].flatMap { String(data: $0, encoding: .utf8) }.map { $0 == "0" }
    var payload: [String: Any] = [
      "id": "\(service.domain)|\(service.type)|\(service.name)",
      "name": service.name,
      "port": service.port,
      "serviceType": service.type,
      "domain": service.domain,
      "source": "bonjour_hap",
    ]
    if let hostName = service.hostName { payload["hostName"] = hostName }
    if let model { payload["model"] = model }
    if let category { payload["categoryIdentifier"] = category }
    if let paired { payload["paired"] = paired }
    return payload
  }

  private func payload(error: String? = nil) -> [String: Any] {
    var result: [String: Any] = [
      "state": state,
      "bridges": bridgePayloads(),
      "capturedAtUnixMs": Int64(Date().timeIntervalSince1970 * 1000),
      "source": "bonjour_hap",
    ]
    if let error { result["error"] = error }
    return result
  }

  private func emit(error: String? = nil) {
    let value = payload(error: error)
    DispatchQueue.main.async { [weak self] in self?.eventSink?(value) }
  }
}
