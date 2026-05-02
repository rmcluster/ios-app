import Foundation
import Combine

/// Centralized persistence for all app settings.
final class RpcSettings: ObservableObject {
    static let shared = RpcSettings()

    enum Keys {
        static let host = "rpcHost"
        static let port = "rpcPort"
        static let storagePort = "rpcStoragePort"
        static let discoveryIp = "rpcDiscoveryIp"
        static let discoveryPort = "rpcDiscoveryPort"
        static let threads = "rpcThreads"
        static let deviceId = "rpcDeviceId"
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    @Published var host: String {
        didSet { UserDefaults.standard.set(host, forKey: Keys.host) }
    }
    @Published var port: Int {
        didSet { UserDefaults.standard.set(port, forKey: Keys.port) }
    }
    @Published var storagePort: Int {
        didSet { UserDefaults.standard.set(storagePort, forKey: Keys.storagePort) }
    }
    @Published var discoveryIp: String {
        didSet { UserDefaults.standard.set(discoveryIp, forKey: Keys.discoveryIp) }
    }
    @Published var discoveryPort: Int {
        didSet { UserDefaults.standard.set(discoveryPort, forKey: Keys.discoveryPort) }
    }
    @Published var threads: Int {
        didSet { UserDefaults.standard.set(threads, forKey: Keys.threads) }
    }
    @Published var deviceId: String {
        didSet { UserDefaults.standard.set(deviceId, forKey: Keys.deviceId) }
    }

    private init() {
        self.host = UserDefaults.standard.string(forKey: Keys.host) ?? "0.0.0.0"
        
        let p = UserDefaults.standard.integer(forKey: Keys.port)
        self.port = (p == 0) ? 47651 : p
        
        let sp = UserDefaults.standard.integer(forKey: Keys.storagePort)
        self.storagePort = (sp == 0) ? 47672 : sp
        
        self.discoveryIp = UserDefaults.standard.string(forKey: Keys.discoveryIp) ?? ""
        
        let dp = UserDefaults.standard.integer(forKey: Keys.discoveryPort)
        self.discoveryPort = (dp == 0) ? 50055 : dp
        
        let t = UserDefaults.standard.integer(forKey: Keys.threads)
        self.threads = (t == 0) ? 4 : t

        if let existing = UserDefaults.standard.string(forKey: Keys.deviceId), !existing.isEmpty {
            self.deviceId = existing
        } else {
            let newId = UUID().uuidString
            UserDefaults.standard.set(newId, forKey: Keys.deviceId)
            self.deviceId = newId
        }
    }

    // ── Storage ──────────────────────────────────────────────────────────────
    var storageDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("StorageApp", isDirectory: true)
    }
}
