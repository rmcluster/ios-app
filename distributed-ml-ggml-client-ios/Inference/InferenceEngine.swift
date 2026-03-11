// InferenceEngine.swift
//
// High-level Swift wrapper around LlamaBridge.
// Manages model lifecycle and exposes an async streaming generation API.
//
// Thread model
// ────────────
// All @Observable state mutations happen on MainActor.
// Blocking llama.cpp work (model load, inference) runs in Task.detached so it
// doesn't block the main thread.  `bridge` is marked nonisolated(unsafe) to
// allow capture from detached tasks; the caller guarantees single-threaded
// access to the bridge (only one operation at a time).

import Foundation
import Combine
import Network
import Darwin
#if canImport(UIKit)
import UIKit
#endif

// ── Chat message ──────────────────────────────────────────────────────────────

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: String    // "user" | "assistant"
    var content: String
}

// ── Generation result ─────────────────────────────────────────────────────────

struct GenerationResult: Sendable {
    let text: String
    let isDone: Bool
    let tokensPerSecond: Double
}

// ── RPC server state ──────────────────────────────────────────────────────────

enum RPCServerState: Equatable {
    case idle
    case starting
    case running(endpoint: String)
    case unavailable(String)
}

// ── Model state ───────────────────────────────────────────────────────────────

enum ModelState: Equatable {
    case unloaded
    case loading
    case ready(modelName: String, nLayers: Int)
    case generating
    case error(String)
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

@MainActor
final class InferenceEngine: ObservableObject {

    // ── Published state ───────────────────────────────────────────────────────
    @Published var modelState: ModelState = .unloaded
    @Published var generatedText: String  = ""
    @Published var tokensPerSecond: Double = 0
    @Published var rpcServerState: RPCServerState = .idle
    @Published var serverRegistrationStatus: String = ""
    @Published var chatMessages: [ChatMessage] = []

    // ── nonisolated(unsafe): accessed from detached tasks, single-threaded ────
    nonisolated(unsafe) private let bridge = LlamaBridge()
    private var generationTask: Task<Void, Never>?
    private var rpcServerTask:  Task<Void, Never>?
    private var keepaliveTask:  Task<Void, Never>?
    private var discoveryTask:  Task<Void, Never>?

    static let shared = InferenceEngine()

    // ── Model loading ─────────────────────────────────────────────────────────

    func loadModel(from url: URL, contextLength: Int = 1024) async {
        modelState    = .loading
        generatedText = ""

        let path = url.path

        // Detach so the blocking file-read doesn't stall the main actor.
        // Await `.value` on its own line so Swift sees the async suspension.
        let loadOp = Task.detached(priority: .userInitiated) { [bridge] in
            // ObjC `- (BOOL)method:(T)a error:(NSError **)e` is bridged as
            // `func method(_ a: T) throws` in Swift — no `error:` argument.
            do {
                try bridge.loadModel(fromPath: path, nCtx: contextLength)
                let name    = bridge.modelInfo?.name    ?? "Unknown"
                let nLayers = bridge.modelInfo?.nLayers ?? 0
                return (name, nLayers, "" as String)
            } catch {
                return ("", 0, error.localizedDescription)
            }
        }
        let (name, nLayers, errMsg) = await loadOp.value

        if errMsg.isEmpty {
            modelState = .ready(modelName: name, nLayers: nLayers)
        } else {
            modelState = .error(errMsg)
        }
    }

    func unloadModel() {
        generationTask?.cancel()
        generationTask = nil
        bridge.unloadModel()
        modelState    = .unloaded
        generatedText = ""
        tokensPerSecond = 0
        chatMessages  = []
    }

    var modelInfo: LlamaModelInfo? { bridge.modelInfo }
    var eosTokenID: Int32 { bridge.modelInfo?.eosTokenID ?? 2 }

    // ── Single-device generation ──────────────────────────────────────────────

    func generate(
        prompt: String,
        config: LlamaGenerationConfig = .defaults()
    ) -> AsyncStream<GenerationResult> {
        // IMPORTANT: do NOT touch `generationTask` here.
        // `generateIntoState` already assigned `generationTask` to the Task
        // that is currently running this call.  If we cancel/replace it here
        // we would cancel our own caller, causing the `for await` loop to exit
        // immediately and the model state to revert to .ready without generating
        // a single token.
        AsyncStream { continuation in
            let innerTask = Task.detached(priority: .userInitiated) { [bridge] in
                var accumulated = ""
                var tokenCount  = 0
                let start       = Date()

                // ObjC callback: (NSString * _Nonnull, BOOL) → (String, Bool)
                // NSString * inside NS_ASSUME_NONNULL_BEGIN is non-optional.
                bridge.generate(fromPrompt: prompt, config: config) { piece, done in
                    accumulated += piece
                    tokenCount  += 1

                    let elapsed = Date().timeIntervalSince(start)
                    let tps     = elapsed > 0 ? Double(tokenCount) / elapsed : 0

                    continuation.yield(GenerationResult(
                        text: accumulated,
                        isDone: done,
                        tokensPerSecond: tps
                    ))
                    if done { continuation.finish() }
                }
            }
            // When the consumer (generateIntoState's for-await loop) is cancelled,
            // also cancel the inner detached task so it can exit cleanly.
            continuation.onTermination = { _ in innerTask.cancel() }
        }
    }

    func generateIntoState(prompt: String, config: LlamaGenerationConfig = .defaults()) {
        guard case .ready = modelState else { return }
        modelState    = .generating
        generatedText = ""
        tokensPerSecond = 0

        generationTask = Task { @MainActor in
            for await result in generate(prompt: prompt, config: config) {
                generatedText   = result.text
                tokensPerSecond = result.tokensPerSecond
            }
            if case .generating = modelState, let info = bridge.modelInfo {
                modelState = .ready(modelName: info.name, nLayers: info.nLayers)
            }
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if let info = bridge.modelInfo {
            modelState = .ready(modelName: info.name, nLayers: info.nLayers)
        }
    }

    // ── Conversation mode ─────────────────────────────────────────────────────

    /// Send a user message and stream the assistant reply into `chatMessages`.
    func sendMessage(_ text: String, config: LlamaGenerationConfig = .defaults()) {
        guard case .ready = modelState else { return }

        // Build the message list for the template (all history + new user turn).
        let historyForTemplate = chatMessages + [ChatMessage(role: "user", content: text)]
        let nsMessages = historyForTemplate.map { ["role": $0.role, "content": $0.content] }

        // Apply the model's built-in chat template (from GGUF metadata).
        guard let formatted = bridge.applyChatTemplate(nsMessages, addAssistantPrefix: true) else {
            modelState = .error("Model has no chat template — cannot use conversation mode")
            return
        }

        chatMessages.append(ChatMessage(role: "user", content: text))
        chatMessages.append(ChatMessage(role: "assistant", content: ""))
        let assistantIndex = chatMessages.count - 1

        modelState      = .generating
        tokensPerSecond = 0

        generationTask = Task { @MainActor in
            for await result in generate(prompt: formatted, config: config) {
                chatMessages[assistantIndex].content = result.text
                tokensPerSecond = result.tokensPerSecond
            }
            if case .generating = modelState, let info = bridge.modelInfo {
                modelState = .ready(modelName: info.name, nLayers: info.nLayers)
            }
        }
    }

    func clearChat() {
        generationTask?.cancel()
        generationTask = nil
        chatMessages   = []
        tokensPerSecond = 0
        if let info = bridge.modelInfo {
            modelState = .ready(modelName: info.name, nLayers: info.nLayers)
        }
    }

    // ── GGML RPC worker server ────────────────────────────────────────────────

    /// Whether the GGML RPC backend was compiled in.
    /// Requires ggml-rpc.xcframework (rebuild with GGML_RPC=ON).
    var rpcAvailable: Bool { LlamaBridge.rpcAvailable() }

    /// Start the GGML RPC server so an external llama-cli can use this device
    /// as a Metal compute backend.  The phone is a leaf node only — it never
    /// coordinates inference itself.
    ///
    /// - Parameter port: TCP port to listen on (default 50052, same as Android team).
    func startRPCServer(
        host: String = "0.0.0.0",
        port: Int = 50052,
        discoveryIp: String = "255.255.255.255",
        discoveryPort: Int = 50055,
        threads: Int = 4
    ) {
        guard case .idle = rpcServerState else { return }
        let endpoint = "\(host):\(port)"
        rpcServerState = .starting
        
        startDiscoveryPing(discoveryIp: discoveryIp, discoveryPort: discoveryPort, servicePort: port)

        rpcServerTask = Task.detached(priority: .userInitiated) { [bridge] in
            if !LlamaBridge.rpcAvailable() {
                await MainActor.run {
                    self.rpcServerState = .unavailable(
                        "ggml-rpc not compiled in. Run scripts/build-ggml-ios.sh then add ggml-rpc.xcframework to the Xcode target.")
                }
                return
            }

            let (freeMB, totalMB) = Self.deviceMemoryMB()
            await MainActor.run {
                self.rpcServerState = .running(endpoint: endpoint)
#if canImport(UIKit)
                UIApplication.shared.isIdleTimerDisabled = true
#endif
            }
            let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?.path
            // Blocking call – returns only when the server socket is closed.
            bridge.startRPCServer(endpoint, cacheDir: cacheDir, freeMB: freeMB, totalMB: totalMB, threads: UInt(threads))
            await MainActor.run {
                self.rpcServerState = .idle
#if canImport(UIKit)
                UIApplication.shared.isIdleTimerDisabled = false
#endif
            }
        }
    }

    /// Returns (freeMB, totalMB) reflecting actual device memory at call time.
    /// Uses os_proc_available_memory() — the per-process jetsam headroom —
    /// so the coordinator never offloads more than the phone can hold.
    /// A 10% safety margin is subtracted for app/Metal overhead.
    private nonisolated static func deviceMemoryMB() -> (freeMB: UInt, totalMB: UInt) {
        let total = UInt(ProcessInfo.processInfo.physicalMemory / 1_048_576)
        let rawFree = UInt(LlamaBridge.processAvailableMemoryBytes() / 1_048_576)
        let free = UInt(Double(rawFree) * 0.9)   // 10% headroom for app/Metal overhead
        return (freeMB: free, totalMB: total)
    }

    /// Stop the RPC server.
    /// Note: this cancels the Swift Task; the underlying C server loop will be
    /// interrupted when the OS reclaims the socket on thread teardown.
    func stopRPCServer() {
        rpcServerTask?.cancel()
        rpcServerTask = nil
        stopKeepalive()
        stopDiscoveryPing()
        rpcServerState = .idle
    }
    
    // ── UDP Discovery Ping ────────────────────────────────────────────────────
    
    private func startDiscoveryPing(discoveryIp: String, discoveryPort: Int, servicePort: Int) {
        stopDiscoveryPing()
        discoveryTask = Task.detached {
            // Enable battery monitoring for the lifetime of this task.
            #if canImport(UIKit)
            await MainActor.run { UIDevice.current.isBatteryMonitoringEnabled = true }
            #endif

            while !Task.isCancelled {
                do {
                    // Collect device info on each ping so values stay fresh.
                    let hwModel   = Self.hardwareModel()
                    let maxBytes  = LlamaBridge.availableProcessMemoryBytes()
                    #if canImport(UIKit)
                    let battery   = await MainActor.run { UIDevice.current.batteryLevel }   // 0–1 or -1
                    #else
                    let battery: Float = -1
                    #endif
                    let tempCode  = await MainActor.run { ProcessInfo.processInfo.thermalState }
                    let tempC     = Self.thermalStateTemperature(tempCode)

                    var comps = URLComponents()
                    comps.scheme = "http"
                    comps.host   = discoveryIp
                    comps.port   = discoveryPort
                    comps.path   = "/announce"
                    var items: [URLQueryItem] = [
                        .init(name: "port",     value: "\(servicePort)"),
                        .init(name: "model",    value: hwModel),
                        .init(name: "max_size", value: "\(maxBytes)"),
                    ]
                    if battery >= 0 {
                        items.append(.init(name: "battery", value: String(format: "%.1f", battery * 100)))
                    }
                    if !tempC.isNaN {
                        items.append(.init(name: "temperature", value: String(format: "%.1f", tempC)))
                    }
                    comps.queryItems = items
                    guard let url = comps.url else { return }

                    var req = URLRequest(url: url)
                    req.httpMethod  = "GET"
                    req.timeoutInterval = 5

                    let (data, response) = try await URLSession.shared.data(for: req)

                    if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let interval = json["interval"] as? Double {
                        try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                    } else {
                        try await Task.sleep(nanoseconds: 1_000_000_000)
                    }
                } catch {
                    if error is CancellationError { break }
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }

            #if canImport(UIKit)
            await MainActor.run { UIDevice.current.isBatteryMonitoringEnabled = false }
            #endif
        }
    }

    // Returns the hardware model identifier, e.g. "iPhone16,1".
    private nonisolated static func hardwareModel() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }

    // Maps ProcessInfo.ThermalState to an approximate temperature in °C.
    private nonisolated static func thermalStateTemperature(_ state: ProcessInfo.ThermalState) -> Double {
        switch state {
        case .nominal:  return 30.0
        case .fair:     return 38.0
        case .serious:  return 45.0
        case .critical: return 55.0
        @unknown default: return Double.nan
        }
    }

    private func stopDiscoveryPing() {
        discoveryTask?.cancel()
        discoveryTask = nil
    }

    // ── Server registration / keepalive ──────────────────────────────────────

    /// Register this device with the orchestration server so the operator can
    /// retrieve the `--rpc` command without manually noting IP addresses.
    func registerWithServer(_ serverURL: URL,
                             deviceID: String,
                             label: String,
                             ip: String,
                             rpcPort: Int) async {
        struct Reg: Encodable {
            let device_id, label, ip: String
            let rpc_port: Int
        }
        let body = Reg(device_id: deviceID, label: label, ip: ip, rpc_port: rpcPort)
        guard let data = try? JSONEncoder().encode(body) else { return }

        var req = URLRequest(url: serverURL.appendingPathComponent("api/v1/devices/register"))
        req.httpMethod  = "POST"
        req.httpBody    = data
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 5

        do {
            let (_, _) = try await URLSession.shared.data(for: req)
            serverRegistrationStatus = "Registered with \(serverURL.host ?? serverURL.absoluteString)"
            startKeepalive(serverURL: serverURL, deviceID: deviceID)
        } catch {
            serverRegistrationStatus = "Registration failed: \(error.localizedDescription)"
        }
    }

    private func startKeepalive(serverURL: URL, deviceID: String) {
        stopKeepalive()
        keepaliveTask = Task {
            let url = serverURL.appendingPathComponent("api/v1/devices/\(deviceID)/keepalive")
            while !Task.isCancelled {
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.timeoutInterval = 5
                _ = try? await URLSession.shared.data(for: req)
                try? await Task.sleep(nanoseconds: 10_000_000_000) // 10 s
            }
        }
    }

    private func stopKeepalive() {
        keepaliveTask?.cancel()
        keepaliveTask = nil
        serverRegistrationStatus = ""
    }

    // ── Distributed shard helpers ─────────────────────────────────────────────

    func runFirstShard(
        tokens: [Int32],
        endLayer: Int
    ) async -> (hiddenState: Data, tokenCount: Int, nEmbd: Int) {
        // The ObjC callback is synchronous (fires before the method returns),
        // so we don't need withCheckedContinuation — just run inside a detached
        // task so we don't block the main actor.
        let op = Task.detached(priority: .userInitiated) { [bridge] in
            let nsTokens = tokens.map { NSNumber(value: $0) }
            var result: (Data, Int, Int) = (Data(), 0, 0)
            bridge.runFirstShard(withTokens: nsTokens, endLayer: endLayer) {
                state, count, embd, _ in
                result = (state, Int(count), Int(embd))
            }
            return result
        }
        return await op.value
    }

    func runShard(
        hiddenState: Data,
        tokenCount: Int,
        startLayer: Int,
        endLayer: Int
    ) async -> (hiddenState: Data, tokenCount: Int, nEmbd: Int) {
        let op = Task.detached(priority: .userInitiated) { [bridge] in
            var result: (Data, Int, Int) = (Data(), 0, 0)
            bridge.runShard(withHiddenState: hiddenState,
                            tokenCount: tokenCount,
                            startLayer: startLayer,
                            endLayer: endLayer) { state, count, embd, _ in
                result = (state, Int(count), Int(embd))
            }
            return result
        }
        return await op.value
    }

    // ── Tokenization helpers ──────────────────────────────────────────────────

    func tokenize(text: String, addBOS: Bool = true) -> [Int32] {
        bridge.tokenizeText(text, addBOS: addBOS).map { $0.int32Value }
    }

    func tokenToPiece(_ tokenID: Int32) -> String {
        // ObjC `tokenToPiece:` is renamed by Swift to `tokenPiece(_:)`
        // via NS_SWIFT_NAME(tokenPiece(_:)) in LlamaBridge.h
        bridge.tokenPiece(tokenID)
    }
}
