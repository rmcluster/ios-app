// InferenceView.swift
//
// Primary UI for single-device GGML inference and GGML RPC worker mode.
//
// Sections:
//   1. Model loader – pick a .gguf file from the Documents directory.
//   2. Model status card – name, layers, embedding dimension, etc.
//   3. Prompt + generation configuration controls (single-device).
//   4. Streaming output area.
//   5. RPC Worker panel – expose this device as a Metal compute backend.
//      The phone is a *leaf node* only; a llama-cli process on a laptop or
//      server acts as the coordinator via the GGML RPC protocol.

import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct InferenceView: View {

    @EnvironmentObject private var engine: InferenceEngine

    // ── UI state ──────────────────────────────────────────────────────────────
    @State private var chatInput     = ""
    @State private var contextSize   = 1024
    @State private var maxTokens     = 200
    @State private var temperature   = 0.8
    @State private var showDocPicker = false
    @State private var localModels: [URL] = []

    // ── RPC worker state ──────────────────────────────────────────────────────
    @AppStorage("rpcHost") private var rpcHost: String = "0.0.0.0"
    @AppStorage("rpcPort") private var rpcPort: Int = 50052
    @AppStorage("rpcDiscoveryIp") private var rpcDiscoveryIp: String = ""
    @AppStorage("rpcDiscoveryPort") private var rpcDiscoveryPort: Int = 4917
    @AppStorage("rpcThreads") private var rpcThreads: Int = 4
    
    @State private var serverURL:  String = ""
    @State private var showRPC:    Bool   = true
    @State private var selectedTab: Int  = 1

    // ── Body ──────────────────────────────────────────────────────────────────

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationView {
                List {
                    modelSection
                    if case .ready = engine.modelState {
                        chatSection
                    }
                    if case .generating = engine.modelState {
                        chatSection
                    }
                }
                .navigationTitle("Inference")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { toolbarContent }
                .sheet(isPresented: $showDocPicker) { documentPicker }
            }
            .navigationViewStyle(.stack)
            .tabItem { Label("Inference", systemImage: "cpu") }
            .tag(0)

            NavigationView {
                List {
                    rpcWorkerSection
                }
                .navigationTitle("GGML RPC Worker")
                .navigationBarTitleDisplayMode(.inline)
            }
            .navigationViewStyle(.stack)
            .tabItem { Label("GGML RPC Worker", systemImage: "network") }
            .tag(1)
        }
        .onAppear { refreshLocalModels() }
    }

    // ── Model section ─────────────────────────────────────────────────────────

    @ViewBuilder
    private var modelSection: some View {
        Section("Model") {
            switch engine.modelState {
            case .unloaded:
                if localModels.isEmpty {
                    Button { showDocPicker = true } label: {
                        Label("Load .gguf model…", systemImage: "doc.badge.plus")
                    }
                } else {
                    ForEach(localModels, id: \.self) { url in
                        Button {
                            Task { await engine.loadModel(from: url, contextLength: contextSize) }
                        } label: {
                            Label(url.lastPathComponent, systemImage: "cpu")
                        }
                    }
                    Button { showDocPicker = true } label: {
                        Label("Load other…", systemImage: "doc.badge.plus")
                    }
                    .foregroundStyle(.secondary)
                }

            case .loading:
                HStack {
                    ProgressView()
                    Text("Loading model…").foregroundStyle(.secondary)
                }

            case .ready(let name, let nLayers):
                VStack(alignment: .leading, spacing: 4) {
                    Label(name, systemImage: "cpu").font(.headline)
                    if let info = engine.modelInfo {
                        HStack(spacing: 16) {
                            StatChip(label: "\(nLayers) layers")
                            StatChip(label: "embd \(info.nEmbd)")
                            StatChip(label: "ctx \(info.nCtx)")
                            StatChip(label: sizeString(info.fileSizeBytes))
                        }
                    }
                }
                .padding(.vertical, 2)
                Button(role: .destructive) { engine.unloadModel() } label: {
                    Label("Unload model", systemImage: "eject")
                }

            case .generating:
                HStack {
                    ProgressView()
                    Text("Generating…").foregroundStyle(.secondary)
                    Spacer()
                    if engine.tokensPerSecond > 0 {
                        Text(String(format: "%.1f tok/s", engine.tokensPerSecond))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Button("Stop") { engine.cancelGeneration() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .tint(.red)
                }

            case .error(let msg):
                Label(msg, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                Button("Try again") { showDocPicker = true }
            }
        }
    }

    // ── Conversation ──────────────────────────────────────────────────────────

    private var isGenerating: Bool {
        if case .generating = engine.modelState { return true }
        return false
    }

    @ViewBuilder
    private var chatSection: some View {
        // Message history
        if !engine.chatMessages.isEmpty {
            Section("Conversation") {
                ForEach(engine.chatMessages) { msg in
                    VStack(alignment: msg.role == "user" ? .trailing : .leading, spacing: 2) {
                        Text(msg.role == "user" ? "You" : "Assistant")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(msg.content.isEmpty ? "…" : msg.content)
                            .font(.body)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity,
                                   alignment: msg.role == "user" ? .trailing : .leading)
                    }
                    .padding(.vertical, 2)
                }
                if engine.tokensPerSecond > 0 {
                    HStack {
                        Spacer()
                        Text(String(format: "%.1f tok/s", engine.tokensPerSecond))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }

        // Input bar
        Section {
            HStack(alignment: .bottom, spacing: 10) {
                Group {
                    if #available(iOS 16.0, *) {
                        TextField("Message…", text: $chatInput, axis: .vertical)
                            .lineLimit(1...5)
                    } else {
                        TextField("Message…", text: $chatInput)
                            .lineLimit(5)
                    }
                }
                .disabled(isGenerating)
                Button {
                    if isGenerating {
                        engine.cancelGeneration()
                    } else {
                        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !text.isEmpty else { return }
                        chatInput = ""
                        let config = LlamaGenerationConfig.defaults()
                        config.maxNewTokens = maxTokens
                        config.temperature  = Float(temperature)
                        engine.sendMessage(text, config: config)
                    }
                } label: {
                    Image(systemName: isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(
                            chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating
                                ? Color.secondary : Color.accentColor
                        )
                }
                .buttonStyle(.plain)
            }
        }

        // Parameters
        Section("Parameters") {
            HStack {
                Text("Max tokens")
                Spacer()
                Stepper("\(maxTokens)", value: $maxTokens, in: 1...2048, step: 50)
            }
            HStack {
                Text(String(format: "Temperature  %.2f", temperature))
                Spacer()
                Slider(value: $temperature, in: 0...2, step: 0.05)
            }
        }

        // Clear
        if !engine.chatMessages.isEmpty {
            Section {
                Button(role: .destructive) {
                    engine.clearChat()
                } label: {
                    Label("Clear conversation", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    // ── RPC Worker panel ──────────────────────────────────────────────────────

    @ViewBuilder
    private var rpcWorkerSection: some View {
        if showRPC {
            let interfaces = ShardNetwork.allLocalIPv4s
            let isRunning  = rpcIsRunning

            // ── Endpoint card ─────────────────────────────────────────────────
            Section("Endpoints") {
                if interfaces.isEmpty {
                    Label("No network interfaces found", systemImage: "wifi.slash")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(interfaces) { iface in
                        HStack(spacing: 12) {
                            Circle()
                                .fill(isRunning ? Color.green : Color.secondary.opacity(0.35))
                                .frame(width: 9, height: 9)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(verbatim: "\(iface.ip):\(rpcPort)")
                                    .font(.system(.body, design: .monospaced).bold())
                                Text(iface.label)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button {
                                UIPasteboard.general.string = "\(iface.ip):\(rpcPort)"
                            } label: {
                                Image(systemName: "doc.on.doc")
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(Color.accentColor)
                        }
                        .padding(.vertical, 2)
                    }
                }

                if isRunning {
                    Text(rpcStateLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                IntStepperField("Thread count", value: $rpcThreads, in: 1...64, disabled: isRunning)
                HStack {
                    Text("Host")
                    Spacer()
                    TextField("0.0.0.0", text: $rpcHost)
                        .disabled(isRunning)
                        .multilineTextAlignment(.trailing)
                        .frame(maxWidth: 160)
                }
                IntStepperField("Port", value: $rpcPort, in: 1024...65535, disabled: isRunning)
                HStack {
                    Text("Discovery IP")
                    Spacer()
                    TextField("LAN IP of server", text: $rpcDiscoveryIp)
                        .disabled(isRunning)
                        .multilineTextAlignment(.trailing)
                        .frame(maxWidth: 160)
                }
                IntStepperField("Discovery Port", value: $rpcDiscoveryPort, in: 1024...65535, disabled: isRunning)
            }

            // ── Start / Stop ──────────────────────────────────────────────────
            Section {
                if case .unavailable(let msg) = engine.rpcServerState {
                    Label(msg, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.red)
                } else if isRunning {
                    Button(role: .destructive) {
                        engine.stopRPCServer()
                    } label: {
                        Label("Stop RPC server", systemImage: "stop.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                } else {
                    Button {
                        engine.startRPCServer(
                            host: rpcHost,
                            port: rpcPort,
                            discoveryIp: rpcDiscoveryIp,
                            discoveryPort: rpcDiscoveryPort,
                            threads: rpcThreads
                        )
                    } label: {
                        Label(
                            engine.rpcServerState == .starting ? "Starting…" : "Start RPC server",
                            systemImage: "play.circle"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(engine.rpcServerState == .starting || interfaces.isEmpty)
                }
            }

            // ── Optional device registry ───────────────────────────────────────
            // Section {
            //     TextField("Registry URL  e.g. http://192.168.1.100:8080",
            //               text: $serverURL)
            //         .keyboardType(.URL)
            //         .autocorrectionDisabled()
            //         .textInputAutocapitalization(.never)
            //     if !engine.serverRegistrationStatus.isEmpty {
            //         Text(engine.serverRegistrationStatus)
            //             .font(.caption)
            //             .foregroundStyle(.secondary)
            //     }
            //     Button("Register this device") {
            //         registerWithServer(ip: primaryIP ?? "")
            //     }
            //     .disabled(serverURL.trimmingCharacters(in: .whitespaces).isEmpty
            //               || primaryIP == nil)
            // } header: {
            //     Text("Device registry (optional)")
            // } footer: {
            //     Text("Register with the orchestration server so coordinators can look up this device's endpoint without manually noting the IP address.")
            // }
        }
    }

    // ── RPC helpers ───────────────────────────────────────────────────────────

    private var rpcIsRunning: Bool {
        if case .running = engine.rpcServerState { return true }
        return false
    }

    private var rpcStateLabel: String {
        switch engine.rpcServerState {
        case .idle:                return "Stopped"
        case .starting:            return "Starting…"
        case .running(let ep):     return "Listening on \(ep)"
        case .unavailable:         return "Unavailable – rebuild with GGML_RPC=ON"
        }
    }

    private func registerWithServer(ip: String) {
        let trimmed = serverURL.trimmingCharacters(in: .whitespaces)
        guard let url = URL(string: trimmed) else { return }
        Task {
            await engine.registerWithServer(
                url,
                deviceID: UIDeviceLabel.deviceID,
                label:    UIDeviceLabel.current,
                ip:       ip,
                rpcPort:  rpcPort
            )
        }
    }

    // ── Toolbar ───────────────────────────────────────────────────────────────

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            if case .unloaded = engine.modelState {
                Button { showDocPicker = true } label: {
                    Image(systemName: "folder")
                }
            }
        }
    }

    // ── Document picker ───────────────────────────────────────────────────────

    private var documentPicker: some View {
        DocumentPicker(contentTypes: [.init(filenameExtension: "gguf") ?? .data]) { url in
            Task { await engine.loadModel(from: url, contextLength: contextSize) }
        }
    }

    // ── Actions ───────────────────────────────────────────────────────────────

    private func refreshLocalModels() {
        let docs = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
        localModels = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil))?.filter {
            $0.pathExtension.lowercased() == "gguf"
        }.sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private func sizeString(_ bytes: UInt) -> String {
        let mb = Double(bytes) / 1_048_576
        if mb >= 1000 { return String(format: "%.1f GB", mb / 1024) }
        return String(format: "%.0f MB", mb)
    }
}

// ── Sub-views ─────────────────────────────────────────────────────────────────

private struct StatChip: View {
    let label: String
    var body: some View {
        Text(label)
            .font(.caption2.monospaced())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
    }
}

// ── UIDocumentPickerViewController wrapper ────────────────────────────────────

private struct DocumentPicker: UIViewControllerRepresentable {
    let contentTypes: [UTType]
    let onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let vc = UIDocumentPickerViewController(forOpeningContentTypes: contentTypes)
        vc.delegate = context.coordinator
        vc.allowsMultipleSelection = false
        return vc
    }

    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController,
                            didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            guard url.startAccessingSecurityScopedResource() else { return }
            defer { url.stopAccessingSecurityScopedResource() }

            // Copy to app's Documents directory so we keep access after the picker closes
            let dest = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.copyItem(at: url, to: dest)
            onPick(dest)
        }
    }
}

// ── Int field with inline text entry + stepper ────────────────────────────────

private struct IntStepperField: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let disabled: Bool

    @State private var text: String = ""
    @FocusState private var focused: Bool

    init(_ label: String, value: Binding<Int>, in range: ClosedRange<Int>, disabled: Bool) {
        self.label    = label
        self._value   = value
        self.range    = range
        self.disabled = disabled
        self._text    = State(initialValue: String(value.wrappedValue))
    }

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            TextField("", text: $text)
                .keyboardType(.numberPad)
                .multilineTextAlignment(.trailing)
                .frame(width: 64)
                .focused($focused)
                .disabled(disabled)
                .onChange(of: focused) { isFocused in
                    if !isFocused { commit() }
                }
                .onChange(of: value) { newVal in
                    if !focused { text = String(newVal) }
                }
            Stepper("", value: $value, in: range, step: 1)
                .labelsHidden()
                .disabled(disabled)
                .onChange(of: value) { newVal in
                    text = String(newVal)
                }
        }
    }

    private func commit() {
        if let parsed = Int(text), range.contains(parsed) {
            value = parsed
        } else {
            text = String(value)
        }
    }
}

// ── Preview ───────────────────────────────────────────────────────────────────

#Preview {
    InferenceView()
        .environmentObject(InferenceEngine.shared)
}
