import Foundation
import Network
import CryptoKit
import Darwin

extension NWConnection {
    func receiveAsync(minimumIncompleteLength: Int, maximumLength: Int) async throws -> (Data?, NWConnection.ContentContext?, Bool) {
        return try await withCheckedThrowingContinuation { continuation in
            self.receive(minimumIncompleteLength: minimumIncompleteLength, maximumLength: maximumLength) { data, context, isComplete, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: (data, context, isComplete))
                }
            }
        }
    }

    func sendAsync(content: Data, isComplete: Bool = true, context: NWConnection.ContentContext = .defaultMessage) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            self.send(content: content, contentContext: context, isComplete: isComplete, completion: .contentProcessed { error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            })
        }
    }
}

final class StorageServer {
    private struct RequestHeader {
        let method: String
        let path: String
        let queryItems: [URLQueryItem]
        let headers: [String: String]
        let leftoverBody: Data
    }

    private struct StorageHealth {
        let timestamp: Date
        let status: String
        let badChunks: [String]
    }

    private static let sha256HexChars = CharacterSet(charactersIn: "0123456789abcdefABCDEF")
    private static let minFreeBytes: Int64 = 50 * 1024 * 1024

    private let storageDir: URL
    private let queue = DispatchQueue(label: "StorageServer.queue")
    private var listener: NWListener?
    private var healthCache: StorageHealth?

    init(storageDir: URL) {
        self.storageDir = storageDir
    }

    func start(port: Int) -> Bool {
        guard let nwPort = NWEndpoint.Port(rawValue: UInt16(port)) else { return false }
        do {
            try FileManager.default.createDirectory(at: storageDir, withIntermediateDirectories: true)
        } catch {
            return false
        }
        guard let listener = try? NWListener(using: .tcp, on: nwPort) else {
            return false
        }
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            self?.handleConnection(connection)
        }
        listener.start(queue: queue)
        return true
    }

    func stop() {
        listener?.cancel()
        listener = nil
        healthCache = nil
    }

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: queue)
        Task {
            do {
                try await handleConnectionAsync(connection)
            } catch {
                connection.cancel()
            }
        }
    }

    private func handleConnectionAsync(_ connection: NWConnection) async throws {
        var buffer = Data()
        let headerMarker = Data("\r\n\r\n".utf8)
        
        // 1. Read headers
        while true {
            if let range = buffer.range(of: headerMarker) {
                let headerData = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
                let leftover = buffer.subdata(in: range.upperBound..<buffer.endIndex)
                guard let requestHeader = parseHeader(headerData, leftover: leftover) else {
                    try await sendResponse(status: 400, body: Data("Invalid Header".utf8), contentType: "text/plain", on: connection)
                    return
                }
                try await routeAsync(requestHeader, on: connection)
                return
            }
            
            let (data, _, isComplete) = try await connection.receiveAsync(minimumIncompleteLength: 1, maximumLength: 64 * 1024)
            if let data = data {
                buffer.append(data)
            }
            if isComplete && buffer.range(of: headerMarker) == nil {
                connection.cancel()
                return
            }
        }
    }

    private func parseHeader(_ data: Data, leftover: Data) -> RequestHeader? {
        guard let headerText = String(data: data, encoding: .utf8) else { return nil }
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return nil }
        
        let method = String(parts[0])
        let target = String(parts[1])
        
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let pieces = line.split(separator: ":", maxSplits: 1)
            if pieces.count == 2 {
                headers[String(pieces[0]).lowercased()] = pieces[1].trimmingCharacters(in: .whitespaces)
            }
        }
        
        let baseURL = URL(string: "http://localhost")!
        guard let url = URL(string: target, relativeTo: baseURL),
              let comps = URLComponents(url: url, resolvingAgainstBaseURL: true) else {
            return nil
        }
        
        return RequestHeader(
            method: method,
            path: comps.path,
            queryItems: comps.queryItems ?? [],
            headers: headers,
            leftoverBody: leftover
        )
    }

    private func routeAsync(_ header: RequestHeader, on connection: NWConnection) async throws {
        if header.path.hasPrefix("/chunk/") {
            let chunkId = String(header.path.dropFirst("/chunk/".count))
            if !isValidSHA256(chunkId) {
                if header.method == "PUT" {
                    let body = jsonBody(["error": "bad_id"])
                    try await sendResponse(status: 400, body: body, contentType: "application/json", on: connection)
                } else {
                    try await sendResponse(status: 400, body: Data("Invalid chunk ID format".utf8), contentType: "text/plain", on: connection)
                }
                return
            }
            
            switch header.method {
            case "GET":
                try await handleGetChunk(id: chunkId, on: connection)
            case "PUT":
                try await handlePutChunk(id: chunkId, header: header, on: connection)
            case "DELETE":
                try await handleDeleteChunk(id: chunkId, on: connection)
            default:
                try await sendResponse(status: 404, body: Data("Not Found".utf8), contentType: "text/plain", on: connection)
            }
            return
        }

        switch (header.method, header.path) {
        case ("GET", "/chunks/list"):
            try await handleListChunks(on: connection)
        case ("GET", "/chunks/healthcheck"):
            try await handleHealthCheck(queryItems: header.queryItems, on: connection)
        case ("GET", "/storage_info"):
            try await handleStorageInfo(on: connection)
        default:
            try await sendResponse(status: 404, body: Data("Not Found".utf8), contentType: "text/plain", on: connection)
        }
    }

    private func handleGetChunk(id: String, on connection: NWConnection) async throws {
        let fileURL = storageDir.appendingPathComponent(id)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            let body = jsonBody(["error": "not_found"])
            try await sendResponse(status: 404, body: body, contentType: "application/json", on: connection)
            return
        }
        guard let actualHash = try? sha256Hex(for: fileURL), actualHash.caseInsensitiveCompare(id) == .orderedSame else {
            let body = jsonBody(["error": "corrupted_chunk"])
            try await sendResponse(status: 404, body: body, contentType: "application/json", on: connection)
            return
        }
        guard let data = try? Data(contentsOf: fileURL) else {
            try await sendResponse(status: 500, body: Data("Read failed".utf8), contentType: "text/plain", on: connection)
            return
        }
        try await sendResponse(status: 200, body: data, contentType: "application/octet-stream", on: connection)
    }

    private func handlePutChunk(id: String, header: RequestHeader, on connection: NWConnection) async throws {
        let contentLength = Int64(header.headers["content-length"] ?? "0") ?? 0
        let available = availableBytes()
        if available - contentLength < Self.minFreeBytes {
            let body = jsonBody(["error": "insufficient_storage"])
            try await sendResponse(status: 507, body: body, contentType: "application/json", on: connection)
            return
        }
        
        let tempURL = storageDir.appendingPathComponent(UUID().uuidString + ".tmp")
        FileManager.default.createFile(atPath: tempURL.path, contents: nil)
        
        do {
            let handle = try FileHandle(forWritingTo: tempURL)
            defer { try? handle.close() }
            
            var bytesWritten: Int64 = 0
            if !header.leftoverBody.isEmpty {
                let toWrite = min(Int64(header.leftoverBody.count), contentLength)
                try handle.write(contentsOf: header.leftoverBody.prefix(Int(toWrite)))
                bytesWritten += toWrite
            }
            
            while bytesWritten < contentLength {
                let toRead = Int(min(contentLength - bytesWritten, 1024 * 1024))
                let (data, _, isComplete) = try await connection.receiveAsync(minimumIncompleteLength: 1, maximumLength: toRead)
                guard let data = data, !data.isEmpty else {
                    if isComplete { break }
                    continue
                }
                try handle.write(contentsOf: data)
                bytesWritten += Int64(data.count)
            }
            try? handle.close()
            
            let actualHash = try sha256Hex(for: tempURL)
            guard actualHash.caseInsensitiveCompare(id) == .orderedSame else {
                try? FileManager.default.removeItem(at: tempURL)
                let body = jsonBody(["error": "checksum_incorrect"])
                try await sendResponse(status: 400, body: body, contentType: "application/json", on: connection)
                return
            }
            
            let targetURL = storageDir.appendingPathComponent(id)
            if FileManager.default.fileExists(atPath: targetURL.path) {
                try? FileManager.default.removeItem(at: targetURL)
            }
            try FileManager.default.moveItem(at: tempURL, to: targetURL)
            healthCache = nil
            try await sendResponse(status: 200, body: Data("OK".utf8), contentType: "text/plain", on: connection)
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            try await sendResponse(status: 500, body: Data("Write failed".utf8), contentType: "text/plain", on: connection)
        }
    }

    private func handleDeleteChunk(id: String, on connection: NWConnection) async throws {
        let fileURL = storageDir.appendingPathComponent(id)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            let body = jsonBody(["error": "not_found"])
            try await sendResponse(status: 404, body: body, contentType: "application/json", on: connection)
            return
        }
        do {
            try FileManager.default.removeItem(at: fileURL)
            healthCache = nil
            try await sendResponse(status: 200, body: Data("OK".utf8), contentType: "text/plain", on: connection)
        } catch {
            try await sendResponse(status: 500, body: Data("Delete failed".utf8), contentType: "text/plain", on: connection)
        }
    }

    private func handleListChunks(on connection: NWConnection) async throws {
        let files = (try? FileManager.default.contentsOfDirectory(at: storageDir, includingPropertiesForKeys: nil)) ?? []
        let chunkIds = files.compactMap { url -> String? in
            let name = url.lastPathComponent
            return isValidSHA256(name) ? name : nil
        }
        let body = jsonBody(chunkIds)
        try await sendResponse(status: 200, body: body, contentType: "application/json", on: connection)
    }

    private func handleHealthCheck(queryItems: [URLQueryItem], on connection: NWConnection) async throws {
        let maxAgeStr = queryItems.first(where: { $0.name == "max_age" })?.value
        let maxAgeSec = Double(maxAgeStr ?? "300") ?? 300

        if let cached = healthCache, Date().timeIntervalSince(cached.timestamp) < maxAgeSec {
            let body = jsonBody(["status": cached.status, "bad_chunks": cached.badChunks])
            try await sendResponse(status: 200, body: body, contentType: "application/json", on: connection)
            return
        }

        var badChunks: [String] = []
        let files = (try? FileManager.default.contentsOfDirectory(at: storageDir, includingPropertiesForKeys: nil)) ?? []
        for url in files where isValidSHA256(url.lastPathComponent) {
            if let actual = try? sha256Hex(for: url), actual.caseInsensitiveCompare(url.lastPathComponent) != .orderedSame {
                badChunks.append(url.lastPathComponent)
            }
        }
        let status = badChunks.isEmpty ? "healthy" : "degraded"
        let snapshot = StorageHealth(timestamp: Date(), status: status, badChunks: badChunks)
        healthCache = snapshot
        let body = jsonBody(["status": status, "bad_chunks": badChunks])
        try await sendResponse(status: 200, body: body, contentType: "application/json", on: connection)
    }

    private func handleStorageInfo(on connection: NWConnection) async throws {
        let total = totalBytes()
        let available = availableBytes()
        let used = usedBytes()
        let body = jsonBody([
            "total_space": total,
            "used_space": used,
            "available_space": available
        ])
        try await sendResponse(status: 200, body: body, contentType: "application/json", on: connection)
    }

    private func isValidSHA256(_ value: String) -> Bool {
        guard value.count == 64 else { return false }
        return value.unicodeScalars.allSatisfy { Self.sha256HexChars.contains($0) }
    }

    private func sha256Hex(for url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 64 * 1024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func totalBytes() -> Int64 {
        let values = try? FileManager.default.attributesOfFileSystem(forPath: storageDir.path)
        return values?[.systemSize] as? Int64 ?? 0
    }

    private func availableBytes() -> Int64 {
        let values = try? FileManager.default.attributesOfFileSystem(forPath: storageDir.path)
        return values?[.systemFreeSize] as? Int64 ?? 0
    }

    private func usedBytes() -> Int64 {
        let urls = (try? FileManager.default.contentsOfDirectory(at: storageDir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        var total: Int64 = 0
        for url in urls {
            if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    private func jsonBody(_ value: Any) -> Data {
        return (try? JSONSerialization.data(withJSONObject: value, options: [])) ?? Data("{}".utf8)
    }

    private func sendResponse(status: Int, body: Data, contentType: String, on connection: NWConnection) async throws {
        let statusText = statusMessage(for: status)
        var header = "HTTP/1.1 \(status) \(statusText)\r\n"
        header += "Content-Length: \(body.count)\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Connection: close\r\n\r\n"
        var response = Data(header.utf8)
        response.append(body)
        try await connection.sendAsync(content: response)
        connection.cancel()
    }

    private func statusMessage(for code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        case 507: return "Insufficient Storage"
        default:  return "Error"
        }
    }
}
