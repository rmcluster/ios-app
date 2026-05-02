//
//  distributed_ml_ggml_client_iosApp.swift
//  distributed-ml-ggml-client-ios
//
//  Created by Sandeep Reehal on 2/23/26.
//

import SwiftUI

@main
struct distributed_ml_ggml_client_iosApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(InferenceEngine.shared)
                .environmentObject(RpcSettings.shared)
        }
    }
}
