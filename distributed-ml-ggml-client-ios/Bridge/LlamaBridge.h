// LlamaBridge.h
//
// Objective-C interface wrapping llama.cpp (GGML-based runtime) for use from Swift.
//
// Design notes for distributed / pipeline-parallel inference
// ──────────────────────────────────────────────────────────
// The primary goal of this bridge is to support *layer sharding*:
//
//   Device A  ──→  Device B  ──→  … ──→  Device N
//   layers 0..k    layers k+1..m          layers m+1..L
//
// Each device runs a contiguous range of transformer layers and forwards the
// resulting hidden-state tensor (shape [n_tokens, n_embd]) to the next device
// over the network.  The final device computes the output logits and returns the
// sampled token(s) back to the orchestrating device (Device A).
//
// Status of distributed support in llama.cpp
// ───────────────────────────────────────────
// llama.cpp does not natively expose partial-layer evaluation yet.  Two API
// hooks have been stubbed in this file (see DISTRIBUTED INFERENCE STUBS below)
// and are wired through `LlamaBridge.mm`.  A companion patch for llama.cpp
// (`scripts/llama-partial-eval.patch`) implements the required C-level functions.
// Until that patch lands upstream, each call falls back to running the full
// model locally so the app remains functional for single-device testing.

#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// ── Callback types ────────────────────────────────────────────────────────────

/// Called for each newly sampled token during generation.
/// `token`  – decoded UTF-8 piece (may be a sub-word fragment).
/// `isDone` – YES when the model emitted EOS or maxTokens was reached.
typedef void (^LlamaTokenCallback)(NSString *token, BOOL isDone);

/// Called at each decode step during a *shard* evaluation.
/// `hiddenState` – raw float32 blob, shape [tokenCount × nEmbd], row-major.
/// `tokenCount`  – number of tokens represented in this activation tensor.
/// `nEmbd`       – embedding dimension of the model.
/// `isDone`      – YES when the shard has finished processing all tokens in the prompt.
typedef void (^LlamaShardCallback)(NSData *hiddenState,
                                   NSInteger tokenCount,
                                   NSInteger nEmbd,
                                   BOOL isDone);


// ── Model information ─────────────────────────────────────────────────────────

/// Snapshot of static model metadata.
@interface LlamaModelInfo : NSObject
@property (nonatomic, readonly) NSString  *name;
@property (nonatomic, readonly) NSInteger  nLayers;      ///< total transformer layer count
@property (nonatomic, readonly) NSInteger  nEmbd;        ///< embedding / hidden-state dimension
@property (nonatomic, readonly) NSInteger  nCtx;         ///< maximum context length (tokens)
@property (nonatomic, readonly) NSInteger  nVocab;       ///< vocabulary size
@property (nonatomic, readonly) int32_t    eosTokenID;   ///< end-of-sequence token ID
@property (nonatomic, readonly) NSUInteger fileSizeBytes; ///< model file size on disk
@end


// ── Generation configuration ──────────────────────────────────────────────────

@interface LlamaGenerationConfig : NSObject
@property (nonatomic) NSInteger maxNewTokens;   ///< upper bound on tokens to generate
@property (nonatomic) float     temperature;    ///< 0 = greedy, >0 = stochastic sampling
@property (nonatomic) float     topP;           ///< nucleus sampling threshold (0–1)
@property (nonatomic) float     repeatPenalty;  ///< repetition penalty (1.0 = disabled)
@property (nonatomic) NSInteger seed;           ///< RNG seed; -1 = random
+ (instancetype)defaults;
@end


// ── Main bridge ───────────────────────────────────────────────────────────────

@interface LlamaBridge : NSObject

// ─ Model lifecycle ────────────────────────────────────────────────────────────

/// Load a GGUF model from `path`.  `nCtx` is the KV-cache token budget.
/// Returns YES on success; on failure `error` is populated.
- (BOOL)loadModelFromPath:(NSString *)path
                     nCtx:(NSInteger)nCtx
                    error:(NSError *__autoreleasing *)error;

/// Release model and context, freeing all GGML memory.
- (void)unloadModel;

/// Whether a model is currently loaded and ready for inference.
@property (nonatomic, readonly, getter=isModelLoaded) BOOL modelLoaded;

/// Returns metadata for the currently loaded model, or nil if none is loaded.
@property (nonatomic, readonly, nullable) LlamaModelInfo *modelInfo;


// ─ Single-device inference ────────────────────────────────────────────────────

/// Generate text from a prompt using all layers on this device.
/// Token pieces are delivered via `callback` on the calling thread.
/// This method blocks until generation completes.
- (void)generateFromPrompt:(NSString *)prompt
                    config:(LlamaGenerationConfig *)config
                  callback:(LlamaTokenCallback)callback;

/// Apply the model's built-in chat template (from GGUF metadata) to a list of
/// messages.  Each message is a dictionary with "role" and "content" keys.
/// Pass addAssistantPrefix:YES to append the start-of-assistant-turn marker so
/// the model continues generating the assistant reply.
/// Returns the formatted prompt string, or nil if the model has no template.
- (nullable NSString *)applyChatTemplate:(NSArray<NSDictionary<NSString *, NSString *> *> *)messages
                      addAssistantPrefix:(BOOL)addAssistantPrefix;

/// Tokenize `text` and return an array of NSNumber-wrapped token IDs.
- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text addBOS:(BOOL)addBOS;

/// Convert a single token ID back to its string piece.
- (NSString *)tokenToPiece:(int32_t)tokenID NS_SWIFT_NAME(tokenPiece(_:));


// ─ GGML RPC worker server ─────────────────────────────────────────────────────
//
// The phone runs as a pure compute-backend (leaf node) for an external
// llama-cli coordinator.  The coordinator loads the model and drives inference;
// the phone provides Metal GPU compute via the GGML RPC protocol.
//
// Usage on the coordinator machine (after the iOS app shows its endpoint):
//   llama-cli -m model.gguf --rpc 192.168.1.42:50052 -p "your prompt"
//
// Requires the ggml-rpc.xcframework (rebuild with GGML_RPC=ON; see build-ggml-ios.sh).

/// Returns YES if the GGML RPC backend was compiled into this build.
/// Requires ggml-rpc.xcframework linked in the Xcode target.
+ (BOOL)rpcAvailable;

/// Bytes the current process can still allocate before iOS jetsam-kills it.
/// Uses os_proc_available_memory() — more accurate than system-wide free pages
/// because it reflects the per-process limit enforced by the kernel.
+ (NSUInteger)processAvailableMemoryBytes;

/// Start the GGML RPC server.  This method BLOCKS until the server stops;
/// call it from a background thread.
///
/// @param endpoint   "host:port" string, e.g. "0.0.0.0:50052".
/// @param cacheDir   Optional path for tensor cache (pass nil to disable).
/// @param freeMB     Bytes of free memory to advertise to the coordinator (MB).
/// @param totalMB    Total memory to advertise (MB).
- (void)startRPCServer:(NSString *)endpoint
              cacheDir:(nullable NSString *)cacheDir
                freeMB:(NSUInteger)freeMB
               totalMB:(NSUInteger)totalMB
               threads:(NSUInteger)threads;

// ─ Distributed inference stubs ───────────────────────────────────────────────
//
// Pipeline-parallel execution splits the model into contiguous layer ranges.
// Each range is a "shard".  The first shard accepts token IDs; subsequent
// shards accept hidden-state tensors from the previous device.
//
// Layer numbering convention
//   layer  0      – transformer block 0
//   layer  L-1    – transformer block L-1  (L = modelInfo.nLayers)
//
// The embedding table and final layer-norm / unembedding projection are
// implicitly part of the first and last shards respectively.
//
// NOTE: Methods marked [REQUIRES PATCH] need the llama-partial-eval.patch
//       applied to llama.cpp before they can function correctly.
//       Without the patch they fall back to running the full model locally.

/// [FIRST SHARD]  Embed tokens and run transformer layers [0, endLayer).
/// `endLayer` must be <= modelInfo.nLayers.  When endLayer == modelInfo.nLayers
/// the full model runs and the callback receives output logits instead of a
/// hidden state (use `generateFromPrompt:config:callback:` for that case).
///
/// The hidden state delivered to `callback` should be forwarded verbatim to
/// the next device via the network transport.                    [REQUIRES PATCH]
- (void)runFirstShardWithTokens:(NSArray<NSNumber *> *)tokenIDs
                       endLayer:(NSInteger)endLayer
                       callback:(LlamaShardCallback)callback;

/// [MIDDLE / LAST SHARD]  Continue evaluation from an incoming hidden state.
/// `hiddenState` is the raw float32 blob produced by the previous shard.
/// `tokenCount`  is the number of tokens encoded in that blob.
/// `startLayer`  is the first layer this shard should compute.
/// `endLayer`    is one past the last layer this shard should compute.
///               Pass modelInfo.nLayers to include the output head.
/// When `endLayer == modelInfo.nLayers`, the callback receives sampled-token
/// output in `hiddenState` as a single float32 representing the token ID
/// (wrapped in NSData for API uniformity), and `isDone` is YES.  [REQUIRES PATCH]
- (void)runShardWithHiddenState:(NSData *)hiddenState
                     tokenCount:(NSInteger)tokenCount
                     startLayer:(NSInteger)startLayer
                       endLayer:(NSInteger)endLayer
                       callback:(LlamaShardCallback)callback;

/// Returns the number of bytes the OS is willing to give this process before
/// it is killed for exceeding its memory limit (i.e. os_proc_available_memory).
+ (NSUInteger)availableProcessMemoryBytes;

@end

NS_ASSUME_NONNULL_END
