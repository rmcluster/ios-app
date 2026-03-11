// LlamaBridge.mm
//
// Objective-C++ implementation of LlamaBridge.
//
// Build requirements:
//   - llama.cpp XCFrameworks (Frameworks/llama.xcframework, ggml.xcframework …)
//     produced by scripts/build-ggml-ios.sh
//   - Xcode target must link: Metal.framework, Accelerate.framework
//
// The "REQUIRES PATCH" sections below require llama-partial-eval.patch applied
// to llama.cpp to expose intermediate hidden states.  Without the patch those
// code paths fall back to full-model single-device inference so the app stays
// usable for local testing.

#import "LlamaBridge.h"
#import <Metal/Metal.h>
#include <os/proc.h>

// Pull in llama.cpp public API.  The header will be available once the
// XCFramework is added to the target.  Guard it so the file can still
// be syntax-checked without the framework present.
#if __has_include(<llama.h>)
  #include <llama.h>
  #include <ggml.h>
  #include <ggml-opt.h>
  #define LLAMA_AVAILABLE 1
#else
  #define LLAMA_AVAILABLE 0
  #warning "llama.cpp XCFramework not found – run scripts/build-ggml-ios.sh"
#endif

// GGML RPC backend (optional – requires GGML_RPC=ON + ggml-rpc.xcframework).
// See scripts/build-ggml-ios.sh and docs at vendor/llama.cpp/ggml/include/ggml-rpc.h
#if LLAMA_AVAILABLE && __has_include(<ggml-rpc.h>)
  #include <ggml-rpc.h>
  #define GGML_RPC_AVAILABLE 1
#else
  #define GGML_RPC_AVAILABLE 0
#endif

#if GGML_RPC_AVAILABLE
  // Metal backend header (included in ggml-metal.xcframework).
  // Needed to create a Metal compute backend for the RPC server.
  #if __has_include(<ggml-metal.h>)
    #include <ggml-metal.h>
  #endif
#endif

#include <vector>
#include <string>
#include <cstring>

#if LLAMA_AVAILABLE
// llama_batch_add was removed from the public header in b5076.
// Inline the same helper that llama.cpp's common/ layer uses internally.
static inline void llama_batch_add_token(
        struct llama_batch & batch,
              llama_token   id,
              llama_pos     pos,
  const std::vector<llama_seq_id> & seq_ids,
              bool          logits)
{
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = (int32_t)seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}
#endif

// ── Error domain ─────────────────────────────────────────────────────────────
NSString *const LlamaBridgeErrorDomain = @"LlamaBridgeErrorDomain";

typedef NS_ENUM(NSInteger, LlamaBridgeError) {
    LlamaBridgeErrorModelLoadFailed  = 1,
    LlamaBridgeErrorContextFailed    = 2,
    LlamaBridgeErrorDecodeFailed     = 3,
    LlamaBridgeErrorNotLoaded        = 4,
    LlamaBridgeErrorInvalidShard     = 5,
};

// ── LlamaModelInfo ────────────────────────────────────────────────────────────
@implementation LlamaModelInfo
- (instancetype)initWithName:(NSString *)name
                    nLayers:(NSInteger)nLayers
                      nEmbd:(NSInteger)nEmbd
                       nCtx:(NSInteger)nCtx
                     nVocab:(NSInteger)nVocab
                 eosTokenID:(int32_t)eosTokenID
               fileSizeBytes:(NSUInteger)fileSizeBytes {
    if ((self = [super init])) {
        _name          = name;
        _nLayers       = nLayers;
        _nEmbd         = nEmbd;
        _nCtx          = nCtx;
        _nVocab        = nVocab;
        _eosTokenID    = eosTokenID;
        _fileSizeBytes = fileSizeBytes;
    }
    return self;
}
@end

// ── LlamaGenerationConfig ─────────────────────────────────────────────────────
@implementation LlamaGenerationConfig
+ (instancetype)defaults {
    LlamaGenerationConfig *c = [LlamaGenerationConfig new];
    c.maxNewTokens  = 200;
    c.temperature   = 0.8f;
    c.topP          = 0.9f;
    c.repeatPenalty = 1.1f;
    c.seed          = -1;
    return c;
}
@end

// ── LlamaBridge private state ─────────────────────────────────────────────────
@interface LlamaBridge () {
#if LLAMA_AVAILABLE
    llama_model   *_model;
    llama_context *_ctx;
    llama_sampler *_sampler;
#else
    void *_model;
    void *_ctx;
    void *_sampler;
#endif
    LlamaModelInfo *_modelInfo;
    /// KV-cache position tracker for distributed first-shard decode steps.
    NSInteger _shardNPast;
}
@end

@implementation LlamaBridge

// ── Init / dealloc ────────────────────────────────────────────────────────────
- (instancetype)init {
    if ((self = [super init])) {
#if LLAMA_AVAILABLE
        llama_backend_init();
#endif
        _model   = nullptr;
        _ctx     = nullptr;
        _sampler = nullptr;
    }
    return self;
}

- (void)dealloc {
    [self unloadModel];
#if LLAMA_AVAILABLE
    llama_backend_free();
#endif
}

// ── Model lifecycle ────────────────────────────────────────────────────────────
- (BOOL)loadModelFromPath:(NSString *)path
                     nCtx:(NSInteger)nCtx
                    error:(NSError **)error {
    [self unloadModel];

#if !LLAMA_AVAILABLE
    if (error) {
        *error = [NSError errorWithDomain:LlamaBridgeErrorDomain
                                     code:LlamaBridgeErrorModelLoadFailed
                                 userInfo:@{NSLocalizedDescriptionKey:
                                            @"llama.cpp framework not linked"}];
    }
    return NO;
#else
    // ── load model ───────────────────────────────────────────────────────────
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;   // offload all layers to Metal on device

    _model = llama_model_load_from_file(path.UTF8String, model_params);
    if (!_model) {
        if (error) {
            *error = [NSError errorWithDomain:LlamaBridgeErrorDomain
                                         code:LlamaBridgeErrorModelLoadFailed
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                [NSString stringWithFormat:
                                                 @"Failed to load model at %@", path]}];
        }
        return NO;
    }

    // ── create inference context ──────────────────────────────────────────────
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = (uint32_t)nCtx;
    ctx_params.n_batch  = 512;
    ctx_params.n_ubatch = 512;

    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx) {
        llama_model_free(_model);
        _model = nullptr;
        if (error) {
            *error = [NSError errorWithDomain:LlamaBridgeErrorDomain
                                         code:LlamaBridgeErrorContextFailed
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                @"Failed to create llama context"}];
        }
        return NO;
    }

    // ── default sampler chain ─────────────────────────────────────────────────
    // Will be reconfigured per-generation from LlamaGenerationConfig.
    // Processing samplers first, dist (discrete selector) last.
    _sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ── populate model info ────────────────────────────────────────────────────
    NSString *modelName = @"unknown";
    {
        char buf[256] = {0};
        if (llama_model_meta_val_str(_model, "general.name", buf, sizeof(buf)) > 0) {
            modelName = [NSString stringWithUTF8String:buf];
        }
    }

    NSDictionary *attrs = [[[NSFileManager defaultManager]
                             attributesOfItemAtPath:path error:nil] copy];
    NSUInteger fileSize = [attrs[NSFileSize] unsignedIntegerValue];

    _shardNPast = 0;
    _modelInfo = [[LlamaModelInfo alloc]
                  initWithName:modelName
                  nLayers:(NSInteger)llama_model_n_layer(_model)
                  nEmbd:(NSInteger)llama_model_n_embd(_model)
                  nCtx:(NSInteger)llama_n_ctx(_ctx)
                  nVocab:(NSInteger)llama_vocab_n_tokens(llama_model_get_vocab(_model))
                  eosTokenID:(int32_t)llama_vocab_eos(llama_model_get_vocab(_model))
                  fileSizeBytes:fileSize];

    return YES;
#endif
}

- (void)unloadModel {
#if LLAMA_AVAILABLE
    if (_sampler) { llama_sampler_free(_sampler); _sampler = nullptr; }
    if (_ctx)     { llama_free(_ctx);              _ctx     = nullptr; }
    if (_model)   { llama_model_free(_model);      _model   = nullptr; }
#endif
    _modelInfo = nil;
}

- (BOOL)isModelLoaded {
    return _model != nullptr && _ctx != nullptr;
}

- (LlamaModelInfo *)modelInfo {
    return _modelInfo;
}

// ── Chat template ─────────────────────────────────────────────────────────────
- (nullable NSString *)applyChatTemplate:(NSArray<NSDictionary<NSString *, NSString *> *> *)messages
                      addAssistantPrefix:(BOOL)addAssistantPrefix {
#if !LLAMA_AVAILABLE
    return nil;
#else
    if (!_model) return nil;

    // Keep strings alive for the duration of the C call.
    std::vector<std::string> roles, contents;
    std::vector<llama_chat_message> msgs;
    roles.reserve(messages.count);
    contents.reserve(messages.count);
    msgs.reserve(messages.count);

    for (NSDictionary<NSString *, NSString *> *msg in messages) {
        roles.push_back(std::string(msg[@"role"].UTF8String ?: "user"));
        contents.push_back(std::string(msg[@"content"].UTF8String ?: ""));
        msgs.push_back({ roles.back().c_str(), contents.back().c_str() });
    }

    const char *tmpl = llama_model_chat_template(_model, /*name=*/nullptr);
    std::vector<char> buf(4096);
    int32_t n = llama_chat_apply_template(tmpl,
                                          msgs.data(), msgs.size(),
                                          (bool)addAssistantPrefix,
                                          buf.data(), (int32_t)buf.size());
    if (n < 0) return nil;
    if (n > (int32_t)buf.size()) {
        buf.resize((size_t)n + 1);
        n = llama_chat_apply_template(tmpl,
                                      msgs.data(), msgs.size(),
                                      (bool)addAssistantPrefix,
                                      buf.data(), (int32_t)buf.size());
        if (n < 0) return nil;
    }
    return [[NSString alloc] initWithBytes:buf.data()
                                    length:(NSUInteger)n
                                  encoding:NSUTF8StringEncoding];
#endif
}

// ── Tokenization ──────────────────────────────────────────────────────────────
- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text addBOS:(BOOL)addBOS {
#if !LLAMA_AVAILABLE
    return @[];
#else
    if (!_model) return @[];
    const char *cstr = text.UTF8String;
    int nMax = (int)(strlen(cstr) + 10);
    std::vector<llama_token> tokens(nMax);
    int n = llama_tokenize(llama_model_get_vocab(_model),
                           cstr, (int32_t)strlen(cstr),
                           tokens.data(), nMax,
                           addBOS, /*special=*/false);
    if (n < 0) return @[];
    tokens.resize(n);
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:n];
    for (int i = 0; i < n; i++) {
        [result addObject:@(tokens[i])];
    }
    return result;
#endif
}

- (NSString *)tokenToPiece:(int32_t)tokenID {
#if !LLAMA_AVAILABLE
    return @"";
#else
    if (!_model) return @"";
    char buf[128] = {0};
    llama_token_to_piece(llama_model_get_vocab(_model), tokenID, buf, sizeof(buf), 0, false);
    return [NSString stringWithUTF8String:buf] ?: @"";
#endif
}

// ── Single-device full-model generation ───────────────────────────────────────
- (void)generateFromPrompt:(NSString *)prompt
                    config:(LlamaGenerationConfig *)config
                  callback:(LlamaTokenCallback)callback {
#if !LLAMA_AVAILABLE
    callback(@"[llama.cpp not available]", YES);
    return;
#else
    if (!self.isModelLoaded) {
        callback(@"[model not loaded]", YES);
        return;
    }

    // ── reconfigure sampler ───────────────────────────────────────────────────
    // Order matters: processing samplers (penalties, temp, top_p) must come
    // before the discrete selector (dist).  dist MUST be last — it sets
    // cur_p.selected; any sampler that runs after it resets selected to -1,
    // which triggers GGML_ASSERT(cur_p.selected >= 0) in llama_sampler_sample.
    llama_sampler_free(_sampler);
    _sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(_sampler,
        llama_sampler_init_penalties((int32_t)llama_n_ctx(_ctx),
                                     config.repeatPenalty,
                                     0.0f, 0.0f));
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(config.topP, 1));
    // dist must be last — it selects the token
    uint32_t seed = config.seed >= 0 ? (uint32_t)config.seed : LLAMA_DEFAULT_SEED;
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(seed));

    // ── tokenize prompt ───────────────────────────────────────────────────────
    std::vector<llama_token> promptTokens;
    {
        const char *cstr = prompt.UTF8String;
        int n = (int)(strlen(cstr) + 10);
        promptTokens.resize(n);
        int actual = llama_tokenize(llama_model_get_vocab(_model), cstr, (int32_t)strlen(cstr),
                                    promptTokens.data(), n,
                                    /*add_special=*/true, /*parse_special=*/false);
        if (actual < 0) {
            callback(@"[tokenization failed]", YES);
            return;
        }
        promptTokens.resize(actual);
    }

    // ── prefill (batch decode prompt) ─────────────────────────────────────────
    llama_memory_clear(llama_get_memory(_ctx), false);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < (int)promptTokens.size(); i++) {
        llama_batch_add_token(batch, promptTokens[i], i, {0}, /*logits=*/false);
    }
    // We need logits for the last token to start sampling
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
    }

    if (llama_decode(_ctx, batch) != 0) {
        llama_batch_free(batch);
        callback(@"[prefill decode failed]", YES);
        return;
    }
    llama_batch_free(batch);

    // ── autoregressive decode loop ────────────────────────────────────────────
    int nPast = (int)promptTokens.size();
    const int32_t eosToken = llama_vocab_eos(llama_model_get_vocab(_model));

    for (int i = 0; i < config.maxNewTokens; i++) {
        // Sample next token.
        // llama_sampler_sample() already calls llama_sampler_accept() internally;
        // do NOT call it again here — that would double-accept and corrupt the
        // repetition-penalty history.
        llama_token newToken = llama_sampler_sample(_sampler, _ctx, -1);

        BOOL done = (newToken == eosToken);

        // Decode token to string piece
        char piece[128] = {0};
        llama_token_to_piece(llama_model_get_vocab(_model), newToken, piece, sizeof(piece), 0, false);
        NSString *tokenStr = [NSString stringWithUTF8String:piece] ?: @"";
        callback(tokenStr, done);

        if (done) break;

        // Feed token back
        llama_batch next = llama_batch_init(1, 0, 1);
        llama_batch_add_token(next, newToken, nPast, {0}, /*logits=*/true);
        if (llama_decode(_ctx, next) != 0) {
            llama_batch_free(next);
            callback(@"", YES);
            break;
        }
        llama_batch_free(next);
        nPast++;
    }
    // If the loop exited because maxNewTokens was reached (no EOS and no decode
    // failure), the callback was never called with done=YES.  The Swift
    // AsyncStream continuation would hang open forever waiting for more tokens.
    // Signal completion unconditionally here — calling it a second time after an
    // EOS/decode-failure break is safe because continuation.finish() is a no-op
    // on an already-finished stream.
    callback(@"", YES);
#endif
}

// ── GGML RPC server ──────────────────────────────────────────────────────────

+ (BOOL)rpcAvailable {
    return GGML_RPC_AVAILABLE;
}

+ (NSUInteger)processAvailableMemoryBytes {
    return (NSUInteger)os_proc_available_memory();
}

- (void)startRPCServer:(NSString *)endpoint
              cacheDir:(nullable NSString *)cacheDir
                freeMB:(NSUInteger)freeMB
               totalMB:(NSUInteger)totalMB
               threads:(NSUInteger)threads {
#if !GGML_RPC_AVAILABLE
    NSLog(@"[LlamaBridge] RPC server not available – rebuild with GGML_RPC=ON (see scripts/build-ggml-ios.sh)");
    return;
#else
    // Look up the device handle directly — do NOT call ggml_backend_*_init() here.
    // ggml_backend_rpc_start_server() calls ggml_backend_dev_init() internally, so
    // calling init() ourselves first and then freeing causes a double-init: the
    // second Metal init fails to allocate its buffer pool and returns null, which
    // the RPC server then dereferences → EXC_BAD_ACCESS at 0x10.
    ggml_backend_dev_t dev = nullptr;
#if !TARGET_OS_SIMULATOR && __has_include(<ggml-metal.h>)
    // Use the GGML function which runs the check inside ggml-metal-device.m
    // where Metal.h is properly set up.  The inline MTLCreateSystemDefaultDevice()
    // check done from a Swift detached-task background thread was unreliable on
    // some devices (e.g. iPhone 13 / A15 incorrectly returning false).
    if (ggml_backend_metal_has_simdgroup_reduction()) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    } else {
        NSLog(@"[LlamaBridge] GPU lacks simdgroup reduction (pre-A15). Using CPU backend for RPC server.");
    }
#endif
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        NSLog(@"[LlamaBridge] No GGML backend device found, cannot start RPC server.");
        return;
    }

    const char *ep   = endpoint.UTF8String;
    const char *cdir = cacheDir ? cacheDir.UTF8String : nullptr;

    NSLog(@"[LlamaBridge] Starting GGML RPC server at %@ with %lu threads…", endpoint, (unsigned long)threads);
    // Blocks until the server is stopped externally (process kill or socket close).
    ggml_backend_rpc_start_server(ep, cdir, (size_t)threads, 1, &dev);
    NSLog(@"[LlamaBridge] GGML RPC server stopped.");
#endif
}

// ── Distributed inference – FIRST SHARD ─────────────────────────────────────
//
// CURRENT BEHAVIOR (without patch):
//   Runs the full model and streams tokens via the callback, ignoring endLayer.
//   The hiddenState output is a zero-length NSData so the caller can detect
//   the fallback path.
//
// PATCHED BEHAVIOR (with llama-partial-eval.patch):
//   Runs only layers [0, endLayer) and returns the hidden-state activation
//   tensor for forwarding to the next shard device.
- (void)runFirstShardWithTokens:(NSArray<NSNumber *> *)tokenIDs
                       endLayer:(NSInteger)endLayer
                       callback:(LlamaShardCallback)callback {
#if !LLAMA_AVAILABLE
    callback([NSData data], 0, 0, YES);
    return;
#else
    if (!self.isModelLoaded) {
        callback([NSData data], 0, 0, YES);
        return;
    }

    NSInteger totalLayers = _modelInfo.nLayers;
    NSInteger nEmbd       = _modelInfo.nEmbd;

    // Clamp endLayer
    if (endLayer <= 0 || endLayer > totalLayers) {
        endLayer = totalLayers;
    }

    // Build token vector
    std::vector<llama_token> tokens;
    for (NSNumber *t in tokenIDs) {
        tokens.push_back((llama_token)t.intValue);
    }
    if (tokens.empty()) {
        callback([NSData data], 0, (NSInteger)nEmbd, YES);
        return;
    }

    // ── [PATCH HOOK] llama_decode_partial would be called here ───────────────
    // The proposed C API addition to llama.cpp:
    //
    //   int llama_decode_partial(
    //       struct llama_context * ctx,
    //       struct llama_batch     batch,
    //       int32_t                layer_start,   // 0 = include embedding
    //       int32_t                layer_end,     // exclusive
    //       float                * hidden_out     // [n_tokens × n_embd]
    //   );
    //
    // Until that is available, fall through to full-model decode.

    if (endLayer < totalLayers) {
        // ── Full-forward-pass proxy for partial sharding ──────────────────────
        // llama_decode_partial() does not exist yet, so we run the FULL model
        // and forward raw vocabulary logits (n_vocab floats) as the "hidden
        // state" to the next device.  The last-shard device calls
        // runShardWithHiddenState: which interprets the blob as logits, applies
        // its own sampler, and returns the sampled token.
        //
        // All real transformer computation therefore happens on the first-shard
        // (coordinator) device; subsequent shards only apply sampling.  This is
        // semantically correct, fully exercises the network protocol, and can be
        // upgraded to true layer-parallel sharding once the patch lands.

        NSInteger nVocab   = _modelInfo.nVocab;
        BOOL      isPrefill = ((NSInteger)tokens.size() > 1);

        if (isPrefill) {
            // New generation session: clear KV cache and reset position counter.
            llama_memory_clear(llama_get_memory(_ctx), false);
            _shardNPast = 0;
        }

        // Build batch; only the last token needs logits computed.
        llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
        for (int i = 0; i < (int)tokens.size(); i++) {
            BOOL wantsLogits = (i == (int)tokens.size() - 1);
            llama_batch_add_token(batch, tokens[i], (llama_pos)(_shardNPast + i), {0}, wantsLogits);
        }
        int rc = llama_decode(_ctx, batch);
        llama_batch_free(batch);
        _shardNPast += (int)tokens.size();

        if (rc != 0) {
            callback([NSData data], 0, nVocab, YES);
            return;
        }

        // Extract raw logits for the last token and ship them as the payload.
        // nEmbd is repurposed to carry n_vocab so the receiver knows how many
        // floats to expect.
        float *logits = llama_get_logits_ith(_ctx, -1);
        NSData *logitData = [NSData dataWithBytes:logits
                                           length:(size_t)(nVocab * sizeof(float))];
        callback(logitData, (NSInteger)tokens.size(), nVocab, YES);
    } else {
        // endLayer == totalLayers → run full model, stream tokens
        // (used when this device is the only shard)
        llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
        for (int i = 0; i < (int)tokens.size(); i++) {
            llama_batch_add_token(batch, tokens[i], i, {0}, (i == (int)tokens.size()-1));
        }
        llama_decode(_ctx, batch);
        llama_batch_free(batch);

        // Sample and return the single next token's "hidden state" as its ID
        llama_token next = llama_sampler_sample(_sampler, _ctx, -1);
        float tokenIDFloat = (float)next;
        NSData *data = [NSData dataWithBytes:&tokenIDFloat length:sizeof(float)];
        callback(data, 1, nEmbd, YES);
    }
#endif
}

// ── Distributed inference – MIDDLE / LAST SHARD ──────────────────────────────
- (void)runShardWithHiddenState:(NSData *)hiddenState
                     tokenCount:(NSInteger)tokenCount
                     startLayer:(NSInteger)startLayer
                       endLayer:(NSInteger)endLayer
                       callback:(LlamaShardCallback)callback {
#if !LLAMA_AVAILABLE
    callback([NSData data], 0, 0, YES);
    return;
#else
    if (!self.isModelLoaded) {
        callback([NSData data], 0, 0, YES);
        return;
    }

    NSInteger totalLayers = _modelInfo.nLayers;
    NSInteger nEmbd       = _modelInfo.nEmbd;
    BOOL isLastShard      = (endLayer >= totalLayers);

    if (startLayer >= totalLayers || startLayer < 0) {
        callback([NSData data], 0, nEmbd, YES);
        return;
    }

    if (isLastShard) {
        // ── Sample from received logits ───────────────────────────────────────
        // The first-shard device ran the full forward pass and sent us n_vocab
        // raw float32 logits.  Apply the local sampler chain and return the
        // selected token ID packed as a single float32.
        //
        // nEmbd in this call carries n_vocab (set by the first-shard sender).
        NSInteger nVocab = (NSInteger)(hiddenState.length / sizeof(float));
        if (nVocab <= 0 || !_sampler) {
            float zero = 0.0f;
            callback([NSData dataWithBytes:&zero length:sizeof(float)], 1, nEmbd, YES);
            return;
        }

        const float *logitsPtr = (const float *)hiddenState.bytes;
        std::vector<llama_token_data> candidates((size_t)nVocab);
        for (int32_t i = 0; i < (int32_t)nVocab; i++) {
            candidates[i] = { i, logitsPtr[i], 0.0f };
        }

        llama_token_data_array curP;
        curP.data     = candidates.data();
        curP.size     = (size_t)nVocab;
        curP.selected = -1;
        curP.sorted   = false;

        llama_sampler_apply(_sampler, &curP);

        llama_token sampledToken;
        if (curP.selected >= 0 && curP.selected < (int32_t)candidates.size()) {
            sampledToken = candidates[curP.selected].id;
        } else {
            // Fallback: greedy argmax
            sampledToken = 0;
            float bestLogit = logitsPtr[0];
            for (int i = 1; i < (int)nVocab; i++) {
                if (logitsPtr[i] > bestLogit) { bestLogit = logitsPtr[i]; sampledToken = (llama_token)i; }
            }
        }
        llama_sampler_accept(_sampler, sampledToken);

        float tokenIDFloat = (float)sampledToken;
        NSData *out = [NSData dataWithBytes:&tokenIDFloat length:sizeof(float)];
        callback(out, 1, nEmbd, YES);
    } else {
        // ── Middle shard: pass activation through unchanged ───────────────────
        // In the current proxy architecture the first shard sends raw logits
        // directly to the last shard, so middle shards just forward the payload.
        // When true partial-layer sharding is added, this becomes an actual
        // hidden-state transformation.
        NSMutableData *outState = [NSMutableData dataWithLength:hiddenState.length];
        memcpy(outState.mutableBytes, hiddenState.bytes, hiddenState.length);
        callback([outState copy], tokenCount, nEmbd, YES);
    }
#endif
}

+ (NSUInteger)availableProcessMemoryBytes {
    return (NSUInteger)os_proc_available_memory();
}

@end
