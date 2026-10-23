// Time one decode step of T tokens (the cost of verifying a T-1 token draft) after a prompt, for several T.
// Each step is rolled back from the KV cache afterwards, so every measurement starts from the same state.
//
// Usage: batch_bench model.gguf moe_cache_mib [reps=20]   (expert weights in RAM, -ot ...exps.weight=CPU style)
// The static MoE cache is enabled with LLAMA_MOE_CACHE_PROFILE, as in llama-server.

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf moe_cache_mib [reps]\n", argv[0]);
        return 1;
    }
    const int cache_mib = atoi(argv[2]);
    const int reps      = argc > 3 ? atoi(argv[3]) : 20;

    llama_backend_init();
    llama_model_tensor_buft_override overrides[] = {
        { "blk\\.[0-9]+\\.ffn_(up|down|gate)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers          = 999;
    mp.tensor_buft_overrides = overrides;
    mp.load_mode             = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 4096;
    cp.n_batch         = 1024;
    cp.n_ubatch        = 1024;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = 14;
    cp.n_threads_batch = 14;
    cp.moe_cache_size  = (size_t) cache_mib*1024*1024;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    // a code-like prompt of ~500 tokens
    std::string text;
    for (int i = 0; i < 60; i++) {
        text += "def f" + std::to_string(i) + "(x, y):\n    return x * " + std::to_string(i) + " + y\n";
    }
    std::vector<llama_token> toks(text.size() + 8);
    const int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    toks.resize(n);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size()))) return 1;
    llama_synchronize(ctx);

    printf("prompt %d tokens; one decode step of T tokens, median of %d:\n", n, reps);
    double t1 = 0.0;
    for (int T : { 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32 }) {
        std::vector<double> ms;
        for (int r = 0; r < reps + 2; r++) {
            std::vector<llama_token> step(toks.begin() + 7 + r % 50, toks.begin() + 7 + r % 50 + T);
            const auto t0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, llama_batch_get_one(step.data(), T))) return 1;
            llama_synchronize(ctx);
            const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            llama_memory_seq_rm(llama_get_memory(ctx), 0, n, -1);
            if (r >= 2) ms.push_back(dt); // skip warm-up
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        if (T == 1) {
            t1 = med;
        }
        // a step of T tokens pays off if it yields more than med/t1 tokens: 1 + accepted drafts
        printf("  T=%2d  %6.2f ms  %5.2fx one token; needs > %.1f of %d drafts accepted\n", T, med, med/t1, med/t1 - 1.0, T - 1);
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
