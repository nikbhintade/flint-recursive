// Teacher-forced logits for engine comparisons: feed fixed token sequences and save the log-probabilities of every
// generated position, decoding those positions one token at a time like generation does.
//
// Input:  the route_dump sequence file (per line: n_prompt, then token ids). The prompt is processed in one batch.
// Output: <out>.lp   float32 [position][n_vocab] log-softmax for every position after the prompt (up to max_gen)
//         <out>.tok  int32 next token at each saved position (the teacher token)
//
// Usage: logit_dump model.gguf sequences.txt out_prefix n_seqs max_gen n_cpu_moe moe_cache_mib
//        (n_cpu_moe 49 puts all experts in RAM; moe_cache_mib 0 disables the MoE cache)
// The static cache is enabled by the LLAMA_MOE_CACHE_PROFILE environment variable, as in llama-server.

#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s model.gguf sequences.txt out_prefix n_seqs max_gen n_cpu_moe moe_cache_mib\n", argv[0]);
        return 1;
    }
    const int n_seqs    = atoi(argv[4]);
    const int max_gen   = atoi(argv[5]);
    const int n_cpu_moe = atoi(argv[6]);
    const int cache_mib = atoi(argv[7]);

    std::vector<std::vector<llama_token>> seqs;
    std::vector<size_t> n_prompts;
    {
        std::ifstream in(argv[2]);
        std::string line;
        while ((int) seqs.size() < n_seqs && std::getline(in, line)) {
            std::istringstream ss(line);
            size_t np;
            ss >> np;
            std::vector<llama_token> s;
            for (llama_token t; ss >> t; ) s.push_back(t);
            seqs.push_back(s);
            n_prompts.push_back(np);
        }
    }

    llama_backend_init();
    std::string pattern = "blk\\.(";
    for (int il = 0; il < n_cpu_moe; il++) pattern += (il ? "|" : "") + std::to_string(il);
    pattern += ")\\.ffn_(up|down|gate)_exps";
    llama_model_tensor_buft_override overrides[] = { { pattern.c_str(), ggml_backend_cpu_buffer_type() }, { nullptr, nullptr } };

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers          = 999;
    mp.tensor_buft_overrides = n_cpu_moe > 0 ? overrides : nullptr;
    mp.load_mode             = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 16384;
    cp.n_batch         = 1024;
    cp.n_ubatch        = 1024;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = 14;
    cp.n_threads_batch = 14;
    cp.moe_cache_size  = (size_t) cache_mib*1024*1024;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    FILE * f_lp  = fopen((std::string(argv[3]) + ".lp").c_str(), "wb");
    FILE * f_tok = fopen((std::string(argv[3]) + ".tok").c_str(), "wb");
    std::vector<float> lp(n_vocab);
    size_t total = 0;
    for (size_t s = 0; s < seqs.size(); s++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const auto & seq = seqs[s];
        const size_t np = n_prompts[s];
        const size_t end = std::min(seq.size() - 1, np + max_gen);
        for (size_t p = 0; p < np; p += cp.n_batch) {
            const int n = (int) std::min((size_t) cp.n_batch, np - p);
            if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(seq.data()) + p, n))) return 1;
        }
        // the last prompt token predicts seq[np]; then one generated token at a time
        for (size_t p = np; ; p++) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            double mx = logits[0];
            for (int i = 1; i < n_vocab; i++) mx = std::max(mx, (double) logits[i]);
            double sum = 0;
            for (int i = 0; i < n_vocab; i++) sum += std::exp(logits[i] - mx);
            const double lse = mx + std::log(sum);
            for (int i = 0; i < n_vocab; i++) lp[i] = (float) (logits[i] - lse);
            fwrite(lp.data(), sizeof(float), n_vocab, f_lp);
            const int32_t next = seq[p];
            fwrite(&next, sizeof(int32_t), 1, f_tok);
            total++;
            if (p >= end) break;
            if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(seq.data()) + p, 1))) return 1;
        }
        fprintf(stderr, "\r[%zu/%zu] %zu positions", s + 1, seqs.size(), total);
    }
    fprintf(stderr, "\nwrote %zu positions x %d vocab\n", total, n_vocab);
    fclose(f_lp);
    fclose(f_tok);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
