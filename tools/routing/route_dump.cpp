// Record Flint's expert routing: replay token sequences through llama.cpp and save, for every token and every
// MoE layer, the 6 selected expert ids and their router scores.
//
// Input:  a text file, one sequence per line: n_prompt, then the space-separated token ids (written by replay.py).
//         The first n_prompt tokens are processed in ubatches like a prompt, the rest one token at a time like
//         generation, so the arithmetic (and so near-tie routing decisions) matches a real run.
// Output: <out>.ids  uint8  [token][moe_layer][6]   (sequences concatenated in input order)
//         <out>.w    float  [token][moe_layer][6]   raw router scores of the selected experts (before normalization)
//
// Replaying a generated answer as a prompt gives the same routing as generating it (causal model, same weights);
// only floating-point rounding differs, which can flip near-tied experts.
//
// Build: see tools/routing/README.md.  Usage: route_dump model.gguf sequences.txt out_prefix [n_cpu_moe] [n_ubatch]

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int K = 6;

struct capture {
    int n_layer_dense = 0;
    int n_moe         = 0;
    int n_tokens      = 0;                // tokens in the current ubatch
    std::vector<uint8_t> ids;             // [token][moe_layer][K]
    std::vector<float>   w;               // [token][moe_layer][K]
    std::vector<uint8_t> seen;            // per moe_layer: 1 = ids captured, 2 = weights captured
    std::vector<uint8_t> scratch;
};

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + n, &end, 10);
    return (end && *end == '\0' && end != name + n) ? (int) il : -1;
}

static bool on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (capture *) user_data;
    const int il_ids = layer_of(t->name, "ffn_moe_topk-");
    const int il_w   = layer_of(t->name, "ffn_moe_weights-");
    if (ask) {
        return il_ids >= 0 || il_w >= 0;
    }
    const int il = il_ids >= 0 ? il_ids : il_w;
    if (il < 0) {
        return true;
    }
    const int ml = il - c->n_layer_dense;
    if (ml < 0 || ml >= c->n_moe) {
        fprintf(stderr, "unexpected MoE layer %d in %s\n", il, t->name);
        exit(1);
    }
    // Tensors can be strided views (top-k is a view of an argsort): copy the whole span once, then pick.
    const size_t span = ggml_nbytes(t);
    c->scratch.resize(span);
    ggml_backend_tensor_get(t, c->scratch.data(), 0, span);

    if (il_ids >= 0) {  // int32 [K, n_tokens]
        if (t->type != GGML_TYPE_I32 || t->ne[0] != K || t->ne[1] != c->n_tokens) {
            fprintf(stderr, "%s: unexpected shape %lld x %lld\n", t->name, (long long) t->ne[0], (long long) t->ne[1]);
            exit(1);
        }
        for (int j = 0; j < c->n_tokens; j++) {
            for (int k = 0; k < K; k++) {
                const int32_t e = *(const int32_t *) (c->scratch.data() + j * t->nb[1] + k * t->nb[0]);
                c->ids[((size_t) j * c->n_moe + ml) * K + k] = (uint8_t) e;
            }
        }
        c->seen[ml] |= 1;
    } else {            // float [1, K, n_tokens]
        if (t->type != GGML_TYPE_F32 || t->ne[1] != K || t->ne[2] != c->n_tokens) {
            fprintf(stderr, "%s: unexpected shape\n", t->name);
            exit(1);
        }
        for (int j = 0; j < c->n_tokens; j++) {
            for (int k = 0; k < K; k++) {
                c->w[((size_t) j * c->n_moe + ml) * K + k] =
                    *(const float *) (c->scratch.data() + j * t->nb[2] + k * t->nb[1]);
            }
        }
        c->seen[ml] |= 2;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf sequences.txt out_prefix [n_cpu_moe=22] [n_ubatch=2048]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * seq_path   = argv[2];
    const std::string out   = argv[3];
    const int n_cpu_moe     = argc > 4 ? atoi(argv[4]) : 22;
    const int n_ubatch      = argc > 5 ? atoi(argv[5]) : 2048;

    std::vector<std::vector<llama_token>> seqs;
    std::vector<size_t> n_prompts;
    size_t max_len = 0;
    {
        std::ifstream in(seq_path);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            size_t n_prompt = 0;
            ss >> n_prompt;
            std::vector<llama_token> s;
            llama_token tok;
            while (ss >> tok) {
                s.push_back(tok);
            }
            if (!s.empty()) {
                max_len = std::max(max_len, s.size());
                seqs.push_back(std::move(s));
                n_prompts.push_back(n_prompt);
            }
        }
    }
    fprintf(stderr, "%zu sequences, longest %zu tokens\n", seqs.size(), max_len);

    llama_backend_init();

    // Experts of the first n_cpu_moe layers stay in RAM, like --n-cpu-moe.
    std::string pattern = "blk\\.(";
    for (int il = 0; il < n_cpu_moe; il++) {
        pattern += (il ? "|" : "") + std::to_string(il);
    }
    pattern += ")\\.ffn_(up|down|gate)_exps";
    llama_model_tensor_buft_override overrides[] = {
        { pattern.c_str(), ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers          = 999;
    mp.tensor_buft_overrides = overrides;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    capture cap;
    // Flint: 2 dense layers, then MoE layers (layer count from the model).
    const int n_layer   = llama_model_n_layer(model);
    cap.n_layer_dense   = 2;
    cap.n_moe           = n_layer - cap.n_layer_dense;
    cap.ids.resize((size_t) n_ubatch * cap.n_moe * K);
    cap.w.resize((size_t) n_ubatch * cap.n_moe * K);
    cap.seen.resize(cap.n_moe);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = (uint32_t) (max_len + 16);
    cp.n_batch           = n_ubatch;
    cp.n_ubatch          = n_ubatch;
    cp.n_seq_max         = 1;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval           = on_tensor;
    cp.cb_eval_user_data = &cap;
    cp.n_threads         = 14;
    cp.n_threads_batch   = 14;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    FILE * f_ids = fopen((out + ".ids").c_str(), "wb");
    FILE * f_w   = fopen((out + ".w").c_str(), "wb");
    // Every token is an output: the last layer drops non-output rows before its MoE, so routing for all tokens
    // needs all outputs.
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);
    size_t total = 0;
    for (size_t s = 0; s < seqs.size(); s++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        auto & seq = seqs[s];
        for (size_t p = 0, n = 0; p < seq.size(); p += n) {
            // prompt: ubatches that stop at the prompt's end; generated part: one token per decode
            n = p < n_prompts[s] ? std::min((size_t) n_ubatch, n_prompts[s] - p) : 1;
            cap.n_tokens = n;
            std::fill(cap.seen.begin(), cap.seen.end(), 0);
            batch.n_tokens = n;
            for (int j = 0; j < n; j++) {
                batch.token[j]     = seq[p + j];
                batch.pos[j]       = (llama_pos) (p + j);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at sequence %zu, position %zu\n", s, p);
                return 1;
            }
            for (int ml = 0; ml < cap.n_moe; ml++) {
                if (cap.seen[ml] != 3) {
                    fprintf(stderr, "MoE layer %d not captured (flags %d)\n", ml + cap.n_layer_dense, cap.seen[ml]);
                    return 1;
                }
            }
            fwrite(cap.ids.data(), 1, (size_t) n * cap.n_moe * K, f_ids);
            fwrite(cap.w.data(), sizeof(float), (size_t) n * cap.n_moe * K, f_w);
            total += n;
        }
        fprintf(stderr, "\r[%zu/%zu] %zu tokens", s + 1, seqs.size(), total);
    }
    fprintf(stderr, "\nwrote %zu tokens x %d MoE layers x %d experts\n", total, cap.n_moe, K);
    fclose(f_ids);
    fclose(f_w);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
