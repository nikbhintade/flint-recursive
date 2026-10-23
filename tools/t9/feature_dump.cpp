// Teacher-forced features for training an EAGLE-3 head on Flint: for every token, the inputs of target layers
// 2, 24, 46 (what llama.cpp's EAGLE-3 reads) and the final normed hidden state (for the target's next-token
// distribution). Runs our engine (static MoE cache via LLAMA_MOE_CACHE_PROFILE).
//
// Input:  a sequence file, one sequence per line: n_prompt, then token ids (tools/t9/features.py writes it).
// Output: <out>.bf16  bf16 [token][4][n_embd]  (layer 2 input, layer 24 input, layer 46 input, final norm)
//         appended in input order; sequences longer than max_len are cut.
//
// Usage: feature_dump model.gguf sequences.txt out_prefix moe_cache_mib [max_len=2048]

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char * k_names[4] = { "l_out-1", "l_out-23", "l_out-45", "result_norm" };

struct capture {
    int n_embd   = 0;
    int n_tokens = 0;
    std::vector<float> rows[4];
    int seen = 0;
};

static bool on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (capture *) user_data;
    int k = -1;
    for (int i = 0; i < 4; ++i) {
        if (strcmp(t->name, k_names[i]) == 0) {
            k = i;
        }
    }
    if (ask) {
        return k >= 0;
    }
    if (k < 0) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != c->n_embd || ggml_nrows(t) != c->n_tokens) {
        fprintf(stderr, "%s: unexpected shape %lld x %lld\n", t->name, (long long) t->ne[0], (long long) ggml_nrows(t));
        exit(1);
    }
    c->rows[k].resize((size_t) c->n_embd*c->n_tokens);
    ggml_backend_tensor_get(t, c->rows[k].data(), 0, ggml_nbytes(t));
    c->seen |= 1 << k;
    return true;
}

static uint16_t to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1); // round to nearest even
    return (uint16_t) (u >> 16);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.gguf sequences.txt out_prefix moe_cache_mib [max_len]\n", argv[0]);
        return 1;
    }
    const int cache_mib = atoi(argv[4]);
    const int max_len   = argc > 5 ? atoi(argv[5]) : 2048;

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

    capture cap;
    cap.n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = max_len + 16;
    cp.n_batch           = 1024;
    cp.n_ubatch          = 1024;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads         = 14;
    cp.n_threads_batch   = 14;
    cp.moe_cache_size    = (size_t) cache_mib*1024*1024;
    cp.cb_eval           = on_tensor;
    cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    FILE * out = fopen((std::string(argv[3]) + ".bf16").c_str(), "wb");
    std::ifstream in(argv[2]);
    std::string line;
    llama_batch batch = llama_batch_init(cp.n_ubatch, 0, 1);
    std::vector<uint16_t> buf;
    size_t total = 0, n_seq = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        size_t n_prompt;
        ss >> n_prompt;
        std::vector<llama_token> seq;
        for (llama_token t; ss >> t && (int) seq.size() < max_len; ) seq.push_back(t);
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t p = 0; p < seq.size(); p += cp.n_ubatch) {
            const int n = (int) std::min((size_t) cp.n_ubatch, seq.size() - p);
            batch.n_tokens = n;
            for (int j = 0; j < n; j++) {
                batch.token[j]     = seq[p + j];
                batch.pos[j]       = (llama_pos) (p + j);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1; // every row reaches the last layer and the final norm
            }
            cap.n_tokens = n;
            cap.seen     = 0;
            if (llama_decode(ctx, batch) != 0 || cap.seen != 15) {
                fprintf(stderr, "decode failed or tensors missing (seen %d) at sequence %zu\n", cap.seen, n_seq);
                return 1;
            }
            buf.resize((size_t) n*4*cap.n_embd);
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < 4; k++) {
                    const float * src = cap.rows[k].data() + (size_t) j*cap.n_embd;
                    uint16_t * dst = buf.data() + ((size_t) j*4 + k)*cap.n_embd;
                    for (int e = 0; e < cap.n_embd; e++) dst[e] = to_bf16(src[e]);
                }
            }
            fwrite(buf.data(), sizeof(uint16_t), buf.size(), out);
            total += n;
        }
        n_seq++;
        fprintf(stderr, "\r%zu sequences, %zu tokens", n_seq, total);
    }
    fprintf(stderr, "\nwrote %zu tokens x 4 x %d\n", total, cap.n_embd);
    fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
