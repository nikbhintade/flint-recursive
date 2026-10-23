// Time Flint's CPU expert work in isolation: one MoE layer (gate, up, down for 6 of 256 experts, one token) in
// NVFP4 on the CPU, cycling over several layers' worth of weights so the selected experts come from RAM, not cache.
//
// Usage: cpu_moe_bench <n_threads> [iterations=400] [layers=8]   (pin with taskset to choose the cores)
// Prints microseconds per layer and the rate at which expert weights are streamed (GB/s).

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static const int N_EMBD = 2048, N_FF = 512, N_EXP = 256, N_USED = 6;

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? atoi(argv[1]) : 14;
    const int iters     = argc > 2 ? atoi(argv[2]) : 400;
    const int n_layers  = argc > 3 ? atoi(argv[3]) : 8;
    const ggml_type wt  = GGML_TYPE_NVFP4;

    const size_t mat_bytes = ggml_row_size(wt, N_EMBD) * N_FF * N_EXP;  // one [2048 x 512 x 256] tensor
    ggml_init_params ip = { (size_t) n_layers * 3 * mat_bytes + 512ull * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // Random weights, quantized once per tensor shape and copied to every layer (content does not affect speed).
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> src((size_t) N_EMBD * N_FF);
    for (auto & v : src) v = nd(rng);
    std::vector<uint8_t> q_gate(ggml_row_size(wt, N_EMBD) * N_FF), q_down(ggml_row_size(wt, N_FF) * N_EMBD);
    ggml_quantize_chunk(wt, src.data(), q_gate.data(), 0, N_FF, N_EMBD, nullptr);
    ggml_quantize_chunk(wt, src.data(), q_down.data(), 0, N_EMBD, N_FF, nullptr);

    struct layer { ggml_tensor * ids; ggml_cgraph * gf; };
    std::vector<layer> layers;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, 1);
    for (int64_t i = 0; i < N_EMBD; i++) ((float *) x->data)[i] = nd(rng) * 50;
    for (int l = 0; l < n_layers; l++) {
        ggml_tensor * gate = ggml_new_tensor_3d(ctx, wt, N_EMBD, N_FF, N_EXP);
        ggml_tensor * up   = ggml_new_tensor_3d(ctx, wt, N_EMBD, N_FF, N_EXP);
        ggml_tensor * down = ggml_new_tensor_3d(ctx, wt, N_FF, N_EMBD, N_EXP);
        for (int e = 0; e < N_EXP; e++) {
            memcpy((char *) gate->data + e * gate->nb[2], q_gate.data(), q_gate.size());
            memcpy((char *) up->data   + e * up->nb[2],   q_gate.data(), q_gate.size());
            memcpy((char *) down->data + e * down->nb[2], q_down.data(), q_down.size());
        }
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);
        ggml_tensor * g = ggml_mul_mat_id(ctx, gate, x, ids);           // [512, 6, 1]
        ggml_tensor * u = ggml_mul_mat_id(ctx, up, x, ids);
        ggml_tensor * h = ggml_mul(ctx, ggml_silu(ctx, g), u);
        ggml_tensor * d = ggml_mul_mat_id(ctx, down, h, ids);           // [2048, 6, 1]
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, d);
        layers.push_back({ ids, gf });
    }

    std::uniform_int_distribution<int> pick(0, N_EXP - 1);
    auto set_ids = [&](ggml_tensor * ids) {
        int32_t * p = (int32_t *) ids->data;
        for (int k = 0; k < N_USED; k++) {
            int e;
            bool dup;
            do {
                e = pick(rng);
                dup = false;
                for (int j = 0; j < k; j++) dup |= p[j] == e;
            } while (dup);
            p[k] = e;
        }
    };

    for (int it = 0; it < 20; it++) {  // warm-up: threads, page faults
        set_ids(layers[it % n_layers].ids);
        ggml_graph_compute_with_ctx(ctx, layers[it % n_layers].gf, n_threads);
    }
    double t_total = 0;
    for (int it = 0; it < iters; it++) {
        auto & L = layers[it % n_layers];
        set_ids(L.ids);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_graph_compute_with_ctx(ctx, L.gf, n_threads);
        t_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    const double us = t_total / iters * 1e6;
    const double bytes = 3.0 * N_USED * (double) mat_bytes / N_EXP;
    printf("threads %2d: %7.1f us per MoE layer (6 experts, %.1f MB) -> %5.1f GB/s\n",
           n_threads, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9);
    ggml_free(ctx);
    return 0;
}
