"""Train an EAGLE-3 draft head for Flint on features from tools/t9/features.py, export it for llama.cpp.

The head mirrors llama.cpp's src/models/eagle3.cpp: fc (3 x 2048 -> 2048) fuses the inputs of Flint layers 2, 24,
46; one decoder layer whose attention reads [norm(token embedding), norm(features)] with the raw features as
residual; an output head over a 32K draft vocabulary (d2t maps it back to Flint's). Token embeddings are Flint's
own, frozen. HF-style rotate-half RoPE: llama.cpp's converter permutes q/k for its own RoPE layout.

Training-time test (EAGLE-3): the head is unrolled for --ttt steps. Step 0 sees Flint's features; step k sees its
own outputs from step k-1. At step k, chain i (drafting from position i) attends to the step-0 entries of
positions <= i and to its own entries of steps 1..k, at RoPE positions i+k - as when llama.cpp drafts.
Targets: Flint's next-token distribution from the stored final hidden state and its output layer (dequantized
NVFP4), restricted to the draft vocabulary. Loss only where the predicted token is part of an answer.

Usage: .venv-t9/bin/python tools/t9/train.py [--dir data/t9] [--features-dir <dir>] [--epochs 3] [--ttt 7] [--lr 2e-4]
       [--out <dir>/head]
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "t9"
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

H, N_HEAD, N_KV, D_HEAD, FF = 2048, 16, 4, 128, 6144
EPS, ROPE_THETA = 1e-6, 10000.0
TARGET_LAYERS = [2, 24, 46]


def load_target(device):
    """Flint's token embeddings (bf16) and output layer (NVFP4 dequantized, times its per-tensor scale)."""
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(str(ROOT / "flint-nvfp4-gguf" / "flint-nvfp4.gguf"))
    t = {x.name: x for x in r.tensors}
    emb = torch.from_numpy(np.array(t["token_embd.weight"].data).view(np.int16).copy()).view(torch.bfloat16)
    out_w = t["output.weight"]
    scale = float(np.array(t["output.scale"].data).ravel()[0])
    rows = []
    for i in range(0, out_w.data.shape[0], 8192):
        rows.append(torch.from_numpy(dequantize(np.array(out_w.data[i:i + 8192]), out_w.tensor_type) * scale).to(torch.bfloat16))
    return emb.reshape(-1, H).to(device), torch.cat(rows).to(device)


class RMSNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * self.weight.to(x.dtype)


def rope(x, pos):
    """HF rotate-half RoPE. x [T, heads, D], pos [T]."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, D_HEAD, 2, device=x.device).float() / D_HEAD))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1)[:, None, :], torch.cat([ang.sin()] * 2, -1)[:, None, :]
    x1, x2 = x[..., : D_HEAD // 2], x[..., D_HEAD // 2:]
    return (x.float() * cos + torch.cat([-x2, x1], -1).float() * sin).to(x.dtype)


class Head(nn.Module):
    def __init__(self, n_draft):
        super().__init__()
        self.fc = nn.Linear(3 * H, H, bias=False)
        self.input_layernorm = RMSNorm(H)
        self.hidden_norm = RMSNorm(H)
        self.q_proj = nn.Linear(2 * H, N_HEAD * D_HEAD, bias=False)
        self.k_proj = nn.Linear(2 * H, N_KV * D_HEAD, bias=False)
        self.v_proj = nn.Linear(2 * H, N_KV * D_HEAD, bias=False)
        self.o_proj = nn.Linear(N_HEAD * D_HEAD, H, bias=False)
        self.post_attention_layernorm = RMSNorm(H)
        self.gate_proj = nn.Linear(H, FF, bias=False)
        self.up_proj = nn.Linear(H, FF, bias=False)
        self.down_proj = nn.Linear(FF, H, bias=False)
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, n_draft, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def step(self, emb, g, k, cache):
        """One unrolled step. emb, g [T, H]; cache holds (keys, values) of earlier steps. Returns (h, logits)."""
        T = emb.shape[0]
        x = torch.cat([self.input_layernorm(emb), self.hidden_norm(g)], -1)
        pos = torch.arange(T, device=emb.device) + k
        q = rope(self.q_proj(x).view(T, N_HEAD, D_HEAD), pos)
        key = rope(self.k_proj(x).view(T, N_KV, D_HEAD), pos)
        val = self.v_proj(x).view(T, N_KV, D_HEAD)
        rep = N_HEAD // N_KV
        key, val = key.repeat_interleave(rep, 1), val.repeat_interleave(rep, 1)
        if k == 0:
            cache.clear()
        cache.append((key, val))
        qh = q.transpose(0, 1).float()                                        # [h, T, D]
        k0, v0 = cache[0][0].transpose(0, 1).float(), cache[0][1].transpose(0, 1).float()
        s0 = qh @ k0.transpose(1, 2) / math.sqrt(D_HEAD)                      # [h, T, T], causal over step 0
        s0 = s0.masked_fill(torch.ones(T, T, device=emb.device, dtype=torch.bool).triu(1), float("-inf"))
        diag = [(qh * kj.transpose(0, 1).float()).sum(-1) / math.sqrt(D_HEAD) for kj, _ in cache[1:]]  # [h, T] each
        s = torch.cat([s0] + [d[..., None] for d in diag], -1)
        p = torch.softmax(s, -1)
        out = p[..., :T] @ v0
        for j, (_, vj) in enumerate(cache[1:]):
            out = out + p[..., T + j: T + j + 1] * vj.transpose(0, 1).float()
        attn = self.o_proj(out.transpose(0, 1).reshape(T, N_HEAD * D_HEAD).to(x.dtype))
        h = attn + g
        h = h + self.down_proj(F.silu(self.gate_proj(self.post_attention_layernorm(h))) * self.up_proj(self.post_attention_layernorm(h)))
        return h, self.lm_head(self.norm(h))


def export(head, d2t, t2d, out_dir):
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in head.state_dict().items()}
    names = {"fc.weight": "fc.weight", "norm.weight": "norm.weight", "lm_head.weight": "lm_head.weight",
             "input_layernorm.weight": "midlayer.input_layernorm.weight", "hidden_norm.weight": "midlayer.hidden_norm.weight",
             "post_attention_layernorm.weight": "midlayer.post_attention_layernorm.weight"}
    for p in ("q", "k", "v", "o"):
        names[f"{p}_proj.weight"] = f"midlayer.self_attn.{p}_proj.weight"
    for p in ("gate", "up", "down"):
        names[f"{p}_proj.weight"] = f"midlayer.mlp.{p}_proj.weight"
    tensors = {names[k]: v for k, v in sd.items()}
    tensors["d2t"] = d2t.cpu().contiguous()
    tensors["t2d"] = t2d.cpu().contiguous()
    save_file(tensors, str(out_dir / "model.safetensors"))
    cfg = {"architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama", "hidden_size": H, "intermediate_size": FF,
           "num_hidden_layers": 1, "num_attention_heads": N_HEAD, "num_key_value_heads": N_KV, "head_dim": D_HEAD,
           "rms_norm_eps": EPS, "rope_theta": ROPE_THETA, "max_position_embeddings": 262144, "vocab_size": int(t2d.numel()),
           "draft_vocab_size": int(d2t.numel()), "hidden_act": "silu", "tie_word_embeddings": False,
           "eagle_aux_hidden_state_layer_ids": TARGET_LAYERS, "target_hidden_size": H, "torch_dtype": "bfloat16",
           "bos_token_id": 150000, "eos_token_id": 150001}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--ttt", type=int, default=7)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tokens-per-step", type=int, default=8192)
    ap.add_argument("--draft-vocab", type=int, default=32000)
    ap.add_argument("--dir", default=str(DATA))
    ap.add_argument("--features-dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    random.seed(0)

    data = pathlib.Path(args.dir)
    args.out = args.out or str(data / "head")
    meta = json.loads((data / "meta.json").read_text())
    seqs, tokens = meta["seqs"], meta["tokens"]
    n_tok = sum(s["n_tokens"] for s in seqs)
    feats = np.memmap(pathlib.Path(args.features_dir or data) / "features.bf16", dtype=np.int16, mode="r").reshape(n_tok, 4, H)

    # draft vocabulary: the most frequent tokens; d2t[i] = target id - i (SpecForge convention)
    counts = np.bincount(np.concatenate([np.array(t) for t in tokens]), minlength=150144)
    draft_ids = torch.from_numpy(np.sort(np.argsort(-counts, kind="stable")[: args.draft_vocab]))
    d2t = draft_ids - torch.arange(args.draft_vocab)
    t2d = torch.zeros(150144, dtype=torch.bool)
    t2d[draft_ids] = True
    covered = counts[draft_ids.numpy()].sum() / counts.sum()
    print(f"{len(seqs)} sequences, {n_tok} tokens; draft vocab {args.draft_vocab} covers {covered:.2%} of tokens", flush=True)

    emb, w_out = load_target(dev)
    w_draft = w_out[draft_ids.to(dev)]            # target logits restricted to the draft vocabulary
    del w_out
    head = Head(args.draft_vocab).to(dev)
    print(f"head: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M parameters", flush=True)

    rng = random.Random(0)
    order = list(range(len(seqs)))
    rng.shuffle(order)
    n_val = max(8, len(order) // 20)
    val, train = order[:n_val], order[n_val:]
    train_tokens = sum(seqs[i]["n_tokens"] for i in train)
    total_steps = int(args.epochs * train_tokens / args.tokens_per_step)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / max(total_steps, 1)))))
    weights = [0.8 ** k for k in range(args.ttt)]

    def run(i, train_mode):
        s = seqs[i]
        L, off, n_prompt = s["n_tokens"], s["offset"], s["n_prompt"]
        x = torch.tensor(tokens[i], device=dev)
        f = torch.from_numpy(np.array(feats[off: off + L])).to(dev).view(torch.bfloat16)   # [L, 4, H]
        with torch.no_grad():
            tgt = torch.softmax((f[:, 3] @ w_draft.T).float(), -1)                    # [L, V]: dist of token j+1
        T = L - 1                                                                    # chains i = 0 .. L-2
        g = head.fc(f[:T, :3].reshape(T, 3 * H))
        cache, loss, stats = [], 0.0, []
        for k in range(args.ttt):
            n = T - k                                   # chains with a step-k target inside the sequence
            if n <= 1:
                break
            e = emb[x[1 + k: 1 + k + T]] if 1 + k + T <= L else torch.cat([emb[x[1 + k:]], emb[x[-1:]].expand(1 + k + T - L, H)])
            g, logits = head.step(e, g, k, cache)
            t = tgt[1 + k: 1 + k + n]                    # target for chain i at step k: dist at position i+1+k
            pred_pos = torch.arange(n, device=dev) + 2 + k
            mask = (pred_pos >= n_prompt).float()          # predicted token is part of the answer
            lp = torch.log_softmax(logits[:n].float(), -1)
            ce = -(t * lp).sum(-1)
            lk = (ce * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + weights[k] * lk
            acc = ((logits[:n].argmax(-1) == t.argmax(-1)).float() * mask).sum() / mask.sum().clamp(min=1)
            stats.append((lk.item(), acc.item(), mask.sum().item()))
        return loss, stats

    step, t0, acc_tok = 0, time.time(), 0
    epoch_order = []
    while step < total_steps:
        if not epoch_order:
            epoch_order = train[:]
            rng.shuffle(epoch_order)
        head.train()
        i = epoch_order.pop()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, stats = run(i, True)
        (loss * seqs[i]["n_tokens"] / args.tokens_per_step).backward()
        acc_tok += seqs[i]["n_tokens"]
        if acc_tok >= args.tokens_per_step:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc_tok = 0
            step += 1
            if step % 25 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {loss.item():.3f} acc@step " +
                      " ".join(f"{a:.2f}" for _, a, _ in stats) +
                      f"  {(time.time() - t0) / 60:.1f} min", flush=True)
            if step % 200 == 0 or step == total_steps:
                head.eval()
                tot = np.zeros((args.ttt, 2))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for j in val:
                        _, st = run(j, False)
                        for k, (_, a, m) in enumerate(st):
                            tot[k] += (a * m, m)
                acc = tot[:, 0] / np.maximum(tot[:, 1], 1)
                # expected accepted drafts per step if the k-th draft is accepted only after all earlier ones
                exp_len = sum(np.prod(acc[: k + 1]) for k in range(args.ttt))
                print(f"  val top-1 per step: {' '.join(f'{a:.3f}' for a in acc)}  -> ~{exp_len:.2f} accepted drafts per step",
                      flush=True)
                export(head, d2t, t2d, pathlib.Path(args.out))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
