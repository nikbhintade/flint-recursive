"""Analyze Flint's recorded expert routing (T4): skew, locality, and GPU expert-cache hit rates.

Reads data/routing/routes.ids + meta.json (from replay.py). "Decode" means the generated tokens, the ones whose
speed we care about; their routing comes from the replay, which matches generation up to rounding.

Cache policies, all with a budget of N expert slots shared by the 47 MoE layers (one slot = one expert of one layer):
  layer-split  llama.cpp --n-cpu-moe: whole layers on the GPU, hit rate = N / 12,032 for any routing
  static       hottest (layer, expert) pairs from a profile; 10-fold: profile on 9/10 of the prompts, test on 1/10
  lru          starts from the static profile; on a miss the expert is copied in (CPU computes it this token),
               evicting the least recently used
  lfu-decay    starts from the static profile; every 16 tokens keep the N highest exponentially decayed counts
               (half-life 256 tokens), counts seeded from the profile
  prompt-warm  like lfu-decay, but before decoding the counts are reset from this prompt's own routing (known
               after prefill), mixed with the profile
Per-sequence simulation: every conversation starts from the profile.

Usage: python3 tools/routing/analyze.py [--budgets 3000,5000,7424,9000] [--routes routes]
"""
import argparse
import collections
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "routing"
E, K = 256, 6
ROUTES = "routes"


def load():
    meta = json.loads((DATA / "meta.json").read_text())
    ids = np.fromfile(DATA / f"{ROUTES}.ids", dtype=np.uint8)
    n_tok = sum(m["n_prompt"] + m["n_gen"] for m in meta)
    n_moe = ids.size // (n_tok * K)
    ids = ids.reshape(n_tok, n_moe, K)
    off = 0
    for m in meta:
        m["start"] = off
        m["gen_start"] = off + m["n_prompt"]
        off += m["n_prompt"] + m["n_gen"]
    return meta, ids


def counts(ids):
    """[n_moe, E] usage counts."""
    n_moe = ids.shape[1]
    flat = (np.arange(n_moe)[None, :, None] * E + ids.astype(np.int64)).ravel()
    return np.bincount(flat, minlength=n_moe * E).reshape(n_moe, E)


def static_hit(profile, test, budget):
    """Hit rate of a static cache holding the `budget` hottest (layer, expert) pairs of `profile` on `test` counts."""
    order = np.argsort(profile.ravel())[::-1][:budget]
    cached = np.zeros(profile.size, bool)
    cached[order] = True
    return test.ravel()[cached].sum() / test.sum()


def simulate(policy, dec, profile, budget, prompt_counts=None):
    """Hits, lookups and swaps for one sequence's decode tokens. dec: [T, n_moe, K] ids."""
    n_moe = dec.shape[1]
    keys = (np.arange(n_moe)[None, :, None] * E + dec.astype(np.int64)).reshape(len(dec), -1)
    prof = profile.ravel().astype(np.float64)
    prof = prof / prof.sum() * 1000.0  # profile as a prior worth ~1000 lookups
    hits = swaps = 0
    if policy == "lru":
        cache = collections.OrderedDict((k, None) for k in reversed(np.argsort(prof)[::-1][:budget].tolist()))
        for row in keys:
            for k in row.tolist():
                if k in cache:
                    hits += 1
                    cache.move_to_end(k)
                else:
                    cache[k] = None
                    swaps += 1
                    if len(cache) > budget:
                        cache.popitem(last=False)
        return hits, keys.size, swaps
    score = prof.copy()
    if policy == "prompt-warm" and prompt_counts is not None:
        pc = prompt_counts.ravel().astype(np.float64)
        score = 0.5 * prof + 0.5 * pc / max(pc.sum(), 1) * 1000.0
    cached = np.zeros(score.size, bool)
    cached[np.argsort(score)[::-1][:budget]] = True
    block = 16
    decay = 0.5 ** (block / 256)  # half-life 256 tokens, applied per 16-token block
    for t0 in range(0, len(keys), block):
        rows = keys[t0: t0 + block]
        hits += int(cached[rows].sum())
        score *= decay
        score += np.bincount(rows.ravel(), minlength=score.size)
        new = np.zeros_like(cached)
        new[np.argsort(score)[::-1][:budget]] = True
        swaps += int((new & ~cached).sum())
        cached = new
    return hits, keys.size, swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="3000,5000,7424,9000")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--routes", default="routes", help="file prefix in data/routing")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    global ROUTES
    ROUTES = args.routes

    meta, ids = load()
    n_moe = ids.shape[1]
    slots = n_moe * E
    gen_idx = np.concatenate([np.arange(m["gen_start"], m["gen_start"] + m["n_gen"]) for m in meta])
    print(f"{len(meta)} sequences, {len(ids)} tokens ({len(gen_idx)} generated), {n_moe} MoE layers, {slots} experts")

    # Sanity: 6 distinct experts per token and layer.
    srt = np.sort(ids, axis=2)
    dup = (srt[:, :, 1:] == srt[:, :, :-1]).any(axis=2).mean()
    print(f"duplicate expert ids within a token/layer: {dup:.2e}")

    # 1. Skew
    c_all, c_gen = counts(ids), counts(ids[gen_idx])
    print("\n## Skew (share of lookups served by each layer's hottest experts, averaged over layers)")
    for name, c in (("all tokens", c_all), ("generated", c_gen)):
        s = np.sort(c, axis=1)[:, ::-1]
        cum = s.cumsum(axis=1) / s.sum(axis=1, keepdims=True)
        cells = "  ".join(f"top {p:>2}%: {cum[:, int(E * p / 100) - 1].mean():.1%}" for p in (10, 25, 50))
        unused = (c == 0).sum()
        print(f"  {name:10}  {cells}   experts never used: {unused}")
    p = c_gen / c_gen.sum(axis=1, keepdims=True)
    ent = -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=1)
    print(f"  routing entropy (generated), bits: mean {ent.mean():.2f} of max {np.log2(E):.2f};"
          f" lowest layer {2 + ent.argmin()} ({ent.min():.2f}), highest {2 + ent.argmax()} ({ent.max():.2f})")

    # 2. Temporal locality on generated tokens
    print("\n## Locality (generated tokens): share of a token's experts used in the previous w tokens, same layer")
    for w in (1, 4, 16, 64):
        reuse = []
        for m in meta:
            d = ids[m["gen_start"]: m["gen_start"] + m["n_gen"]].astype(np.int64)
            if len(d) <= w:
                continue
            onehot = np.zeros((len(d), n_moe, E), bool)
            np.put_along_axis(onehot, d, True, axis=2)
            csum = np.concatenate([np.zeros((1, n_moe, E), np.int32), onehot.cumsum(axis=0, dtype=np.int32)])
            window = csum[w:-1] - csum[: len(d) - w]  # uses in tokens [t-w, t)
            used = np.take_along_axis(window, d[w:], axis=2) > 0
            reuse.append(used.mean())
        print(f"  w={w:>3}: {np.mean(reuse):.1%}")

    # 3. Cache hit rates (decode), k-fold over sequences
    print(f"\n## Decode hit rate vs. GPU expert slots ({args.folds}-fold; layer-split = llama.cpp --n-cpu-moe)")
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(meta)) % args.folds
    policies = ["static", "lru", "lfu-decay", "prompt-warm"]
    res = {(pol, b): [0, 0, 0] for pol in policies for b in budgets}
    oracle = {b: [0, 0] for b in budgets}
    by_kind = collections.defaultdict(lambda: [0, 0])
    for f in range(args.folds):
        train = [m for i, m in enumerate(meta) if fold[i] != f]
        test = [m for i, m in enumerate(meta) if fold[i] == f]
        profile = counts(np.concatenate([ids[m["start"]: m["start"] + m["n_prompt"] + m["n_gen"]] for m in train]))
        for m in test:
            dec = ids[m["gen_start"]: m["gen_start"] + m["n_gen"]]
            pc = counts(ids[m["start"]: m["gen_start"]])
            for b in budgets:
                for pol in policies:
                    if pol == "static":
                        h = static_hit(profile, counts(dec), b) * dec[:, :, :].size
                        r = (h, dec.size, 0)
                    else:
                        r = simulate(pol, dec, profile, b, pc)
                    acc = res[(pol, b)]
                    acc[0] += r[0]; acc[1] += r[1]; acc[2] += r[2]
                    if pol == "prompt-warm" and b == 7424:
                        by_kind[m["kind"]][0] += r[0]; by_kind[m["kind"]][1] += r[1]
                oracle[b][0] += static_hit(counts(dec), counts(dec), b) * dec.size
                oracle[b][1] += dec.size
    header = "  policy        " + "".join(f"{b:>10}" for b in budgets)
    print(header)
    print("  layer-split   " + "".join(f"{b / slots:>10.1%}" for b in budgets))
    for pol in policies:
        print(f"  {pol:13} " + "".join(f"{res[(pol, b)][0] / res[(pol, b)][1]:>10.1%}" for b in budgets))
    print("  oracle*       " + "".join(f"{oracle[b][0] / oracle[b][1]:>10.1%}" for b in budgets))
    print("  * best fixed set chosen with hindsight per conversation (upper bound for static caching)")
    tok = len(gen_idx)
    print("\n  swaps per generated token (PCIe copies of 1.77 MB):")
    for pol in ("lru", "lfu-decay", "prompt-warm"):
        print(f"    {pol:13} " + "".join(f"{res[(pol, b)][2] / tok:>10.2f}" for b in budgets))
    print("\n  prompt-warm @ 7424 slots by prompt kind: " +
          "  ".join(f"{k} {v[0] / v[1]:.1%}" for k, v in sorted(by_kind.items())))

    # 4. What it means for the CPU
    print("\n## CPU expert work per generated token at 7424 slots (282 lookups x 1.77 MB)")
    for pol in ["layer-split"] + policies:
        hit = 7424 / slots if pol == "layer-split" else res[(pol, 7424)][0] / res[(pol, 7424)][1]
        miss = (1 - hit) * n_moe * K
        print(f"  {pol:13} {miss:6.1f} experts  = {miss * 1.77:6.1f} MB from RAM")


if __name__ == "__main__":
    main()
