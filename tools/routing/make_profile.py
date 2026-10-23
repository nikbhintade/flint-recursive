"""Write data/routing/profile.txt: every (layer, expert) pair, most used first, from the T4 recording.

One line per pair: "<layer> <expert> <count>", with layer as llama.cpp's layer index (MoE layers start at 2).
Used to fill a static GPU expert cache (LLAMA_MOE_CACHE_PROFILE).

Usage: python3 tools/routing/make_profile.py
"""
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "routing"
E, K, N_DENSE = 256, 6, 2

ids = np.fromfile(DATA / "routes.ids", dtype=np.uint8)
n_moe = 47
ids = ids.reshape(-1, n_moe, K).astype(np.int64)
counts = np.bincount((np.arange(n_moe)[None, :, None] * E + ids).ravel(), minlength=n_moe * E)
order = np.argsort(-counts, kind="stable")
with (DATA / "profile.txt").open("w") as f:
    for key in order:
        f.write(f"{N_DENSE + key // E} {key % E} {counts[key]}\n")
print(f"{len(order)} pairs -> {DATA / 'profile.txt'}; top 6516 cover {counts[order[:6516]].sum() / counts.sum():.1%} of lookups")
