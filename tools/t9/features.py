"""Replay Flint's answers through our engine and store the features the EAGLE-3 head trains on.

Reads data/t9/sequences.jsonl (tools/t9/generate.py), writes data/t9/seqs.txt (feature_dump input),
data/t9/meta.json (per sequence: id, kind, n_prompt, n_tokens, offset in tokens) and runs feature_dump, which writes
data/t9/features.bf16: [token][4][2048] = inputs of layers 2, 24, 46 and the final normed hidden state.
Needs the GPU (stop the server first).

Usage: python3 tools/t9/features.py [--dir data/t9] [--features-dir <dir>] [--max-len 2048] [--cache-mib 12000]
(--features-dir: where features.bf16 goes, e.g. the external SSD for the full run; default --dir)
"""
import argparse
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "t9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--cache-mib", type=int, default=12000)
    ap.add_argument("--dir", default=str(DATA))
    ap.add_argument("--features-dir")
    args = ap.parse_args()
    data = pathlib.Path(args.dir)
    feat_dir = pathlib.Path(args.features_dir or args.dir)

    meta, off = [], 0
    with (data / "seqs.txt").open("w") as f:
        for line in (data / "sequences.jsonl").open():
            r = json.loads(line)
            toks = (r["prompt_tokens"] + r["gen_tokens"])[: args.max_len]
            if len(toks) <= len(r["prompt_tokens"]) + 1:
                continue  # nothing generated to learn from after the cut
            f.write(" ".join(map(str, [len(r["prompt_tokens"])] + toks)) + "\n")
            meta.append({"id": r["id"], "kind": r["kind"], "n_prompt": len(r["prompt_tokens"]), "n_tokens": len(toks),
                         "offset": off})
            off += len(toks)
    tokens = []
    for line in (data / "seqs.txt").open():
        tokens.append(list(map(int, line.split()[1:])))
    (data / "meta.json").write_text(json.dumps({"tokens": tokens, "seqs": meta}))
    print(f"{len(meta)} sequences, {off} tokens, {off * 4 * 2048 * 2 / 1e9:.1f} GB of features", flush=True)

    env = dict(os.environ, LLAMA_MOE_CACHE_PROFILE=str(ROOT / "profiles" / "flint-coding.txt"))
    subprocess.run(["taskset", "-c", "0-7,10-15", str(ROOT / "tools/t9/build/feature_dump"),
                    str(ROOT / "flint-nvfp4-gguf/flint-nvfp4.gguf"), str(data / "seqs.txt"), str(feat_dir / "features"),
                    str(args.cache_mib), str(args.max_len)], check=True, env=env)


if __name__ == "__main__":
    main()
