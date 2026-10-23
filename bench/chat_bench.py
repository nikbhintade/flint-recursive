"""Time Flint on real coding prompts through llama-server or Ollama.

Usage: python3 bench/chat_bench.py llamacpp|ollama [--runs N] [--max-tokens N]

Prompts: a short question, and two "explain/fix this file" prompts built from llama.cpp source files
(~2K and ~8K tokens). Prints prompt tokens, prompt tok/s and generation tok/s as each server reports them.
"""
import argparse
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "llama.cpp" / "src"


def file_prompt(paths, max_chars):
    text = ""
    for p in paths:
        text += f"\n\n### {p.name}\n```cpp\n{p.read_text()}\n```"
        if len(text) > max_chars:
            break
    return ("Here is part of a C++ inference engine." + text[:max_chars] +
            "\n\nExplain what this code does, then point out one bug or inefficiency and show a fix.")


PROMPTS = {
    "short": "Write a Python function that merges overlapping intervals, with a short docstring and two tests.",
    "file-2k": file_prompt([SRC / "models" / "flint.cpp", SRC / "llama-graph.cpp"], 7_000),
    "file-8k": file_prompt([SRC / "llama-graph.cpp", SRC / "llama-model.cpp"], 28_000),
}


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def run_llamacpp(prompt, max_tokens):
    r = post("http://127.0.0.1:8080/v1/chat/completions", {
        "model": "flint", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "cache_prompt": False,
    })
    t = r["timings"]
    return t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]


def run_ollama(prompt, max_tokens):
    r = post("http://127.0.0.1:11435/api/chat", {
        "model": "flint:latest", "messages": [{"role": "user", "content": prompt}], "stream": False,
        "options": {"num_predict": max_tokens, "num_ctx": 32768},
    })
    pn, pd = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 1)
    en, ed = r.get("eval_count", 0), r.get("eval_duration", 1)
    return pn, pn / pd * 1e9, en, en / ed * 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["llamacpp", "ollama"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    run = run_llamacpp if args.engine == "llamacpp" else run_ollama

    run(PROMPTS["short"], 16)  # warm-up: load weights, fill caches
    print(f"{'prompt':8} {'run':>3} {'prompt tok':>10} {'prompt tok/s':>12} {'gen tok':>8} {'gen tok/s':>9}")
    for name, prompt in PROMPTS.items():
        for i in range(args.runs):
            t0 = time.time()
            pn, pps, gn, gps = run(prompt, args.max_tokens)
            print(f"{name:8} {i + 1:>3} {pn:>10} {pps:>12.1f} {gn:>8} {gps:>9.1f}   ({time.time() - t0:.1f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
