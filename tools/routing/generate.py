"""Answer every T4 prompt with llama-server and save the exact token sequences for the routing replay.

Needs llama-server on :8080 (with --jinja). For each prompt: render the chat template (with tools), tokenize it,
generate with Flint's recommended sampling, and keep the generated token ids. Writes data/routing/sequences.jsonl
with prompt_tokens, gen_tokens and timings. Resumes: prompts already in the output are skipped.

Usage: python3 tools/routing/generate.py [--max-tokens 1024]
"""
import argparse
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "routing"
URL = "http://127.0.0.1:8080"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    out_path = DATA / "sequences.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open()}
    prompts = [json.loads(line) for line in (DATA / "prompts.jsonl").open()]

    with out_path.open("a") as out:
        for i, p in enumerate(prompts):
            if p["id"] in done:
                continue
            body = {"messages": p["messages"]}
            if "tools" in p:
                body["tools"] = p["tools"]
            prompt = post("/apply-template", body)["prompt"]
            prompt_tokens = post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
            t0 = time.time()
            r = post("/completion", {
                "prompt": prompt_tokens, "n_predict": args.max_tokens, "return_tokens": True, "cache_prompt": False,
                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": i,
            })
            t = r["timings"]
            row = {"id": p["id"], "kind": p["kind"], "prompt_tokens": prompt_tokens, "gen_tokens": r["tokens"],
                   "stop": r.get("stop_type"), "text": r["content"],
                   "prompt_tps": t["prompt_per_second"], "gen_tps": t["predicted_per_second"]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i + 1:3}/{len(prompts)}] {p['id']:13} prompt {len(prompt_tokens):5}  gen {len(r['tokens']):4}  "
                  f"{t['prompt_per_second']:6.0f} / {t['predicted_per_second']:5.1f} tok/s  {time.time() - t0:5.1f} s",
                  flush=True)


if __name__ == "__main__":
    main()
