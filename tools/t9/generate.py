"""Flint answers the T9 training prompts (self-distillation): exact token ids of prompt and answer.

Needs run-flint.sh on :8080 with PARALLEL >= --workers. Reads <dir>/prompts.jsonl, writes <dir>/sequences.jsonl and
resumes from it (dir: data/t9 for the pilot, data/t9/full for the full run).
Sampling is Flint's recommended one (temperature 1.0, top-p 0.95, top-k 20), so the head learns the text Flint
really produces.

--deadline (unix time): no new requests after it; answers already running finish and are kept.

Usage: python3 tools/t9/generate.py [--dir data/t9] [--workers 4] [--max-tokens 1536] [--deadline <unix time>]
"""
import argparse
import concurrent.futures
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "t9"
URL = "http://127.0.0.1:8080"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def answer(p, i, max_tokens, deadline):
    if time.time() > deadline:
        return None
    body = {"messages": p["messages"]}
    if "tools" in p:
        body["tools"] = p["tools"]
    try:
        prompt = post("/apply-template", body)["prompt"]
        prompt_tokens = post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
        r = post("/completion", {
            "prompt": prompt_tokens, "n_predict": max_tokens, "return_tokens": True, "cache_prompt": False,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 1000 + i,
        })
    except urllib.error.HTTPError as e:  # e.g. a prompt longer than a slot: skip it, keep going
        print(f"skipped {p['id']}: HTTP {e.code}", flush=True)
        return None
    return {"id": p["id"], "kind": p["kind"], "prompt_tokens": prompt_tokens, "gen_tokens": r["tokens"],
            "stop": r.get("stop_type")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--dir", default=str(DATA))
    ap.add_argument("--deadline", type=float, default=float("inf"))
    args = ap.parse_args()
    data = pathlib.Path(args.dir)

    out_path = data / "sequences.jsonl"
    done = {json.loads(line)["id"] for line in out_path.open()} if out_path.exists() else set()
    prompts = [(i, json.loads(line)) for i, line in enumerate((data / "prompts.jsonl").open())]
    todo = [(i, p) for i, p in prompts if p["id"] not in done]
    lock = threading.Lock()
    t0, n_gen, n_done = time.time(), 0, 0

    with out_path.open("a") as out, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(answer, p, i, args.max_tokens, args.deadline) for i, p in todo]
        for f in concurrent.futures.as_completed(futures):
            row = f.result()
            if row is None:
                continue
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_gen += len(row["gen_tokens"])
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    dt = time.time() - t0
                    print(f"[{len(done) + n_done}/{len(prompts)}] {n_gen} tokens generated, {n_gen / dt:.0f} tok/s, "
                          f"ETA {(len(todo) - n_done) * dt / n_done / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
