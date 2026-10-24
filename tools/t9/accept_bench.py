"""Draft acceptance and generation speed on held-out prompts (never answered, so never trained on).

Needs run-flint.sh on :8080 (with whatever SPEC setting is being measured). Takes prompts from <dir>/prompts.jsonl
that have no answer in <dir>/sequences.jsonl, Flint's sampling (or --temp 0), one request at a time.

Usage: python3 tools/t9/accept_bench.py [--dir data/t9/full] [--n 12] [--max-tokens 384] [--temp 1.0]
"""
import argparse
import json
import pathlib
import random
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]


def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8080" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "t9" / "full"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    done = {json.loads(line)["id"] for line in (d / "sequences.jsonl").open()}
    held = [json.loads(line) for line in (d / "prompts.jsonl").open()]
    held = [p for p in held if p["id"] not in done]
    random.Random(7).shuffle(held)
    acc = drafted = gen = 0
    secs = 0.0
    for p in held[: args.n]:
        body = {"messages": p["messages"]}
        if "tools" in p:
            body["tools"] = p["tools"]
        prompt = post("/apply-template", body)["prompt"]
        r = post("/completion", {"prompt": prompt, "n_predict": args.max_tokens, "temperature": args.temp, "top_p": 0.95,
                                 "top_k": 20, "min_p": 0.0, "seed": 1, "cache_prompt": False})
        t = r["timings"]
        acc += t.get("draft_n_accepted", 0)
        drafted += t.get("draft_n", 0)
        gen += t["predicted_n"]
        secs += t["predicted_ms"] / 1000
        print(f"{p['kind']:10s} {t['predicted_n']:4d} tokens {t['predicted_per_second']:6.1f} tok/s  "
              f"drafts {t.get('draft_n_accepted', 0)}/{t.get('draft_n', 0)}", flush=True)
    print(f"total: {gen / secs:.1f} tok/s, {gen} tokens; drafts accepted {acc}/{drafted} "
          f"({acc / max(drafted, 1):.1%})")


if __name__ == "__main__":
    main()
