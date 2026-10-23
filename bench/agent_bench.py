"""Multi-turn agent benchmark: conversations that grow by a tool result each turn, like a coding agent.

Each episode starts from one T4 agent prompt (system, issue, a tool call that read a file, its output). Every turn
Flint answers (greedy, up to --max-tokens); its tool call, or a nudge if it made none, is followed by a tool result of
300-1,500 tokens of real source code. Prompt caching is on, so a turn should only process the new tokens.

Usage: python3 bench/agent_bench.py [--episodes 4] [--turns 6] [--max-tokens 256]
Prints per turn: tokens reused from the cache, tokens processed, prompt and generation time; then the total wall time.
Needs llama-server on :8080 with --jinja.
"""
import argparse
import json
import pathlib
import random
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "llama.cpp" / "src"


def post(body):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def tool_outputs(seed):
    """Endless excerpts of real source files, 1,000-5,000 characters (~300-1,500 tokens)."""
    rng = random.Random(seed)
    files = sorted(p for p in SRC.rglob("*.cpp") if p.stat().st_size > 20_000)
    while True:
        text = rng.choice(files).read_text(errors="replace")
        n = rng.randint(1_000, 5_000)
        start = rng.randrange(0, len(text) - n)
        yield text[start:start + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / "data/routing/prompts.jsonl").open()]
    agents = [r for r in rows if r["kind"] == "agent"][: args.episodes]
    outputs = tool_outputs(0)

    post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4})  # warm-up
    t0 = time.time()
    tot = {"cached": 0, "processed": 0, "prompt_ms": 0.0, "gen": 0, "gen_ms": 0.0}
    print(f"{'ep':>2} {'turn':>4} {'cached':>7} {'processed':>9} {'prompt ms':>9} {'gen':>4} {'gen ms':>7}")
    for ep, row in enumerate(agents):
        messages = list(row["messages"])
        for turn in range(args.turns):
            r = post({"messages": messages, "tools": row["tools"], "max_tokens": args.max_tokens, "temperature": 0,
                      "cache_prompt": True})
            t = r["timings"]
            msg = r["choices"][0]["message"]
            cached, processed = t.get("cache_n", 0), t["prompt_n"]
            tot["cached"] += cached
            tot["processed"] += processed
            tot["prompt_ms"] += t["prompt_ms"]
            tot["gen"] += t["predicted_n"]
            tot["gen_ms"] += t["predicted_ms"]
            print(f"{ep:>2} {turn:>4} {cached:>7} {processed:>9} {t['prompt_ms']:>9.0f} {t['predicted_n']:>4} "
                  f"{t['predicted_ms']:>7.0f}", flush=True)

            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                assistant["reasoning_content"] = msg["reasoning_content"]
            calls = msg.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if calls:
                for c in calls:
                    messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": next(outputs)})
            else:
                messages.append({"role": "user", "content": "Here is more of the code:\n" + next(outputs)})
    wall = time.time() - t0
    print(f"\nwall {wall:.1f} s | prompt: {tot['processed']} tokens processed in {tot['prompt_ms'] / 1000:.1f} s, "
          f"{tot['cached']} reused | generation: {tot['gen']} tokens in {tot['gen_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
