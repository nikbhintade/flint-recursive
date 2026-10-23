"""Training prompts for the EAGLE-3 draft head (T9): coding instructions and agent turns, none from the eval sets.

Writes data/t9/prompts.jsonl (id, kind, messages, optional tools). Sources:
  magicoder  ise-uiuc/Magicoder-OSS-Instruct-75K   ("problem", many languages)
  ossinst    bigcode/self-oss-instruct-sc2-exec-filter-50k ("instruction")
  oci        nvidia/OpenCodeInstruct                ("input")
  agent      SWE-bench Verified rows 100-499: issue -> earlier tool call that read a file -> excerpt -> next action
  issue      SWE-bench Verified rows 100-499: explain the root cause and outline a fix
Rows 0-99 of SWE-bench Verified are left out: the T4 routing set and the agent benchmarks come from them.
The full run (--full) takes its agent and issue prompts from SWE-bench's train split instead (other repositories,
no overlap with Verified), uses another seed and skips every text already in the pilot set.

Usage: python3 tools/t9/make_prompts.py [--scale 1.0]                    -> data/t9/prompts.jsonl (pilot)
       python3 tools/t9/make_prompts.py --full --scale 6.5             -> data/t9/full/prompts.jsonl
"""
import argparse
import json
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "routing"))
from make_prompts import AGENT_SYSTEM, TOOLS, preimage  # noqa: E402

DATA = ROOT / "data" / "t9"


def rows(dataset, config, split, offset, length):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == 7:
                raise
            time.sleep(10 * 2 ** attempt)  # rate limited


def sample(dataset, config, split, total, n, rng, field, skip):
    """n distinct rows from pages at random offsets, keeping texts of a sensible length."""
    out, seen = [], set(skip)
    while len(out) < n:
        page = rows(dataset, config, split, rng.randrange(0, total - 100), 100)
        for r in page:
            text = r.get(field)
            if text and 40 < len(text) < 6000 and text not in seen:
                seen.add(text)
                out.append(text)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--skip", nargs="*", default=[], help="more prompt files whose texts to leave out")
    ap.add_argument("--out")
    ap.add_argument("--id-prefix", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed if args.full else 9)
    out_path = pathlib.Path(args.out) if args.out else DATA / "full" / "prompts.jsonl" if args.full else DATA / "prompts.jsonl"
    skip = set()
    if args.full:  # nothing from the pilot set (or the given sets)
        for path in [DATA / "prompts.jsonl"] + [pathlib.Path(s) for s in args.skip]:
            for line in path.open():
                skip.update(m["content"] for m in json.loads(line)["messages"] if m.get("content"))
    n = lambda k: int(k * args.scale)  # noqa: E731
    out = []

    for name, ds, cfg, total, k, field in (
        ("magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", "default", 75197, 900, "problem"),
        ("ossinst", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", 50661, 500, "instruction"),
        ("oci", "nvidia/OpenCodeInstruct", "train", 1400000, 500, "input"),
    ):
        for i, text in enumerate(sample(ds, cfg, "train", total, n(k), rng, field, skip)):
            out.append({"id": f"{args.id_prefix}{name}-{i:04d}", "kind": name, "messages": [{"role": "user", "content": text}]})
        print(f"{name}: {n(k)}", flush=True)

    swe = []
    if args.full:
        for off in rng.sample(range(0, 19000, 100), (n(250) + n(150)) // 100 + 2):
            swe += [r for r in rows("princeton-nlp/SWE-bench", "default", "train", off, 100) if r.get("patch") and
                    f"Fix this issue in {r['repo']}:\n\n{r['problem_statement'][:5000]}" not in skip]
    else:
        for off in range(100, 500, 100):
            swe += rows("princeton-nlp/SWE-bench_Verified", "default", "test", off, 100)
    rng.shuffle(swe)
    for i, row in enumerate(swe[: n(250)]):
        path, excerpt = preimage(row["patch"])
        out.append({"id": f"{args.id_prefix}agent-{i:04d}", "kind": "agent", "tools": TOOLS, "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"Fix this issue in {row['repo']}:\n\n{row['problem_statement'][:5000]}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": f"sed -n '1,400p' {path}"})}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": excerpt[:12000]},
        ]})
    for i, row in enumerate(swe[n(250): n(250) + n(150)]):
        out.append({"id": f"{args.id_prefix}issue-{i:04d}", "kind": "issue", "messages": [{"role": "user", "content":
                    f"Repository: {row['repo']}\n\n{row['problem_statement'][:6000]}\n\n"
                    "Explain the most likely root cause and outline the fix."}]})

    rng.shuffle(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    kinds = {}
    for r in out:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{len(out)} prompts -> {out_path}  {kinds}")


if __name__ == "__main__":
    main()
