"""Turn data/routing/sequences.jsonl into the route_dump input and run it.

Writes data/routing/sequences.txt (one sequence per line: prompt length, then the prompt + generated token ids) and
data/routing/meta.json (id, kind, prompt and generated lengths per sequence, in file order), then runs
route_dump to produce data/routing/routes.ids / routes.w.

Usage: python3 tools/routing/replay.py [--n-cpu-moe 24] [--ubatch 1024]
"""
import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "routing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cpu-moe", type=int, default=24)
    ap.add_argument("--ubatch", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(line) for line in (DATA / "sequences.jsonl").open()]
    meta = []
    with (DATA / "sequences.txt").open("w") as f:
        for r in rows:
            f.write(" ".join(map(str, [len(r["prompt_tokens"])] + r["prompt_tokens"] + r["gen_tokens"])) + "\n")
            meta.append({"id": r["id"], "kind": r["kind"], "n_prompt": len(r["prompt_tokens"]),
                         "n_gen": len(r["gen_tokens"])})
    (DATA / "meta.json").write_text(json.dumps(meta))
    print(f"{len(meta)} sequences, {sum(m['n_prompt'] for m in meta)} prompt + {sum(m['n_gen'] for m in meta)} "
          "generated tokens")

    subprocess.run(["taskset", "-c", "0-7,10-15", str(ROOT / "tools/routing/build/route_dump"),
                    str(ROOT / "flint-nvfp4-gguf/flint-nvfp4.gguf"), str(DATA / "sequences.txt"),
                    str(DATA / "routes"), str(args.n_cpu_moe), str(args.ubatch)], check=True)


if __name__ == "__main__":
    main()
